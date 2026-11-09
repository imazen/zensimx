# Shaping tails and training reliability, November 9, 2026

The owner wants reusable input-shaping defaults that reduce outliers and scatter
sprawl and make good training runs easier to achieve and reason about. The
existing learned-bank results answer only part of that question.

This descriptive analysis uses all 72 matched partition/setting/seed conditions
for each of ten neural primary families: six inner source partitions, four
settings and three seeds. It reuses 720 completed native scalar panels. Human
aggregates first average the four rated populations equally; the teacher is
separate. It does not select on these new diagnostics, read outer/final holdouts,
or establish source-bootstrap uncertainty. These are all-setting comparisons,
not the independently tuned primary-screen scores.

## Scale-normalized tails and outlier ratio

Native quantile-normalized shape P99 is the 99th percentile of absolute residual
divided by the target's P99–P1 span. It removes marginal prediction scale. Native
outlier ratio instead counts logistic-rescaled residuals beyond 1.96 times the
corpus human-score standard deviation. It is not an observer-uncertainty outlier
test, and small values do not establish tight scatter.

| Approach | Human shape P99 | Lower than ReLU, of 72 | Human outlier ratio |
|---|---:|---:|---:|
| ReLU | 0.365007 | — | 0.025665% |
| Joint product, identity start | 0.356967 | 42 | 0.029007% |
| Product, pretrained frozen | 0.357833 | 46 | 0.034712% |
| Product, pretrained joint | 0.360175 | 42 | 0.045276% |
| Joint separate, identity start | 0.380415 | 26 | 0.092111% |
| Separate, pretrained frozen | 0.385555 | 17 | 0.152842% |
| Separate, pretrained joint | 0.387010 | 15 | 0.210953% |
| Softplus | 0.363369 | 42 | 0.025810% |
| Monotone SiLU | 0.364594 | 35 | 0.027257% |
| Ordinary SiLU diagnostic | 0.415251 | 8 | 0.246916% |

Joint product shaping has a modest average normalized-tail improvement, but it
is mixed across conditions. Its raw OLS residual P99 falls from 45.211367 to
25.760757 in prediction units; that also responds to prediction scale, so it
cannot by itself establish a comparable reduction in error.

For joint product shaping, the population detail is:

| Human population | ReLU shape P99 | Product shape P99 | ReLU OR | Product OR |
|---|---:|---:|---:|---:|
| kadid | 0.388176 | 0.367363 | 0.014902% | 0.013847% |
| tid2013 | 0.295246 | 0.300255 | 0.056134% | 0.027778% |
| konfig | 0.384970 | 0.380104 | 0.000000% | 0.074405% |
| cid22_a25 | 0.391637 | 0.380144 | 0.031626% | 0.000000% |

The mean OR increase is localized to KonFiG: five of 72 product conditions have
higher OR, and the other 67 tie ReLU at zero. All five are in `inner-0-1` across
different settings/seeds. The other three human populations have lower mean OR.
This identifies a diagnostic target; it does not authorize choosing a default
on that validation population or discarding those cases.

## Repeatability across seeds

For each of 24 fixed partition/setting groups, compute the sample SD of the
three seed values, then average those 24 SDs. This separates seed spread from
differences between settings and partitions. It is descriptive, not a confidence
interval or proof about optimizer causes.

| Approach | Mean within-condition seed SD of human shape P99 | Groups with lower SD, of 24 |
|---|---:|---:|
| ReLU | 0.00334727 | — |
| Joint product, identity start | 0.00596014 | 6 |
| Product, pretrained frozen | 0.00677570 | 4 |
| Product, pretrained joint | 0.00815082 | 4 |
| Joint separate, identity start | 0.01013776 | 2 |
| Separate, pretrained frozen | 0.01140945 | 5 |
| Separate, pretrained joint | 0.01210599 | 4 |
| Softplus | 0.00316607 | 12 |
| Monotone SiLU | 0.00307073 | 11 |
| Ordinary SiLU diagnostic | 0.01897693 | 0 |

Current joint product shaping therefore does not establish easier or more
repeatable training. The frozen-bank arms are also not a trial of one common
reusable default: `shaping_pretraining.matching_parent` binds each child to its
own seed-matched learned prefix. A common bank across seeds remains a distinct
experiment. Its derivation must exclude the evaluation source complement, state
its anchor/scale conventions, and preserve the native monotonicity/identity
contracts. It needs actual matched training before a transferable-default claim.

## Reproduction and retention

`just shaping-tail-summary STUDY DASHBOARD OUTPUT` preserves full comparisons in
`PAIRED_CASES.csv`; `just shaping-tail-seed-spread CASES OUTPUT` computes the
within-condition SD summary. Native full/shape metrics are reused, not rescored.
The two-second metadata analysis reported `peak-RSS 0.09GiB | min-avail 43619MiB |
peak-load 2.20`. The complete 720 native JSON inputs are retained as a standard
`tar.zst`, 205,503,105 original bytes compressed to 66,234,709 bytes. Every
individual native panel hash passed streaming extraction verification.

The source overlay, plain CSV, summaries and input package passed durable cold
checksum readback under `primary-tail-evidence-20261109`: 42 payload files,
76,754,679 bytes, plus its receipt. This archive is sufficient to inspect and
reproduce the tabulated metrics without retraining or rebuilding native code.
Seed summaries have their separate closure and reuse that archived CSV.

The seed-spread closure passed durable cold readback under
`primary-tail-seed-spread-verified-20261109`: six payload files, 185,651 bytes,
plus its receipt. It includes the calculator and references the existing shared
CSV archive. The committed calculator reproduces all 27 full summaries with
finite-value, complete-stratum and distinct-seed checks; wrapper zero seconds,
`peak-RSS 0.02GiB | min-avail 43604MiB | peak-load 1.91`.
