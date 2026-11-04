# New public A: weight audit, 2026-11-04

A means the adopted mixF3 seed-0 bake, not the historical profile with the same letter. The site calls this bake F0. This audit changes no metric arithmetic, model bytes, extraction, thresholds or public API. It measures model structure and controlled feature interventions, not perceptual accuracy or a release qualification.

Tool source: zensimx `7703327264402764354e09a6ee0f9353973d4c11`; exporter built in the assigned weights workspace. All inference, hidden activation checks, ablations and PDP/ICE points use the existing Rust exporter and its canonical `BakeScorer` / `Predictor` owners. Python only summarizes exported values; weight correlations use NumPy 2.5.2, reconstructing exported f32 values before f64 descriptive calculations.

| Bake | Damage + product inputs | Hidden width | SHA-256 |
| --- | --- | --- | --- |
| A0 | 492 + 492 | 128 | `185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088` |
| A1 | 492 + 492 | 128 | `c806f18393d88c76a783f6d9916a79abb7e7a3d4d51aed110634a74a337135d5` |
| A2 | 492 + 492 | 128 | `1ed0ffda1b3a525e7c2f54d6acc2ff64f0c87da8e57e4528fdd4bf6186e0a687` |
| E | 410 + 410 | 128 | `7642eff4c965d9b7b838747042253ec31efe7389ea832280b49deb802b916006` |

Exporter SHA-256: `d9e1ca3e9624036f567c29dbc5cd79668a697fe4a63b4bb33c9cf30fdaed2f6f`. Native certifier SHA-256: `ce9ff24b62b940877ddc94c1a794437c0a1b07075cbe6ddcf21ad404f116b942`. fx2 declaration SHA-256: `a3802ddeeede6ec5c9e3d03efdd6cc5af683ab4ea2b0d1143c8e9183c5bd46e5`. Per-output `provenance.json` also pins compiled metric, inspection-source and Cargo-lock hashes. Bake paths and complete commands are in `pins.json` / `plan.json` in the artifact directory below.

| Bank | Rows used for audit/ablation | Rows for ICE | SHA-256 |
| --- | --- | --- | --- |
| CID22 TRAIN oracle teacher fit | 12163 | 200 | `07e2f4b71fe0ccc67848f2896344e6d0b6a255b0e91a7da9062d471ea3e11100` |
| HDR teacher TRAIN, cube-root:203 | 7390 | 200 | `742fd8f314dcc5d1f812faa9924ac8b3fb048ba07dbc5f53a0b7f2b984857101` |

Every audit and ablation uses every row of its bank. ICE uses 200 deterministic, evenly spaced observed rows; each single-input grid uses 21 full-bank quantiles from p1 through p99. Geometry is the tool’s fixed 512 × 512. These are raw-feature interventions, not newly extracted image pairs. Fragility stays reference-only. No AIC table or human response column was read. HDR findings below cover the explicitly supplied teacher TRAIN bank; they do not claim coverage of every HDR collection or the complete training mixture.

## 1. What is flagged

Counts by kind and the pinned exporter’s severity, on CID22:

| Flag kind | Severity | A0 | A1 | A2 | E |
| --- | --- | --- | --- | --- | --- |
| bias_outlier | note | 0 | 1 | 1 | 0 |
| column_outlier | note | 2275 | 11458 | 929 | 3375 |
| constant_input | critical | 0 | 0 | 0 | 0 |
| dead-input | critical | 0 | 2 | 0 | 0 |
| dead_hidden | critical | 95 | 95 | 125 | 93 |
| dead_input | critical | 64 | 152 | 126 | 122 |
| drift | critical | 0 | 0 | 0 | 0 |
| duplicate_input | critical | 54 | 54 | 54 | 46 |
| row_outlier | note | 32344 | 18913 | 2511 | 20564 |
| saturated_hidden | critical | 95 | 95 | 125 | 93 |
| scaler_sign | critical | 0 | 0 | 0 | 0 |
| sign | critical | 0 | 0 | 0 | 0 |

“Critical” above is the tool’s presentation label, not this report’s defect verdict. The hidden-near-zero and flat-inactive flags refer to the same units and must not be added as distinct units. `duplicate_input` counts pairs with |Pearson| > 0.999, not proven identical feature computations. MAD outliers are notes, including undefined zero-MAD z-scores. Zero drift flags on CID22 mean no comparison bank was supplied, not absence of distribution shift.

| CID22 severity totals | A0 | A1 | A2 | E |
| --- | --- | --- | --- | --- |
| critical | 308 | 398 | 430 | 354 |
| serious | 0 | 0 | 0 | 0 |
| warning | 0 | 0 | 0 | 0 |
| note | 34619 | 30372 | 3441 | 23939 |

| Constraint / sparsity measurement | A0 | A1 | A2 | E |
| --- | --- | --- | --- | --- |
| Native certificate | pass | pass | pass | pass |
| Live max pairs | 1 | 1 | 1 | 3 |
| Negative encoder entries | 260 | 429 | 243 | 553 |
| Negative entries in max-pair witnesses | 260 | 429 | 243 | 553 |
| Negative entries outside valid witnesses | 0 | 0 | 0 | 0 |
| Exact-zero encoder rows | 0 | 2 | 0 | 0 |
| Encoder rows within 1e-6 | 0 | 26 | 2 | 0 |
| Exact-zero output rows | 64 | 126 | 124 | 122 |

A0’s 135 negative direct weights and 125 negative product weights all belong to unit 127, paired with 126. Twelve of those negative product entries also independently satisfy the fragility-gate inequality. A1/A2 use the same pair; E uses (122,123), (124,126), (125,127). Every bake has zero ramps, zero free ramps and zero free amplitude. The coupled condition is `a >= 0`, `a + b >= 0`, matching nonpositive output coefficients, and zero partner bias, giving a non-increasing max of two nonnegative damage projections. The native certificate, rather than individual encoder signs, owns that claim. Its real-arithmetic guarantee does not erase floating-point cancellation, measured below.

The only exactly dead damage slot is seed-1 f516, `v2_edge_width_change_s1_y`: both its direct row 295 and product row 787 are zero. This is seed-specific, not a dead family in the adopted A0. The A0 “64 dead_input” flags are all hidden→output zeros, not 64 unused extracted features.

| HDR compared with CID22 p1–p99 | A0 | A1 | A2 | E |
| --- | --- | --- | --- | --- |
| Constant input columns | 8 | 8 | 8 | 6 |
| Correlated input pairs | 46 | 46 | 46 | 39 |
| Input columns with any outside observation | 842 | 842 | 842 | 700 |
| Outside row × input cells | 462114 | 462114 | 462114 | 416397 |

All other audit counts remain their CID22 values on HDR. The three A seeds share the same declared input values, so their distribution flags coincide. A0’s constant HDR slots are f12, f25, f51 and f103 and their products (basic contrast increase); E omits f12. A0’s median flagged-column outside fraction is 3.903924% and its maximum is 41.055480%. A p1–p99 envelope deliberately leaves tails outside even on its own distribution; any outside value triggering “drift” is descriptive exposure, not a calibrated significance test or an HDR defect.

| Bank / model | Single-feature rises | Family rises | Total strict ICE rises | Largest rise (score points) | Rises > 0.01 |
| --- | --- | --- | --- | --- | --- |
| A0 | 1663 | 0 | 1663 | 1.2293133e-05 | 0 |
| hdr-A0 | 1804 | 4 | 1808 | 8.5260205e-06 | 0 |
| A1 | 20794 | 11 | 20805 | 1.7926467e-05 | 0 |
| A2 | 1021 | 2 | 1023 | 1.1196732e-05 | 0 |
| E | 1917 | 68 | 1985 | 2.8251514e-05 | 0 |
| hdr-E | 7067 | 192 | 7259 | 1.7605648e-05 | 0 |

Each row above covers 529 curves / 2,116,000 adjacent ICE comparisons. E includes the 82 added feature probes for the same comparison grid, although those probes are not E inputs. A0’s CID22 family curves have zero rises; the other measured family rise counts are retained in the table. All observed rises are below 0.001, hence “note” under the tool’s magnitude ladder. Counts are strict, with no tolerance or discarded rises. Seed differences change count, not this magnitude conclusion.

In the served concern-table JSON, a monotonicity flag’s `value` is the score after the intervention; `statistic` is the score rise. The magnitude columns here use the rise, not the absolute served score.

## 2. Which features A uses

Identity ablation sets one raw damage slot to exactly zero and calls `BakeScorer`, which recomputes its product with the row’s retained fragility. The effect is mean |Δ served score|. The 984-row `inputs-A0.tsv` records every individual encoder-row L1 norm, nonzero count and its joint damage-slot ablation; the effect is intentionally shared by the direct/product pair. Weight L1 is not standardized sensitivity.

| Rank | Damage ID / registry name | Direct Σ\|w\| | Product Σ\|w\| | CID22 mean \|Δ\| | HDR mean \|Δ\| |
| --- | --- | --- | --- | --- | --- |
| 1 | f20 `basic_edge_det_4th_s0_y` | 0.017038465 | 7.1278496 | 2.4309131 | 1.9226109 |
| 2 | f585 `v2_art_soft_peak_s2_y` | 11.166101 | 0.020542383 | 2.0810499 | 2.0134663 |
| 3 | f59 `basic_edge_det_4th_s1_y` | 4.5396214 | 3.560915 | 1.7972825 | 1.8393114 |
| 4 | f409 `v2_hf_mag_loss_s0_y` | 2.1424351 | 1.1593323 | 1.4978468 | 0.97807221 |
| 5 | f582 `v2_hf_loss_s2_y` | 9.0493813 | 0.013757586 | 1.3206689 | 1.1928024 |
| 6 | f403 `v2_ssim_dev4_s0_y` | 6.3317297 | 7.200067 | 1.3146287 | 1.0419154 |
| 7 | f495 `v2_hf_loss_s1_y` | 3.7316098 | 0.015447617 | 0.8594409 | 0.8507469 |
| 8 | f426 `v2_blockiness_s0_y` | 0.97341919 | 1.5970612 | 0.85437191 | 0.52046095 |
| 9 | f687 `v2_blockiness_s3_y` | 3.498951 | 0.025740504 | 0.7551808 | 0.33520003 |
| 10 | f592 `v2_iw_art_s2_y` | 12.817912 | 0.013409853 | 0.68640091 | 0.70569907 |

All ten highest-impact CID22 inputs are Y-channel features. Chroma contribution is measured separately in section 5. All 492 A0 damage IDs have nonzero ablation effect on CID22; 487 do on HDR. The HDR zero-effect slots are f12, f25, f51, f64 and f103 (all basic contrast increase). Four of those are also constant inputs; f64 is not constant but its identity intervention has zero served-score effect at the measured precision. No declared signal family has identically zero effect on either bank.

| Bank / bake | 50% mass: IDs / fraction | 90% mass: IDs / fraction | 99% mass: IDs / fraction | Sum of single-slot effects |
| --- | --- | --- | --- | --- |
| A0 | 13/492 (2.6423%) | 69/492 (14.0244%) | 149/492 (30.2846%) | 30.358423 |
| hdr-A0 | 12/492 (2.4390%) | 65/492 (13.2114%) | 137/492 (27.8455%) | 25.680792 |
| A1 | 14/492 (2.8455%) | 71/492 (14.4309%) | 145/492 (29.4715%) | 30.727223 |
| A2 | 14/492 (2.8455%) | 72/492 (14.6341%) | 151/492 (30.6911%) | 30.640137 |
| E | 10/410 (2.4390%) | 49/410 (11.9512%) | 110/410 (26.8293%) | 30.635493 |
| hdr-E | 9/410 (2.1951%) | 44/410 (10.7317%) | 104/410 (25.3659%) | 24.716773 |

These fractions normalize the sum of single-slot effects; they are not additive portions of an individual score. Family interventions overlap and are also non-additive. Family ranking below uses the exporter’s registry family × signal groups, with all damage slots in each group jointly set to zero:

| Rank | Signal family | CID22 mean \|Δ\| | HDR mean \|Δ\| |
| --- | --- | --- | --- |
| 1 | basic × edge_det | 8.2303771 | 7.3355827 |
| 2 | v2 × art_soft_peak | 4.1587734 | 4.0456277 |
| 3 | v2 × hf_mag_loss | 3.1136794 | 1.9909855 |
| 4 | v2 × blockiness | 2.5863805 | 1.6473961 |
| 5 | v2 × hf_loss | 2.4840599 | 2.3826114 |
| 6 | v2 × ssim_dev4 | 1.5918656 | 1.2075215 |
| 7 | basic × edge_art | 1.0839456 | 1.1550383 |
| 8 | v2 × iw_art | 0.93587319 | 0.84699877 |
| 9 | v2 × det_soft_peak | 0.77393832 | 0.66813842 |
| 10 | v2 × gms | 0.69215617 | 0.44218524 |
| 11 | v2 × masked_det | 0.6009718 | 0.56562381 |
| 12 | v2 × iw_det | 0.59728808 | 0.38338747 |
| 13 | v2 × ringing | 0.54142853 | 0.52339834 |
| 14 | basic × var_loss | 0.53281321 | 0.4542269 |
| 15 | v2 × hf_gain | 0.48135038 | 0.43138775 |
| 16 | basic × tex_loss | 0.45889243 | 0.22323874 |
| 17 | v2 × det | 0.41090438 | 0.28750296 |
| 18 | v2 × ssim_dev2 | 0.22208226 | 0.15551487 |
| 19 | v2 × pjnd_transducer_low_k | 0.22044677 | 0.19848487 |
| 20 | v2 × pjnd_transducer | 0.19033897 | 0.16998479 |
| 21 | v2 × art | 0.17703345 | 0.17626577 |
| 22 | v2 × pjnd_transducer_high_k | 0.13911394 | 0.12276395 |
| 23 | v2 × banding | 0.11219311 | 0.05736518 |
| 24 | v2 × iw_ssim | 0.040543142 | 0.012971864 |
| 25 | v2 × iw_mse | 0.033959285 | 0.022808525 |
| 26 | basic × ssim | 0.029184322 | 0.013415141 |
| 27 | v2 × ssim | 0.018871472 | 0.0067710451 |
| 28 | v2 × edge_width_change | 0.017905963 | 0.0022039005 |
| 29 | v2 × ssim_soft_peak | 0.0098573161 | 0.0049736346 |
| 30 | basic × contrast_inc | 0.0077391007 | 0.00044601851 |
| 31 | v2 × mse | 0.0061381054 | 0.0042669068 |
| 32 | v2 × masked_art | 0.004662116 | 0.0039589695 |
| 33 | v2 × masked_mse | 0.0036439589 | 0.0022453455 |
| 34 | v2 × masked_ssim | 0.0034810666 | 0.002239176 |
| 35 | basic × mse | 0.0011139099 | 0.0011368624 |

Every input is below 1% of A0’s top input in the following complete list of signal families. CID22 cutoff = 0.024309131 points. None is all-dead:

| Signal family | Damage IDs | Largest individual effect |
| --- | --- | --- |
| basic × ssim | 36 | 0.016900336 |
| basic × mse | 12 | 0.00041415104 |
| basic × contrast_inc | 12 | 0.0073794756 |
| v2 × ssim | 12 | 0.016972942 |
| v2 × mse | 12 | 0.002511505 |
| v2 × ssim_soft_peak | 12 | 0.0046013099 |
| v2 × masked_ssim | 12 | 0.0017472995 |
| v2 × masked_art | 12 | 0.0032114656 |
| v2 × masked_mse | 12 | 0.0015857507 |
| v2 × iw_mse | 12 | 0.0091323583 |
| v2 × edge_width_change | 12 | 0.0047224625 |

Using each bank’s own top-input cutoff, the HDR list is: `basic × ssim`, `basic × mse`, `basic × contrast_inc`, `v2 × ssim`, `v2 × mse`, `v2 × ssim_soft_peak`, `v2 × masked_ssim`, `v2 × masked_art`, `v2 × masked_mse`, `v2 × iw_ssim`, `v2 × iw_mse`, `v2 × edge_width_change`.

Extraction implication: there is no measured whole dead family to remove from A0. The low-effect list names SSIM/moment and soft-peak pooling, MSE, contrast ratios, masking, information-weighted MSE and edge-width work worth profiling in a separately authorized experiment. They remain nonzero and share extraction kernels with higher-impact features, so these data authorize neither omission nor a speed claim. Exact dormant hidden units are a separate model-compute opportunity, with floating-point parity required before any compaction.

## 3. The hidden layer

| Bake / bank | Never active | Active >99% | Active with nonzero output | Max-pair share of mean \|raw contribution\| |
| --- | --- | --- | --- | --- |
| A0 | 95 | 32 | 33 | 99.99846693% |
| hdr-A0 | 95 | 32 | 33 | 99.99840640% |
| A1 | 95 | 32 | 2 | 100.00000000% |
| hdr-A1 | 95 | 32 | 2 | 100.00000000% |
| A2 | 125 | 2 | 3 | 99.99998959% |
| hdr-A2 | 125 | 2 | 3 | 99.99998935% |
| E | 93 | 32 | 6 | 100.00000000% |
| hdr-E | 93 | 32 | 6 | 100.00000000% |

A0 units 0–94 have exactly zero encoder columns and nonpositive bias: they are structurally inactive, not merely unobserved on one bank. Units 95–126 are active on every row; 127 is active on 58.307983% of CID22 and 34.316644% of HDR rows. Positive ReLU is linear, not saturated. Units 95–125 have identical weights, bias and output coefficient, so the 33 active units contain three distinct response shapes. All are kept in this audit.

| A0 live units | Top encoder inputs (signed weights) | Output coefficient |
| --- | --- | --- |
| 95–125 (each) | f426×frag 0.016220093, f409×frag 0.008148193, f20×frag 0.0055732727 | -3.1052721e-05 |
| 126 | f403×frag 6.6171875, f582 5.9648438, f403 4.8632812, f585 4.7773438, f411 4.5, f124 3.9609375, f59 3.8105469, f20×frag 3.7226562 | -0.49840412 |
| 127 | f592 9.984375, f585 6.3554688, f411 -4.4804688, f597 4.4492188, f581 3.7226562, f124 -3.6640625, f687 3.4726562, f578 3.4570312 | -0.49840412 |

These are largest stored coefficients, not an assertion that differently scaled inputs contribute in that order. A0’s repeated units chiefly read fragility-weighted blockiness f426, high-frequency magnitude loss f409 and detail loss f20. Unit 126 mixes SSIM deviation, high-frequency loss and artifact/detail detectors; 127 supplies the signed correction branch. The pair carries almost all raw output contribution, but the small residual is nonzero. A1 has 33 active units but only 126/127 reach output; A2 has 125/126/127 active and reaching output. E has 35 active units, of which the six max-pair members reach output. Per-unit rates, output coefficients and named top-eight inputs for every live unit of every model/bank are retained in `analysis.json`. Raw contribution excludes the identity/tail output stage and dial spline.

## 4. Monotonicity in practice

| Top-impact input | p1 | p99 | Strict ICE rises / 4,000 comparisons | PDP mean rises |
| --- | --- | --- | --- | --- |
| f20 | 0.0043240604 | 0.038127537 | 0 | 0 |
| f585 | 0.0080105139 | 0.13050785 | 0 | 0 |
| f59 | 0.0019566723 | 0.019393936 | 0 | 0 |
| f409 | 0.0087557834 | 0.40495981 | 0 | 0 |
| f582 | 0.0039184031 | 0.091427689 | 0 | 0 |
| f403 | 0.0039253419 | 0.12885601 | 0 | 0 |
| f495 | 0.0059428387 | 0.16189506 | 0 | 0 |
| f426 | 0.0093895045 | 0.077213224 | 0 | 0 |
| f687 | 0.0019820926 | 0.029130889 | 0 | 0 |
| f592 | 0.00031124898 | 0.020960993 | 0 | 0 |

All ten curves are non-increasing on every sampled row and in their mean. The same is true for HDR’s ten highest-impact inputs. This is finite sampled evidence, not a proof about every floating-point input. A0’s rises occur outside this top ten. The largest observed rises are located below; “p” is the quantile of the feature value in the full bank, not the quantile of served scores:

| Bank | Damage ID | Bank row | Quantile interval | Damage interval | Rise (score points) |
| --- | --- | --- | --- | --- | --- |
| A0 | f594 | 6294 | p30.4–p35.3 | 0.00076798601 → 0.00087950497 | 1.2293133e-05 |
| A0 | f671 | 5011 | p5.9–p10.8 | 0.00010900922 → 0.00017893091 | 8.6242731e-06 |
| A0 | f100 | 5011 | p10.8–p15.7 | 2.7296929e-06 → 3.7592486e-06 | 8.62427e-06 |
| A0 | f580 | 5011 | p30.4–p35.3 | 0.00066555014 → 0.00076808644 | 8.62427e-06 |
| A0 | f590 | 5011 | p25.5–p30.4 | 0.00038489264 → 0.00045872072 | 8.62427e-06 |
| hdr-A0 | f141 | 519 | p25.5–p30.4 | 0.00047715276 → 0.00052405494 | 8.5260205e-06 |
| hdr-A0 | f141 | 519 | p40.2–p45.1 | 0.00061717116 → 0.00066874734 | 8.5260205e-06 |
| hdr-A0 | f141 | 519 | p50.0–p54.9 | 0.00071904439 → 0.00077377406 | 8.5260205e-06 |
| hdr-A0 | f667 | 519 | p94.1–p99.0 | 0.0016033192 → 0.0023463445 | 8.5260205e-06 |
| hdr-A0 | f100 | 519 | p74.5–p79.4 | 2.7076295e-05 → 4.5255338e-05 | 8.5260181e-06 |

All strict rise tuples are retained in `rises-A0.json` / `hdr-rises-A0.json`; no rounding tolerance was applied. The observed tiny rises are consistent with the signed max-pair’s floating-point cancellation and subsequent score mapping. The native certificate explicitly allows rounding disagreement for paired terms. Their count is real; interpreting them as 0.01-point or larger reversals would be incorrect.

## 5. Seeds, E and the 82 added chroma inputs

Per-input weight drift aligns `(damage ID, optional fragility ID)` and correlates each stored 128-weight row. It does not align/permutate neurons. The full per-input correlations, L1 norms and indexwise L2 changes are in `weight-drift-*.tsv`. Constant-zero rows have undefined correlation and are excluded explicitly:

| Comparison | Shared encoder rows / defined | Row r: p10 / median / p90 | Direct-row median r | Across-row L1-norm r | Damage-ablation r | Top-10 overlap |
| --- | --- | --- | --- | --- | --- | --- |
| A0-A1 | 984 / 982 | 0.019241182 / 0.63919362 / 0.99849278 | 0.69192654 | 0.97499864 | 0.97244393 | 8 |
| A0-A2 | 984 / 984 | 0.062674452 / 0.35278121 / 0.99780706 | 0.45999581 | 0.97516186 | 0.98050858 | 8 |
| A1-A2 | 984 / 982 | 0.044700152 / 0.88348624 / 0.99979811 | 0.94069792 | 0.99362133 | 0.99310345 | 9 |
| A0-E | 820 / 820 | -0.42106259 / 0.23495949 / 0.82471861 | 0.21284383 | 0.61880415 | 0.87124126 | 7 |

E shares 410 damage IDs / 820 direct-plus-product rows with A. A0 and E keep the same dominant family order at the top: basic detail loss, soft artifact peak, high-frequency magnitude loss, blockiness and high-frequency loss. Per-row weight drift is substantially larger against E; hidden representation changes do not by themselves establish a functional regression. The ablation correlation and top-feature overlap measure the common read pattern more directly.

| Added scale-0 X/B chroma: bank / seed | Nonzero slot effects / 82 | Sum of single-slot effects | Fraction of all single-slot mass |
| --- | --- | --- | --- |
| A0 | 82/82 | 5.4774854 | 18.042721% |
| A1 | 82/82 | 5.3551133 | 17.427912% |
| A2 | 82/82 | 5.2669859 | 17.189825% |
| hdr-A0 | 81/82 | 4.630523 | 18.031076% |

A0’s ten highest-effect additions:

| Added ID / name | CID22 mean \|Δ\| |
| --- | --- |
| f380 `v2_hf_mag_loss_s0_x` | 0.56074377 |
| f438 `v2_hf_mag_loss_s0_b` | 0.53789947 |
| f33 `basic_edge_det_4th_s0_b` | 0.41693667 |
| f379 `v2_hf_loss_s0_x` | 0.25061719 |
| f6 `basic_edge_det_mean_s0_x` | 0.22489598 |
| f7 `basic_edge_det_4th_s0_x` | 0.22302551 |
| f437 `v2_hf_loss_s0_b` | 0.20575311 |
| f37 `basic_tex_loss_s0_b` | 0.19427815 |
| f8 `basic_edge_det_2nd_s0_x` | 0.1832716 |
| f34 `basic_edge_det_2nd_s0_b` | 0.17481259 |

The additions are full-resolution chroma damage detectors plus their fragility products, not dead padding. The summed effect fractions are sensitivity summaries, not additive score shares, perceptual gains, or a measured speed/quality trade-off. This audit does not refit or compare held-out quality.

## 6. What this means for shipping A

For [public-crate PR 2](https://github.com/imazen/zensim/pull/2), this audit finds no violated sign/scaler constraint or exactly dead extracted family in adopted A0; all 492 damage IDs affect CID22 scores and 487 affect HDR teacher scores. A reviewer should recognize the legal signed max pair (126/127), 95 structurally inactive hidden units, 31 repeated small-response units, seed-1-only dead f516, and the explicitly measured HDR tail exposure. The concern table’s MAD outliers, near-collinearity, dormant units and tiny strict floating-point rises are useful diagnostics, not independent evidence of a shipping defect. Every observed rise is below 0.01 points, and the top-ten feature curves have none. Strict bitwise monotonicity is nevertheless false on the recorded lower-impact interventions; retain that numerical limitation explicitly. Removing even low-effect features or merging hidden units needs separate bit/quality/runtime qualification; this report neither changes the numeric path nor substitutes for the remaining production gates.

## Reproduction and retained evidence

Artifact root (paths checked on dev): `/mnt/v/output/zensim/a-weight-audit-2026-11-04/`; mirror: `/mnt/tower/output/zensim/a-weight-audit-2026-11-04/`. Large JSON, model binaries and bank data are not committed. `SHA256SUMS` covers retained artifacts, `site-equality.json` records the site cross-check, and `analysis.json` contains the descriptive measurements behind these tables. `plan.json` records every native argument and output path; `logs/` preserves complete command output and resource receipts; `native/` preserves the exporter built here.

Verification: all six site audit objects and all six ablation-summary objects match this independent exporter run field for field. Every captured site rise tuple (intervention, bank row, grid point and exact delta) matches. Six native `curve-audit` replays independently recheck 12,696,000 saved adjacent comparisons and their recorded flags. `just lint-scripts` passes; Markdown table structure and added-file whitespace checks pass. These checks validate the report’s reproduction, not new image/production behavior.

Executed from the weights workspace (the first audit uses the named recipe; the existing serial bundle owner reacquires the same resource guard for every subsequent job):

```sh
just viz-fmt
just viz-build
export VIZ_CERTIFIER="$HOME/tmp/e36test/r10/bin-mixF3-s0/bake_dial_refit"
just viz-audit "$HOME/tmp/e36test/r10/runs/mixF3-s0/pack_e059/refit/production-f16.bin" \
  /var/tmp/byc-fchroma-20261029/r3/w100/root/wide/main/real/cid22_fit.parquet \
  /mnt/v/output/zensim/a-weight-audit-2026-11-04/audit-A0
just viz-bundle "$HOME/tmp/zensim-weights/plan.json"
just viz-bundle "$HOME/tmp/zensim-weights/curve-plan.json"
OPENBLAS_NUM_THREADS=1 python3 /mnt/v/output/zensim/a-weight-audit-2026-11-04/summarize.py
```

The bundle runs `audit BAKE --bank CID22` for the remaining three bakes, `audit BAKE --bank HDR --training-bank CID22` for all four, and `mlp-bank BAKE --bank BANK` for four CID22 bakes plus HDR A0/E. E includes `--other A0` so the comparison probes all 492 IDs. Every native job runs with `ZENSIM_FORMULA_REV=7`, `flock "$HOME/tmp/heavy.lock"` and `run-heavy --mem 16G --jobs 8`. No competing lock was bypassed. No runtime speed claim is made.
