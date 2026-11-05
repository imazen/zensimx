# R14 measured read tables

A is the registered F baseline, seeds 0–2. Candidate seeds: 0; means use only those registered seeds.
No across-seed interval is implied. Per-seed whole-reference/family intervals remain in the native read JSON.
Candidate seeds 1–2 are not measured here. The registration requires seed-0 product/development eligibility and a separate owner cost decision.

## E36: 40 origins, 646,677 selection pairs, 10,000 draws

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| E36 inversion share (%) | 3.396647 | 3.227577 | 3.869474 | 5.353987 |
| E36 change vs A (percentage points), 95% source bootstrap | baseline | -0.169069 [-0.324519, -0.025491] | +0.472827 [+0.272937, +0.678314] | +1.957340 [+1.718568, +2.225726] |

## EVAL MOS median residual / target 1–99% span (%)

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| cid-iq (n=210) | 7.78359 | 8.08307 | 9.32427 | 8.91935 |
| cid-oa (n=91) | 8.41368 | 8.14282 | 8.19112 | 7.10748 |
| colourlab-rating-variability-2022 (n=192) | 11.15986 | 11.95108 | 13.89177 | 14.24897 |
| iqa-odi (n=48) | 7.86276 | 7.62396 | 8.74231 | 8.08210 |
| irccyn-ivc (n=60) | 6.65655 | 7.22757 | 6.02374 | 7.13602 |
| live-md (n=180) | 6.97772 | 7.10098 | 6.81426 | 6.55936 |
| mdid (n=480) | 6.19979 | 6.23829 | 6.08584 | 5.93831 |
| nits-iqa (n=405) | 11.20201 | 11.27634 | 11.12388 | 11.30851 |
| vcl-fer (n=192) | 5.99554 | 6.20245 | 6.01628 | 6.33452 |
| all nine, equal collection mean | 8.02794 | 8.20518 | 8.46816 | 8.40385 |
| eight collections excluding training-source MDID | 8.25646 | 8.45104 | 8.76595 | 8.71204 |
| common three: IQA-ODI, NITS-IQA, VCL-FER | 8.35343 | 8.36759 | 8.62749 | 8.57504 |

## EVAL MOS p90 residual / target 1–99% span (%)

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| cid-iq (n=210) | 21.66163 | 20.45771 | 23.36330 | 22.31020 |
| cid-oa (n=91) | 18.54655 | 17.73680 | 16.82785 | 16.58254 |
| colourlab-rating-variability-2022 (n=192) | 31.98866 | 31.73472 | 34.71204 | 35.37378 |
| iqa-odi (n=48) | 18.54617 | 18.73334 | 21.43581 | 20.95628 |
| irccyn-ivc (n=60) | 20.00259 | 18.77512 | 18.50619 | 19.19964 |
| live-md (n=180) | 15.91087 | 15.39860 | 16.35868 | 15.40626 |
| mdid (n=480) | 19.42745 | 19.80508 | 17.70877 | 17.09864 |
| nits-iqa (n=405) | 30.48377 | 31.04556 | 30.54532 | 30.36295 |
| vcl-fer (n=192) | 14.16272 | 14.34287 | 14.10158 | 14.34895 |

## EVAL MOS absolute SROCC

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| cid-iq (n=210) | 0.86925 | 0.86847 | 0.84462 | 0.85230 |
| cid-oa (n=91) | 0.90886 | 0.91209 | 0.91248 | 0.92766 |
| colourlab-rating-variability-2022 (n=192) | 0.73067 | 0.72726 | 0.64742 | 0.65145 |
| iqa-odi (n=48) | 0.86550 | 0.86073 | 0.83782 | 0.85530 |
| irccyn-ivc (n=60) | 0.91898 | 0.92064 | 0.91656 | 0.90419 |
| live-md (n=180) | 0.89622 | 0.89538 | 0.89691 | 0.88594 |
| mdid (n=480) | 0.92592 | 0.92222 | 0.93605 | 0.93418 |
| nits-iqa (n=405) | 0.73227 | 0.72720 | 0.74238 | 0.74096 |
| vcl-fer (n=192) | 0.94934 | 0.94901 | 0.95373 | 0.95073 |
| all nine, equal collection mean | 0.86634 | 0.86478 | 0.85422 | 0.85586 |
| eight collections excluding training-source MDID | 0.85889 | 0.85760 | 0.84399 | 0.84606 |
| common three: IQA-ODI, NITS-IQA, VCL-FER | 0.84904 | 0.84565 | 0.84464 | 0.84899 |

## EVAL compression: signed SROCC / within-reference SROCC / normalized median error

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| cid-iq (n=70) | 0.93332 / 0.94325 / 5.988% | 0.93164 / 0.93891 / 5.710% | 0.91928 / 0.94325 / 6.409% | 0.92670 / 0.94498 / 5.955% |
| colourlab-rating-variability-2022 (n=18) | 0.80083 / 0.83682 / 7.268% | 0.80083 / 0.83682 / 6.897% | 0.76048 / 0.83682 / 10.492% | 0.77807 / 0.83682 / 7.679% |
| iqa-odi (n=48) | 0.86550 / undefined / 7.863% | 0.86073 / undefined / 7.624% | 0.83782 / undefined / 8.742% | 0.85530 / undefined / 8.082% |
| irccyn-ivc (n=50) | 0.91171 / 0.97149 / 6.882% | 0.90976 / 0.97036 / 6.817% | 0.90270 / 0.96530 / 5.854% | 0.88905 / 0.96401 / 7.301% |
| nits-iqa (n=135) | 0.85176 / 0.88189 / 9.031% | 0.85173 / 0.88216 / 8.920% | 0.84792 / 0.87918 / 9.257% | 0.85077 / 0.88136 / 9.029% |
| vcl-fer (n=96) | 0.94934 / 0.98193 / 5.098% | 0.94935 / 0.98164 / 5.178% | 0.94749 / 0.98339 / 5.490% | 0.94438 / 0.98339 / 6.186% |
| row-weighted SROCC / equal-collection error | 0.89449 / 7.021% | 0.89341 / 6.858% | 0.88445 / 7.707% | 0.88704 / 7.372% |

## EVAL noncodec: signed SROCC / within-reference SROCC / normalized median error

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| cid-iq (n=140) | 0.83917 / 0.85950 / 9.572% | 0.83656 / 0.85696 / 10.432% | 0.81485 / 0.83416 / 12.375% | 0.81507 / 0.83406 / 11.000% |
| colourlab-rating-variability-2022 (n=174) | 0.70911 / 0.71955 / 11.875% | 0.70495 / 0.71560 / 13.330% | 0.61920 / 0.62895 / 15.111% | 0.62162 / 0.62917 / 15.748% |
| cid-oa (n=91) | 0.90886 / 0.99025 / 8.414% | 0.91209 / 0.98999 / 8.143% | 0.91248 / 0.98999 / 8.191% | 0.92766 / 0.98842 / 7.107% |
| irccyn-ivc (n=10) | 0.91515 / 1.00000 / 5.279% | 0.91515 / 1.00000 / 5.411% | 0.91515 / 1.00000 / 8.526% | 0.91515 / 1.00000 / 8.549% |
| live-md (n=90) | 0.89456 / 0.95396 / 6.341% | 0.89004 / 0.95595 / 6.576% | 0.90801 / 0.96934 / 6.020% | 0.89054 / 0.96457 / 6.620% |
| nits-iqa (n=270) | 0.66970 / 0.68805 / 12.750% | 0.66255 / 0.67575 / 12.601% | 0.68849 / 0.70630 / 13.036% | 0.68378 / 0.70341 / 12.701% |
| vcl-fer (n=96) | 0.95460 / 0.97028 / 5.853% | 0.95047 / 0.96591 / 5.598% | 0.95825 / 0.96853 / 5.587% | 0.94696 / 0.95542 / 6.505% |
| row-weighted SROCC / equal-collection error | 0.78725 / 8.584% | 0.78320 / 8.870% | 0.77338 / 9.835% | 0.77097 / 9.747% |

## G-CHROMA-L: dev views >64 px; MAE in score points against A0

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| ycc444 | 2.762208 | 2.622400 | 2.518256 | 2.887094 |
| ycc420 | 3.324290 | 3.035234 | 2.939608 | 3.117190 |
| xyb-capped | 2.060476 | 1.819028 | 1.706458 | 1.677752 |
| xyb-extended | 3.576321 | 3.462384 | 4.204848 | 4.566479 |
| band-ac | 2.138349 | 2.091953 | 1.871941 | 2.040953 |
| level pass / rank pass (seeds) | reference A0 | 1/1 / 1/1 | 0/1 / 1/1 | 0/1 / 1/1 |

## Label-free pixel and dial checks

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| G-STEER successes /135 | 133 / 134 / 134 | 134 | 127 | 128 |
| C1–C6, both standard and ladder grids | 3/3 pass | 1/1 pass | 1/1 pass | 1/1 pass |
| Identity: 744 scores exactly 100; read sets invariant | 3/3 pass | 1/1 pass | 1/1 pass | 1/1 pass |
| NEARID N1 / N2 | 3/3 / 3/3 pass | 1/1 / 1/1 pass | 1/1 / 1/1 pass | 1/1 / 1/1 pass |
| NEARID N3 monotone ladders /144 (bar 122) | 132 / 132 / 132 | 131 | 135 | 137 |

## standard: contract and addressability measurements (seed values)

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| A1 | 98.999126 (not_measured) / 99.169362 (not_measured) / 99.161280 (not_measured) | 98.959188 (not_measured) | 99.303383 (not_measured) | 99.444579 (not_measured) |
| A2 | -34.437015 (not_measured) / -33.377712 (not_measured) / -31.875432 (not_measured) | -30.449197 (not_measured) | -21.297499 (not_measured) | -11.479221 (not_measured) |
| A3 | 96.671093 (not_measured) / 96.914713 (not_measured) / 96.967713 (not_measured) | 96.811772 (not_measured) | 97.335734 (not_measured) | 97.277691 (not_measured) |
| A4 | 16.840151 (not_measured) / 17.249426 (not_measured) / 17.742258 (not_measured) | 17.470988 (not_measured) | 22.061506 (not_measured) | 20.692152 (not_measured) |
| A5 | 133.436141 (not_measured) / 132.547074 (not_measured) / 131.036712 (not_measured) | 129.408385 (not_measured) | 120.600882 (not_measured) | 110.923801 (not_measured) |
| A6 | 79.830942 (not_measured) / 79.665288 (not_measured) / 79.225455 (not_measured) | 79.340784 (not_measured) | 75.274228 (not_measured) | 76.585539 (not_measured) |
| A7r | undefined (not_measured) / undefined (not_measured) / undefined (not_measured) | undefined (not_measured) | undefined (not_measured) | undefined (not_measured) |
| A8r | -131.560754 (not_measured) / -127.737196 (not_measured) / -126.506806 (not_measured) | -91.647582 (not_measured) | -48.587504 (not_measured) | -31.343358 (not_measured) |
| C1 | 0.997453 (pass) / 0.998379 (pass) / 0.997916 (pass) | 0.996758 (pass) | 0.992821 (pass) | 0.993284 (pass) |
| C2 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C3 | 0.768000 (pass) / 0.741000 (pass) / 0.723500 (pass) | 0.679500 (pass) | 0.522000 (pass) | 0.311500 (pass) |
| C4 | -204.532324 (pass) / -201.435230 (pass) / -201.361266 (pass) | -139.238887 (pass) | -71.696899 (pass) | -51.954180 (pass) |
| C5 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C6 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| G-ADDR regression state | NOT MEASURABLE (unregistered dial grid) / NOT MEASURABLE (unregistered dial grid) / NOT MEASURABLE (unregistered dial grid) | NOT MEASURABLE (unregistered dial grid) | NOT MEASURABLE (unregistered dial grid) | NOT MEASURABLE (unregistered dial grid) |
| G-DIAL p5 / p95 / mono; pass | 16.840151 / 96.671093 / 0.997453; True ; 17.249426 / 96.914713 / 0.998379; True ; 17.742258 / 96.967713 / 0.997916; True | 17.470988 / 96.811772 / 0.996758; True | 22.061506 / 97.335734 / 0.992821; True | 20.692152 / 97.277691 / 0.993284; True |

## ladder: contract and addressability measurements (seed values)

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| A1 | 100.000000 (not_measured) / 100.000000 (not_measured) / 100.000000 (not_measured) | 100.000000 (not_measured) | 100.000000 (not_measured) | 100.000000 (not_measured) |
| A2 | -91.737782 (not_measured) / -90.754042 (not_measured) / -90.396596 (not_measured) | -72.637631 (not_measured) | -49.359958 (not_measured) | -35.623329 (not_measured) |
| A3 | 94.215658 (not_measured) / 94.422051 (not_measured) / 94.538832 (not_measured) | 94.581835 (not_measured) | 95.062749 (not_measured) | 94.886669 (not_measured) |
| A4 | -3.461464 (not_measured) / -2.990058 (not_measured) / -2.190195 (not_measured) | -1.044086 (not_measured) | 3.903071 (not_measured) | 4.949442 (not_measured) |
| A5 | 191.737782 (not_measured) / 190.754042 (not_measured) / 190.396596 (not_measured) | 172.637631 (not_measured) | 149.359958 (not_measured) | 135.623329 (not_measured) |
| A6 | 97.677122 (not_measured) / 97.412109 (not_measured) / 96.729027 (not_measured) | 95.625921 (not_measured) | 91.159678 (not_measured) | 89.937227 (not_measured) |
| A7r | undefined (not_measured) / undefined (not_measured) / undefined (not_measured) | undefined (not_measured) | undefined (not_measured) | undefined (not_measured) |
| A8r | -131.560754 (not_measured) / -127.737196 (not_measured) / -126.506806 (not_measured) | -91.647582 (not_measured) | -48.587504 (not_measured) | -31.343358 (not_measured) |
| C1 | 0.992349 (pass) / 0.993837 (pass) / 0.992349 (pass) | 0.992774 (pass) | 0.985443 (pass) | 0.980873 (pass) |
| C2 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C3 | 0.768000 (pass) / 0.741000 (pass) / 0.723500 (pass) | 0.679500 (pass) | 0.522000 (pass) | 0.311500 (pass) |
| C4 | -204.532324 (pass) / -201.435230 (pass) / -201.361266 (pass) | -139.238887 (pass) | -71.696899 (pass) | -51.954180 (pass) |
| C5 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C6 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| G-ADDR regression state | NOT MEASURABLE (unregistered dial grid) / NOT MEASURABLE (unregistered dial grid) / NOT MEASURABLE (unregistered dial grid) | NOT MEASURABLE (unregistered dial grid) | NOT MEASURABLE (unregistered dial grid) | NOT MEASURABLE (unregistered dial grid) |
| G-DIAL p5 / p95 / mono; pass | -3.461464 / 94.215658 / 0.992349; True ; -2.990058 / 94.422051 / 0.993837; True ; -2.190195 / 94.538832 / 0.992349; True | -1.044086 / 94.581835 / 0.992774; True | 3.903071 / 95.062749 / 0.985443; True | 4.949442 / 94.886669 / 0.980873; True |

A G-ADDR state of not measured is not a pass. Registered regression floors and mentor truth
must be present for that owner to grade A1–A7r; R14 does not change those gates or inputs.

## HDR cube-root:203 — pooled signed SROCC

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 0.805356 | 0.803800 | 0.797720 | 0.797720 |
| ladder_eval (n=3245, refs=455) | 0.786123 | 0.783962 | 0.775373 | 0.762921 |
| hdrteach_val (n=3900, refs=300) | 0.891399 | 0.889412 | 0.896233 | 0.905753 |
| upiq_dev (n=50, refs=4) | 0.941128 | 0.938151 | 0.930660 | 0.946699 |
| upiq_all (n=380, refs=30) | 0.811412 | 0.809175 | 0.798587 | 0.825999 |

## HDR cube-root:203 — raw pooled PLCC

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 0.789547 | 0.787202 | 0.777283 | 0.777984 |
| ladder_eval (n=3245, refs=455) | 0.574024 | 0.568175 | 0.554673 | 0.545825 |
| hdrteach_val (n=3900, refs=300) | 0.976716 | 0.977221 | 0.969941 | 0.971893 |
| upiq_dev (n=50, refs=4) | 0.934591 | 0.934029 | 0.909584 | 0.913457 |
| upiq_all (n=380, refs=30) | 0.807453 | 0.807833 | 0.798206 | 0.819516 |

## HDR cube-root:203 — mean within-reference Spearman

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 0.816518 | 0.821188 | 0.827332 | 0.829945 |
| ladder_eval (n=3245, refs=455) | 0.987108 | 0.985353 | 0.988571 | 0.977038 |
| hdrteach_val (n=3900, refs=300) | 0.999982 | 0.999982 | 0.999872 | 0.999982 |
| upiq_dev (n=50, refs=4) | 0.966334 | 0.966334 | 0.966334 | 0.964585 |
| upiq_all (n=380, refs=30) | 0.962233 | 0.963536 | 0.957536 | 0.956577 |

## HDR cube-root:203 — within-reference inversion count

| Read | A | T1a | T1b | T1c |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 45 / 44 / 45 | 44 | 43 | 42 |
| ladder_eval (n=3245, refs=455) | 50 / 51 / 49 | 54 | 59 | 95 |
| hdrteach_val (n=3900, refs=300) | 1 / 1 / 1 | 1 | 7 | 1 |
| upiq_dev (n=50, refs=4) | 15 / 15 / 15 | 15 | 15 | 16 |
| upiq_all (n=380, refs=30) | 121 / 121 / 124 | 120 | 129 | 131 |

Pooled ladder correlations are descriptive only; the registered ladder outcome is within-ladder rank.
4Kdtb has one family, so its family bootstrap interval is undefined. HDR is report-only; no HDR training leg was added.
The EVAL logistic fit is an assessment transform, never a fitted served-model update.
