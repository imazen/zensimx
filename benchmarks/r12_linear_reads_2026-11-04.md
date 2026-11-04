# R12 measured read tables

A is the registered F baseline. Cells are seed means unless seed values are separated by `/`.
No across-seed interval is implied. Per-seed whole-reference/family intervals remain in the native read JSON.

## E36: 40 origins, 646,677 selection pairs, 10,000 draws

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| E36 inversion share (%) | 3.396647 | 3.148249 | 3.897773 | 3.595252 |
| E36 change vs A (percentage points), 95% source bootstrap | baseline | -0.248398 [-0.427255, -0.060774] | +0.501126 [+0.291565, +0.665201] | +0.198605 [+0.042429, +0.364538] |

## EVAL MOS median residual / target 1–99% span (%)

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| cid-iq (n=210) | 7.78359 | 8.98746 | 9.65205 | 8.86183 |
| cid-oa (n=91) | 8.41368 | 8.76049 | 9.96115 | 8.20911 |
| colourlab-rating-variability-2022 (n=192) | 11.15986 | 12.67576 | 12.62250 | 12.25193 |
| iqa-odi (n=48) | 7.86276 | 8.03897 | 7.91955 | 8.05816 |
| irccyn-ivc (n=60) | 6.65655 | 6.47936 | 6.83264 | 6.92206 |
| live-md (n=180) | 6.97772 | 6.79215 | 6.98499 | 6.73645 |
| mdid (n=480) | 6.19979 | 5.92247 | 6.33192 | 5.81896 |
| nits-iqa (n=405) | 11.20201 | 11.43363 | 11.22013 | 11.23910 |
| vcl-fer (n=192) | 5.99554 | 6.07780 | 5.70988 | 6.30069 |
| all nine, equal collection mean | 8.02794 | 8.35201 | 8.58165 | 8.26648 |
| eight without MDID (same training population) | 8.25646 | 8.65570 | 8.86286 | 8.57241 |
| common three: IQA-ODI, NITS-IQA, VCL-FER | 8.35343 | 8.51680 | 8.28319 | 8.53265 |

## EVAL MOS p90 residual / target 1–99% span (%)

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| cid-iq (n=210) | 21.66163 | 20.77877 | 23.93066 | 21.18466 |
| cid-oa (n=91) | 18.54655 | 18.09371 | 20.01365 | 17.78451 |
| colourlab-rating-variability-2022 (n=192) | 31.98866 | 31.47626 | 31.71791 | 32.53293 |
| iqa-odi (n=48) | 18.54617 | 17.90165 | 18.29372 | 17.29013 |
| irccyn-ivc (n=60) | 20.00259 | 19.15682 | 18.28876 | 19.47148 |
| live-md (n=180) | 15.91087 | 15.00794 | 14.96454 | 14.50307 |
| mdid (n=480) | 19.42745 | 18.17997 | 19.25602 | 18.33206 |
| nits-iqa (n=405) | 30.48377 | 30.47282 | 30.17137 | 30.67136 |
| vcl-fer (n=192) | 14.16272 | 15.61870 | 15.06158 | 15.09134 |

## EVAL MOS absolute SROCC

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| cid-iq (n=210) | 0.86925 | 0.86444 | 0.82682 | 0.86953 |
| cid-oa (n=91) | 0.90886 | 0.90751 | 0.89100 | 0.91410 |
| colourlab-rating-variability-2022 (n=192) | 0.73067 | 0.72747 | 0.72454 | 0.72357 |
| iqa-odi (n=48) | 0.86550 | 0.86156 | 0.86670 | 0.86633 |
| irccyn-ivc (n=60) | 0.91898 | 0.92392 | 0.92221 | 0.91210 |
| live-md (n=180) | 0.89622 | 0.90783 | 0.90599 | 0.91381 |
| mdid (n=480) | 0.92592 | 0.93117 | 0.92727 | 0.93089 |
| nits-iqa (n=405) | 0.73227 | 0.73532 | 0.73275 | 0.73293 |
| vcl-fer (n=192) | 0.94934 | 0.94879 | 0.95210 | 0.94848 |
| all nine, equal collection mean | 0.86634 | 0.86756 | 0.86104 | 0.86797 |
| eight without MDID (same training population) | 0.85889 | 0.85960 | 0.85276 | 0.86011 |
| common three: IQA-ODI, NITS-IQA, VCL-FER | 0.84904 | 0.84856 | 0.85052 | 0.84925 |

## EVAL compression: signed SROCC / within-reference SROCC / normalized median error

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| cid-iq (n=70) | 0.93332 / 0.94325 / 5.988% | 0.93083 / 0.94353 / 6.664% | 0.92701 / 0.94382 / 6.552% | 0.93104 / 0.94757 / 6.555% |
| colourlab-rating-variability-2022 (n=18) | 0.80083 / 0.83682 / 7.268% | 0.75048 / 0.83682 / 9.488% | 0.76841 / 0.84990 / 7.708% | 0.78186 / 0.83682 / 8.341% |
| iqa-odi (n=48) | 0.86550 / undefined / 7.863% | 0.86156 / undefined / 8.039% | 0.86670 / undefined / 7.920% | 0.86633 / undefined / 8.058% |
| irccyn-ivc (n=50) | 0.91171 / 0.97149 / 6.882% | 0.91706 / 0.97378 / 5.966% | 0.91695 / 0.97472 / 6.055% | 0.90159 / 0.97290 / 6.704% |
| nits-iqa (n=135) | 0.85176 / 0.88189 / 9.031% | 0.85702 / 0.88791 / 9.319% | 0.85632 / 0.88686 / 8.983% | 0.85889 / 0.88712 / 9.163% |
| vcl-fer (n=96) | 0.94934 / 0.98193 / 5.098% | 0.94830 / 0.98339 / 5.180% | 0.95043 / 0.98339 / 4.847% | 0.94933 / 0.98310 / 5.426% |
| row-weighted SROCC / equal-collection error | 0.89449 / 7.021% | 0.89355 / 7.442% | 0.89452 / 7.011% | 0.89447 / 7.375% |

## EVAL noncodec: signed SROCC / within-reference SROCC / normalized median error

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| cid-iq (n=140) | 0.83917 / 0.85950 / 9.572% | 0.83413 / 0.85867 / 10.958% | 0.78341 / 0.81857 / 12.111% | 0.84274 / 0.86935 / 10.851% |
| colourlab-rating-variability-2022 (n=174) | 0.70911 / 0.71955 / 11.875% | 0.70798 / 0.71902 / 13.261% | 0.70536 / 0.71349 / 13.282% | 0.70209 / 0.71032 / 13.551% |
| cid-oa (n=91) | 0.90886 / 0.99025 / 8.414% | 0.90751 / 0.99025 / 8.760% | 0.89100 / 0.98998 / 9.961% | 0.91410 / 0.98999 / 8.209% |
| irccyn-ivc (n=10) | 0.91515 / 1.00000 / 5.279% | 0.91515 / 1.00000 / 7.995% | 0.91515 / 1.00000 / 8.999% | 0.91515 / 1.00000 / 7.305% |
| live-md (n=90) | 0.89456 / 0.95396 / 6.341% | 0.91543 / 0.97153 / 6.653% | 0.91083 / 0.96796 / 6.751% | 0.92071 / 0.97123 / 5.822% |
| nits-iqa (n=270) | 0.66970 / 0.68805 / 12.750% | 0.67541 / 0.68507 / 12.802% | 0.66926 / 0.68434 / 12.727% | 0.67004 / 0.67893 / 12.743% |
| vcl-fer (n=96) | 0.95460 / 0.97028 / 5.853% | 0.95709 / 0.96795 / 5.625% | 0.95978 / 0.96503 / 5.250% | 0.95603 / 0.96591 / 5.365% |
| row-weighted SROCC / equal-collection error | 0.78725 / 8.584% | 0.79028 / 9.436% | 0.77779 / 9.869% | 0.78994 / 9.121% |

## G-CHROMA-L: dev views >64 px; MAE in score points against A0

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| ycc444 | 2.762208 | 2.587828 | 2.883024 | 2.643433 |
| ycc420 | 3.324290 | 3.070540 | 3.087318 | 3.349980 |
| xyb-capped | 2.060476 | 1.761096 | 2.010437 | 1.977961 |
| xyb-extended | 3.576321 | 4.029702 | 4.934336 | 3.946562 |
| band-ac | 2.138349 | 1.956934 | 2.245216 | 2.049520 |
| level pass / rank pass (seeds) | reference A0 | 0/3 / 3/3 | 0/3 / 3/3 | 0/3 / 3/3 |

## Label-free pixel and dial checks

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| G-STEER successes /135 | 133 / 134 / 134 | 135 / 135 / 135 | 135 / 135 / 135 | 135 / 135 / 135 |
| C1–C6, both standard and ladder grids | 3/3 pass | 3/3 pass | 3/3 pass | 3/3 pass |
| Identity: 744 scores exactly 100; read sets invariant | 3/3 pass | 3/3 pass | 3/3 pass | 3/3 pass |
| NEARID N1 / N2 | 3/3 / 3/3 pass | 3/3 / 3/3 pass | 3/3 / 3/3 pass | 3/3 / 3/3 pass |
| NEARID N3 monotone ladders /144 (bar 122) | 132 / 132 / 132 | 132 / 133 / 132 | 143 / 142 / 142 | 131 / 131 / 131 |

## HDR cube-root:203 — pooled signed SROCC

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 0.805356 | 0.807076 | 0.801357 | 0.797539 |
| ladder_eval (n=3245, refs=455) | 0.786123 | 0.774957 | 0.763546 | 0.781193 |
| hdrteach_val (n=3900, refs=300) | 0.891399 | 0.902268 | 0.924863 | 0.900700 |
| upiq_dev (n=50, refs=4) | 0.941128 | 0.937063 | 0.945194 | 0.935078 |
| upiq_all (n=380, refs=30) | 0.811412 | 0.812916 | 0.820184 | 0.809578 |

## HDR cube-root:203 — raw pooled PLCC

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 0.789547 | 0.776130 | 0.778071 | 0.775769 |
| ladder_eval (n=3245, refs=455) | 0.574024 | 0.557298 | 0.545878 | 0.566249 |
| hdrteach_val (n=3900, refs=300) | 0.976716 | 0.976509 | 0.977693 | 0.974144 |
| upiq_dev (n=50, refs=4) | 0.934591 | 0.933467 | 0.941437 | 0.930766 |
| upiq_all (n=380, refs=30) | 0.807453 | 0.807778 | 0.824092 | 0.803828 |

## HDR cube-root:203 — mean within-reference Spearman

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 0.816518 | 0.820333 | 0.816824 | 0.821188 |
| ladder_eval (n=3245, refs=455) | 0.987108 | 0.987886 | 0.983462 | 0.985617 |
| hdrteach_val (n=3900, refs=300) | 0.999982 | 0.999963 | 0.999835 | 0.999963 |
| upiq_dev (n=50, refs=4) | 0.966334 | 0.964585 | 0.967932 | 0.961089 |
| upiq_all (n=380, refs=30) | 0.962233 | 0.961829 | 0.961991 | 0.961412 |

## HDR cube-root:203 — within-reference inversion count

| Read | A | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| 4kdtb_eval (n=48, refs=4) | 45 / 44 / 45 | 46 / 46 / 46 | 46 / 47 / 47 | 43 / 44 / 44 |
| ladder_eval (n=3245, refs=455) | 50 / 51 / 49 | 49 / 58 / 42 | 72 / 91 / 68 | 53 / 53 / 54 |
| hdrteach_val (n=3900, refs=300) | 1 / 1 / 1 | 2 / 2 / 2 | 8 / 9 / 7 | 2 / 2 / 2 |
| upiq_dev (n=50, refs=4) | 15 / 15 / 15 | 15 / 15 / 16 | 17 / 15 / 14 | 16 / 16 / 16 |
| upiq_all (n=380, refs=30) | 121 / 121 / 124 | 126 / 125 / 125 | 127 / 126 / 125 | 125 / 124 / 126 |

Pooled ladder correlations are descriptive only; the registered ladder outcome is within-ladder rank.
4Kdtb has one family, so its family bootstrap interval is undefined. HDR is report-only; no HDR training leg was added.
The EVAL logistic fit is an assessment transform, never a fitted served-model update.
