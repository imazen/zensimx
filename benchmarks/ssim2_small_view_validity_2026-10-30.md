# SSIMULACRA2 is not valid on 32 px and 64 px chroma-ladder views (2026-10-30)

Question (owner, 2026-10-30): what happens to the chroma results if 3 px are cropped from every edge?
Answer: E, ECodec and butteraugli barely move. SSIMULACRA2 moves by up to +64 points, and only on 32 px views: the crop
changes the frame size, and SSIMULACRA2 itself misbehaves on small frames. The Cloudinary C++ reference agrees with our
frozen teacher, so this is the metric, not the port.

## Setup

- Population: the G-CHROMA held-out rows (roles dev + calibration, ladders with at least 3 kept rungs) of the stage-C
  chroma bank, 23,690 rows in 2,120 ladders. Long edges in the data: 32, 64, 128, 192, 256, 384, 512 px.
- For each rung, the exact decoded PNG from its EVIF; the reference is the view's source PNG. "Crop" removes 3 px from
  every edge of both images.
- Scorers: E and ECodec through `score_pair_with_bake` (`/var/tmp/zensim-bin-10d087cd`), the frozen SSIMULACRA2 teacher
  (`chroma-frozen-teacher`, fast-ssim2 0.8.2, sha256 `2e6d5f77…`), butteraugli from `zenmetrics-diagnostics compare`.
- Controls: all 23,690 full-frame teacher scores are bit-identical to the stored teacher bits; full-frame E and ECodec
  match the stored G-CHROMA scores within 5e-7.
- Commands: `scripts/byconstruction/chroma_crop3.py <out> --jobs 24` (run-heavy, dev, 128 s, peak RSS 0.85 GiB), then
  `scripts/byconstruction/chroma_crop3_analyze.py <out>/rows.jsonl`. Rows: dev `~/tmp/crop3/full/rows.jsonl`
  (sha256 `bbe91e4f…`).

## Near-lossless rung (q95, unit chroma) by frame size

| long edge | ladders | teacher median | E median | butteraugli p3 median |
|---|---:|---:|---:|---:|
| 32 | 290 | 64.4 | 92.1 | 0.324 |
| 64 | 287 | 83.5 | 92.3 | 0.354 |
| 128 | 227 | 93.1 | 91.9 | 0.380 |
| 192 | 260 | 92.9 | 91.6 | 0.386 |
| 256 | 526 | 92.4 | 91.2 | 0.398 |
| 384 | 264 | 92.5 | 91.0 | 0.397 |
| 512 | 266 | 92.0 | 90.8 | 0.412 |

Butteraugli rates the small views as good as or better than the large ones; SSIMULACRA2 scores them 9 to 28 points lower.

## The C++ reference agrees

Cloudinary's ssimulacra2 (repo HEAD `81feacf`, built from a copy of `src/` with the LCMS2 minimum lowered to 2.12;
binary sha256 `83c28e7b…`) on 144 sampled pairs (12 ladders per size, first/middle/last rung):

| long edge | pairs | max abs(teacher − C++) | teacher median | C++ median | E median | butteraugli p3 median |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 36 | 0.183 | 7.035 | 7.048 | 85.816 | 0.446 |
| 64 | 36 | 0.039 | 69.498 | 69.504 | 86.973 | 0.449 |
| 128 | 36 | 0.074 | 83.872 | 83.854 | 77.215 | 0.888 |
| 256 | 36 | 0.075 | 83.114 | 83.096 | 78.080 | 0.869 |

Example, a 32 px xyb-extended ladder: rung 0 teacher 7.6 (C++ 7.6), butteraugli p3 0.37, E 91.0.
On a 32×32 near-lossless pair, removing 1 px from every side (30×30) leaves the score near −13; removing 2 px (28×28)
gives 72. Mechanism not established.

## Full frame vs 3-px crop (E)

Mean absolute error against the teacher, held-out rows:

| view long edge | rows | E MAE full | E MAE crop | E within-ladder ρ full / crop |
|---|---:|---:|---:|---|
| ≤64 | 6,022 | 45.77 | 14.53 | 0.994 / 0.990 |
| 65–128 | 2,444 | 7.72 | 7.00 | 0.999 / 0.995 |
| 129–256 | 8,936 | 5.60 | 4.41 | 0.998 / 0.998 |
| 257–512 | 6,288 | 4.49 | 3.12 | 0.999 / 0.999 |

Mean absolute score change from the crop on ≤64 px views: teacher 32.3, E 2.2, ECodec 2.1, butteraugli p3 0.04.
On 32 px views alone the teacher rises by 64.3 on average.

## What it changes

- E's reported chroma-ladder error against the teacher (13.3–21.7 points per family) is mostly these views. On views
  of 128 px and up, E's error is 3.6–6.8.
- F's G-CHROMA-L "level" improvement came from the small views, where F moved toward the teacher's low scores (F − E
  mean score shift: −9.2 on 32 px, −6.3 on 64 px, +0.8 on ≥128 px). On ≥128 px views F's error against the teacher is
  higher than E's on band-ac (+0.94) and xyb-capped (+2.29), lower on ycc420 (−1.22), ycc444 (−0.53), xyb-extended
  (−0.34). Fcodec is the reverse: worse than ECodec on small views, better on ≥128 px in every family (−0.26 to −2.53).
- 21.1% of the chroma leg's fitting rows (293,318) have a long edge of 64 px or less, labelled by this teacher.
- E's SafeSyn training data (196,086 pairs, SSIMULACRA2-anchored) are all 257–1024 px, so E's labels are not affected.
- Any truth rule, gate or training leg that uses SSIMULACRA2 on frames of 64 px or less needs another reference there.
  Sizes between 65 and 127 px are not in this data and were not tested.

## Not done

- fast-ssim2 main was not built here; 0.8.2 already matches the C++ reference within 0.18.
- The cause inside SSIMULACRA2 (scale count, blur borders) was not traced.
