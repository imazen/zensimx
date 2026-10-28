# Comparison-site corpus inventory (2026-05-12)

All 5 in-repo human-rated corpora available via the interactive
comparison site at <https://imazen.github.io/zensimx/compare.html>.
Parquets live under `site/data/parquet/`, queried client-side via
DuckDB-WASM with HTTP-range fetch.

## Corpora at a glance

| Corpus | Rows | Cols | Codecs | Quality scale | Use case |
|---|---:|---:|---:|---|---|
| AIC-3 CTC EPFL    | 600   | 12 | 6  | reconstructed JND ∈ [-2.5, -0.25] | low-q sub-perceptual coverage; AVIF/HM/JPEG-1/JPEG-2000/JPEGXL/VVC × 10 ref × 10 q |
| AIC-4 sample      | 300   | 23 | 6  | reconstructed JND ∈ [0.0, 3.8] + CI bounds | structured high-fidelity; JPEG-AI included; paper PSNR/SSIM/MS-SSIM/IW-SSIM/VMAF-neg/SSIMULACRA2/HDR-VDP/CVVDP pre-computed |
| **CID22**         | 4341  | 11 | 10 | **MCOS (mean cardinal opinion score) ∈ [27.7, 91.9]** | **gold-standard codec-output; 49 ref × 10 codec families × ~10 settings each** |
| KADID-10k         | 10125 | 8  | 25 distortion types | DMOS ∈ [1, 5] | analytic distortion taxonomy (blur, noise, color, geometric — NOT compression) |
| TID2013           | 3000  | 7  | 24 distortion types | MOS ∈ [0.2, 7.2] | analytic distortion taxonomy (similar to KADID; older corpus) |

Total: **18,366 human-rated rows in ~251 KB of repo data.**

## Schema details

### AIC-3 CTC EPFL (`aic3_ctc_epfl.parquet`)

`corpus / ref_path / dist_path / image_name / codec / q / quality_index /
bpp / human_jnd / score_dssim / score_zensim / score_ssim2_gpu /
score_butter_max / score_butter_p3`

- `q` = encoder quality parameter (different meaning per codec)
- `quality_index` = 1..10, the human-rated quality level
- `human_jnd` is the reconstructed-JND distortion score: negative
  values are sub-JND distortion. The CID22 paper's PJND anchor is
  approximately at -1.0 here.

### AIC-4 sample (`aic4_sample.parquet`)

`corpus / ref_path / dist_path / image_name / codec / dlevel /
human_jnd / human_jnd_ci_lo / human_jnd_ci_hi / score_psnr_y / score_ssim /
score_ms_ssim / score_iw_ssim / score_vmaf_neg / score_ssim2_paper /
score_hdr_vdp_2 / score_hdr_vdp_3 / score_cvvdp / score_dssim /
score_zensim / score_ssim2_gpu / score_butter_max / score_butter_p3`

- `dlevel` = 1..10 distortion level
- `human_jnd` positive (different convention from AIC-3): higher =
  more distorted; JND ≈ 1 is the perceptibility threshold.
- `human_jnd_ci_lo / hi` = paper-published 95% confidence bounds for
  the reconstructed JND. These feed the candlestick whiskers when
  Y=human_jnd is selected.
- Paper pre-computed columns are the JPEG WG1 AIC-4 official
  evaluation reference numbers.

### CID22 (`cid22.parquet`) — the canonical zensim shipping gate

`corpus / ref_path / dist_path / image_name / codec / version /
bpp / human_mos / human_dmos / human_elo / nb_pc_opinions`

- `codec` = encoder name: `JPEG`, `JPEG_XL`, `JPEG_2000`, `WebP`,
  `HEIC`, `AVIF_aom_s1`, `AVIF_aom_s7`, `AVIF_aurora_fast`,
  `AVIF_aurora_slow`, `Reference`.
- `version` = setting string: `q30`, `q40`, ..., `e7_q30` (JXL effort
  7 + quality 30), `s1_q40` (AVIF speed 1 + quality 40), etc. 32
  distinct settings across all codecs.
- `human_mos` = MCOS (mean cardinal opinion score), 0..100 scale.
  This is the paper-canonical scale for CID22 SROCC numbers.
- `human_dmos` = RMOS (relative MOS), 0..1 normalized.
- `human_elo` = Elo from paired-comparison experiments.
- `nb_pc_opinions` = count of pairwise opinions contributing to the row.

### KADID-10k (`kadid.parquet`)

`corpus / ref_path / dist_path / image_name / codec / version /
human_dmos / human_dmos_var`

- `codec` = distortion type ID, `01`..`25`. Examples:
  - `01` = Gaussian blur
  - `06` = white noise
  - `13` = JPEG (compression artifacts)
  - `21` = brighten
  - etc. (full mapping in KADID-10k paper)
- `version` = distortion level, `01`..`05` (mild → severe).
- `human_dmos` ∈ [1, 5] where 1 = imperceptible, 5 = very annoying.
- ⚠️ KADID is **NOT compression-tuned**. Use as an integrity guard,
  not for codec-decision optimization.

### TID2013 (`tid.parquet`)

`corpus / ref_path / dist_path / image_name / codec / version /
human_mos`

- `codec` = distortion type ID, `01`..`24`.
- `version` = level `1`..`5`.
- `human_mos` ∈ [0.2, 7.2] (lower = worse).
- ⚠️ Like KADID, TID is **not compression-tuned**.

## Typical workflows

### "Codec rate-quality on CID22"

- Corpus: `cid22`
- X axis: `bpp`
- Y axis: `score_zensim_v0_16` (or `score_ssim2_gpu` for fast-ssim2)
- Filter codec: pick one to see a single curve, or leave at "(all)"
  to see all 10 codecs overlaid.
- Result: classic R-D scatter; CVVDP-class metrics top per-band SROCC.

### "What zensim score does each codec give me at JND=-1?"

- Corpus: `aic3_ctc_epfl`
- Y axis: `human_jnd`
- "Y → codec param lookup" form: target Y = `-1.0`, tolerance =
  `0.25`. Lists (codec, quality_level) hitting the perceptibility
  band, with median ssim2/dssim/zensim per group.

### "Compare V0_16 vs V0_20 across all corpora"

- Corpus: select all 5
- Y axis: `score_zensim_v0_16` then re-run with `score_zensim_v0_20`
- Compare per-band SROCC tables.
- ⚠️ Currently JS-MLP bake application needs `feat_*` columns in
  the parquet. AIC + CID + KADID + TID parquets don't carry them
  (the unified codec-sweep parquets do). Until R2 unlocks those,
  the JS-MLP path falls back to `score_zensim` (a single column;
  matches the bake the sweep was scored against at sweep time).

## Outstanding TODO

- **Codec-sweep parquets** (zenjpeg / zenavif / zenjxl / zenwebp /
  zenpng v12-v15) carrying `feat_0..feat_299` + sweep-time `score_*`
  columns: hosted on R2 once user enables public-read URL.
- **CID22 metric backfill** in flight (zenmetrics batch chain;
  /tmp/cid22_metrics/) — adds score_dssim / score_ssim2_gpu /
  score_butter_max / score_butter_p3 / score_zensim columns to
  cid22.parquet. ~50min total wall.
- **KADID + TID metric backfill** (separate ticks; ~3h each at
  scale, GPU sequential).
- **dssim co-training signal** experiment (cycle-7): the AIC
  combined per-codec table shows dssim is strongest on 6/12 cells;
  retraining V_X recipe with dssim as the truth signal (or as a
  co-training signal alongside ssim2) is a structural improvement
  candidate.
