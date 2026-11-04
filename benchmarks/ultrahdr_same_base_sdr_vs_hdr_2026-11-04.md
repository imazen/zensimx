# Ultra HDR, same base encode: SDR score against HDR score (2026-11-04)

Owner question (2026-11-04): "how well does an ultrahdr image compressed (keeping the gain map intact) score when compared
in sdr vs hdr (applied gain map) with the same base sdr encode?" Nothing earlier measured this: the gain-map census of
2026-08-27 scored our own gain-map encodes against HDR references, a different question.

## Method

`zensim-ultrahdr-probe` (this repository, standalone root; zensimx `1eebc045`; binary sha256 `fddcd604…`, built on i270,
run on dev under a 24 GiB cap: 269 s, peak RSS 20.66 GiB with two image workers). For every input container and every
base quality q in {35, 45, 55, 65, 75, 82, 88, 94}: decode the base with the sibling ultrahdr-rs (zenjpeg), re-encode
it with zenjpeg (YCbCr 4:2:0) at q, reassemble the container with the **original gain-map JPEG and metadata
untouched** (`Encoder::encode_from_jpegs`), decode both containers the same way, and score

- **SDR**: original base against re-encoded base (zensim `F` through `compute`, SSIMULACRA2 on `RGB8_SRGB`, butteraugli's
  sRGB-u8 path, 3-norm and max);
- **HDR**: the gain map applied to both bases (`decode_hdr`, ultrahdr-rs's linear 1.0 = 203 cd/m²), scored as absolute
  light at a 4× display boost (812 cd/m² peak) and at the gain map's full boost: zensim `F` through `compute_hdr`
  (`HdrEncoding::Linear`, the PU21 front end), SSIMULACRA2's native HDR path (`RGBF32_LINEAR` nits with a 1 cd/m²
  diffuse white), butteraugli's linear path (nits over the boost's peak, `intensity_target` = that peak).

Inputs. **Camera**: the eight Ultra HDR JPEG originals of imazen-26 (Samsung Fold 7 / S25 Ultra; 4000×3000, one
4000×2252; gain maps 1000×750 single channel; declared max boost 4.92, so 4× and full nearly coincide), taken from
`/mnt/tower/output/imazen-26-originals-staging-2026-10-15/` (sha256s in the artifact folder). **Census**: nine of our own
gain-map encodes from the 2026-08-27 census (`/mnt/v/output/zenmetrics/gainmap-census-2026-08-27/*_t80_k3.jpg`; 384×512
to 1536×2048; single-channel gain maps at 1/4 scale; declared max boost 49.26, the PQ ceiling). Artifacts (TSVs, logs,
file lists, binary and input hashes): `/mnt/v/output/zensim/ultrahdr-probe-2026-11-04/` on dev, mirrored to
`/mnt/tower/output/zensim/ultrahdr-probe-2026-11-04/`.

## Reading

- **The HDR view of the same base encode is harsher than the SDR view, by a lot for camera files.** zensim `F` reads the
  camera pairs 12 points lower in HDR than in SDR at q75 (72.9 against 60.8), 16 points at q35 and 5 at q94; SSIMULACRA2
  4 points lower at q75 (73.6 against 69.5); butteraugli's 3-norm is 1.5× higher (1.17 against 1.78) and its max 1.6×.
  At the gain map's full boost the gaps grow by about half a point.
- **The gap is image dependent.** At q75 zensim's HDR minus SDR runs from -3 (the coffered ceiling) to -23 (the tiled
  arched interior, the forest sunset); SSIMULACRA2's from -1.6 to -13.5 on the same two images. The gain map multiplies
  the base's errors where it boosts, so images whose highlights carry most of the gain lose most.
- **Ordering is preserved.** Within an image every metric orders the q ladder identically in SDR and HDR (within-image
  rank agreement 1.000; butteraugli max 0.985). Across images and qualities the agreement is 0.93 (zensim), 0.97
  (SSIMULACRA2), 0.94 (butteraugli 3-norm), 0.89 (butteraugli max): the SDR score is a good proxy for *which* encode is
  better and a poor proxy for *how good* the HDR rendering is.
- **Equal score costs quality points.** zensim 73 in SDR at q75 needs about q88 in HDR (74.4); SSIMULACRA2 73.6 at q75
  needs about q82 (73.2). A pipeline that targets an SDR score for an Ultra HDR base over-compresses the HDR rendering.
- **Our own encodes barely show the effect.** On the census files zensim's HDR reading is 1.7 below SDR at q75 and level
  from q88 up; SSIMULACRA2 reads the HDR pair about one point *higher* than the SDR pair; butteraugli's 3-norm 0.26 higher.
  Their gain maps carry little actual boost at 1/4 scale (the declared 49× ceiling is not content), so the base's errors
  are amplified much less than under the camera files' real 4.9× highlights.
- **Caveat on scales.** zensim's SDR and HDR readings are different regimes: the HDR path feeds PU21-encoded absolute
  light to a model trained on SDR content (UPIQ pooled 0.78 for F), so its absolute HDR level is not human-anchored; the
  SDR-versus-HDR gap above is a statement about the metric's two readings of one encode, not about viewers. The same
  holds for SSIMULACRA2's native path and butteraugli's linear path.

## Tables (`summarize_probe.py`)

## out/camera.tsv: 8 images, 128 rows, boosts [4.0, 4.925, 4.926]

### display boost 4x (812 cd/m² peak)
| q | base kB | zensim SDR | zensim HDR | Δ | ssim2 SDR | ssim2 HDR | Δ | butter3 SDR | butter3 HDR | butter max SDR | butter max HDR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 453 | 57.68 | 41.56 | -16.12 | 58.25 | 52.39 | -5.86 | 1.692 | 2.558 | 4.088 | 6.411 |
| 45 | 501 | 60.98 | 45.72 | -15.26 | 61.88 | 56.37 | -5.51 | 1.589 | 2.405 | 3.780 | 6.019 |
| 55 | 566 | 64.86 | 50.78 | -14.08 | 65.83 | 60.77 | -5.05 | 1.464 | 2.209 | 3.521 | 5.460 |
| 65 | 645 | 68.50 | 55.51 | -12.99 | 69.92 | 65.48 | -4.44 | 1.328 | 2.003 | 3.176 | 4.886 |
| 75 | 793 | 72.86 | 60.82 | -12.04 | 73.61 | 69.52 | -4.10 | 1.170 | 1.776 | 2.728 | 4.245 |
| 82 | 958 | 76.52 | 65.49 | -11.03 | 76.91 | 73.23 | -3.68 | 1.043 | 1.597 | 2.365 | 3.821 |
| 88 | 1335 | 82.50 | 74.41 | -8.09 | 82.34 | 79.36 | -2.97 | 0.817 | 1.266 | 1.960 | 3.082 |
| 94 | 2239 | 88.12 | 83.18 | -4.94 | 88.01 | 85.81 | -2.20 | 0.543 | 0.882 | 1.439 | 2.330 |

- zensim: SDR vs HDR rank agreement across all rows 0.929, within-image mean 1.000; HDR minus SDR median -10.089, 10th/90th pct -24.163 / -2.939
- ssim2: SDR vs HDR rank agreement across all rows 0.966, within-image mean 1.000; HDR minus SDR median -2.816, 10th/90th pct -9.519 / -1.446
- butter3: SDR vs HDR rank agreement across all rows 0.942, within-image mean 1.000; HDR minus SDR median +0.584, 10th/90th pct +0.325 / +0.990
- buttermax: SDR vs HDR rank agreement across all rows 0.886, within-image mean 0.985; HDR minus SDR median +1.320, 10th/90th pct +0.717 / +3.026

| image | gain-map max boost | zensim SDR q75 | zensim HDR q75 | Δ | ssim2 SDR q75 | ssim2 HDR q75 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1228_interiors_stained-glass-windows_casa-batllo | 4.92 | 71.06 | 53.07 | -17.99 | 71.34 | 68.32 | -3.02 |
| 1231_interiors_tiled-arched-interior_casa-batllo | 4.93 | 71.81 | 48.89 | -22.92 | 71.50 | 58.03 | -13.47 |
| 1233_interiors_cathedral-ceiling-interior_sagrad | 4.92 | 74.56 | 68.99 | -5.57 | 74.17 | 71.85 | -2.32 |
| 1239_interiors_decorative-coffered-ceiling_casa- | 4.92 | 76.29 | 73.08 | -3.21 | 76.39 | 74.80 | -1.59 |
| 1245_interiors_circular-building-interior_la-ped | 4.92 | 74.23 | 64.46 | -9.77 | 74.47 | 72.22 | -2.24 |
| 1493_nature_sunset-over-sea_lower-davie-vancouve | 4.92 | 70.71 | 59.91 | -10.81 | 75.05 | 72.81 | -2.25 |
| 1494_nature_sunset-over-forest_colorado_zfold7_i | 4.92 | 66.09 | 44.48 | -21.60 | 67.40 | 61.85 | -5.55 |
| 1639_food_sashimi-on-plate_colorado_s25u_iso200- | 4.92 | 78.10 | 73.69 | -4.41 | 78.59 | 76.25 | -2.34 |

### display boost full (gain map full boost, per image)
| q | base kB | zensim SDR | zensim HDR | Δ | ssim2 SDR | ssim2 HDR | Δ | butter3 SDR | butter3 HDR | butter max SDR | butter max HDR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 453 | 57.68 | 40.91 | -16.77 | 58.25 | 50.91 | -7.34 | 1.692 | 2.620 | 4.088 | 6.608 |
| 45 | 501 | 60.98 | 45.08 | -15.90 | 61.88 | 54.97 | -6.91 | 1.589 | 2.464 | 3.780 | 6.182 |
| 55 | 566 | 64.86 | 50.18 | -14.68 | 65.83 | 59.49 | -6.34 | 1.464 | 2.263 | 3.521 | 5.611 |
| 65 | 645 | 68.50 | 54.94 | -13.56 | 69.92 | 64.36 | -5.56 | 1.328 | 2.052 | 3.176 | 5.017 |
| 75 | 793 | 72.86 | 60.26 | -12.59 | 73.61 | 68.47 | -5.15 | 1.170 | 1.820 | 2.728 | 4.352 |
| 82 | 958 | 76.52 | 64.96 | -11.57 | 76.91 | 72.30 | -4.62 | 1.043 | 1.637 | 2.365 | 3.945 |
| 88 | 1335 | 82.50 | 73.95 | -8.54 | 82.34 | 78.59 | -3.74 | 0.817 | 1.300 | 1.960 | 3.171 |
| 94 | 2239 | 88.12 | 82.84 | -5.28 | 88.01 | 85.23 | -2.78 | 0.543 | 0.909 | 1.439 | 2.409 |

- zensim: SDR vs HDR rank agreement across all rows 0.921, within-image mean 1.000; HDR minus SDR median -10.454, 10th/90th pct -25.615 / -3.069
- ssim2: SDR vs HDR rank agreement across all rows 0.960, within-image mean 1.000; HDR minus SDR median -3.630, 10th/90th pct -11.944 / -1.816
- butter3: SDR vs HDR rank agreement across all rows 0.928, within-image mean 1.000; HDR minus SDR median +0.616, 10th/90th pct +0.343 / +1.086
- buttermax: SDR vs HDR rank agreement across all rows 0.881, within-image mean 0.982; HDR minus SDR median +1.407, 10th/90th pct +0.755 / +3.270

| image | gain-map max boost | zensim SDR q75 | zensim HDR q75 | Δ | ssim2 SDR q75 | ssim2 HDR q75 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1228_interiors_stained-glass-windows_casa-batllo | 4.92 | 71.06 | 52.49 | -18.57 | 71.34 | 67.45 | -3.89 |
| 1231_interiors_tiled-arched-interior_casa-batllo | 4.93 | 71.81 | 47.07 | -24.74 | 71.50 | 54.85 | -16.65 |
| 1233_interiors_cathedral-ceiling-interior_sagrad | 4.92 | 74.56 | 68.82 | -5.74 | 74.17 | 71.22 | -2.95 |
| 1239_interiors_decorative-coffered-ceiling_casa- | 4.92 | 76.29 | 72.96 | -3.33 | 76.39 | 74.34 | -2.05 |
| 1245_interiors_circular-building-interior_la-ped | 4.92 | 74.23 | 64.17 | -10.06 | 74.47 | 71.61 | -2.85 |
| 1493_nature_sunset-over-sea_lower-davie-vancouve | 4.92 | 70.71 | 59.47 | -11.25 | 75.05 | 72.15 | -2.90 |
| 1494_nature_sunset-over-forest_colorado_zfold7_i | 4.92 | 66.09 | 43.60 | -22.49 | 67.40 | 60.44 | -6.96 |
| 1639_food_sashimi-on-plate_colorado_s25u_iso200- | 4.92 | 78.10 | 73.51 | -4.58 | 78.59 | 75.68 | -2.91 |
## out/census.tsv: 9 images, 144 rows, boosts [4.0, 49.261]

### display boost 4x (812 cd/m² peak)
| q | base kB | zensim SDR | zensim HDR | Δ | ssim2 SDR | ssim2 HDR | Δ | butter3 SDR | butter3 HDR | butter max SDR | butter max HDR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 107 | 61.21 | 56.73 | -4.47 | 59.93 | 61.81 | +1.87 | 1.921 | 2.314 | 4.718 | 5.645 |
| 45 | 119 | 64.98 | 61.18 | -3.79 | 63.69 | 65.44 | +1.76 | 1.754 | 2.117 | 4.300 | 5.053 |
| 55 | 135 | 69.23 | 66.16 | -3.07 | 68.00 | 69.57 | +1.57 | 1.554 | 1.874 | 3.885 | 4.607 |
| 65 | 158 | 73.84 | 71.54 | -2.31 | 72.76 | 74.09 | +1.33 | 1.337 | 1.610 | 3.301 | 4.017 |
| 75 | 194 | 80.17 | 78.51 | -1.67 | 79.08 | 80.18 | +1.10 | 1.038 | 1.263 | 2.689 | 3.307 |
| 82 | 220 | 83.05 | 81.85 | -1.19 | 81.94 | 82.87 | +0.93 | 0.915 | 1.117 | 2.451 | 3.027 |
| 88 | 275 | 88.76 | 88.78 | +0.02 | 87.94 | 88.45 | +0.51 | 0.658 | 0.798 | 1.962 | 2.434 |
| 94 | 328 | 89.98 | 90.04 | +0.06 | 89.77 | 90.21 | +0.44 | 0.597 | 0.727 | 1.872 | 2.290 |

- zensim: SDR vs HDR rank agreement across all rows 0.990, within-image mean 0.997; HDR minus SDR median -1.125, 10th/90th pct -5.241 / +0.457
- ssim2: SDR vs HDR rank agreement across all rows 0.999, within-image mean 1.000; HDR minus SDR median +1.168, 10th/90th pct +0.460 / +1.837
- butter3: SDR vs HDR rank agreement across all rows 0.997, within-image mean 0.997; HDR minus SDR median +0.259, 10th/90th pct +0.119 / +0.384
- buttermax: SDR vs HDR rank agreement across all rows 0.993, within-image mean 0.976; HDR minus SDR median +0.630, 10th/90th pct +0.286 / +0.963

| image | gain-map max boost | zensim SDR q75 | zensim HDR q75 | Δ | ssim2 SDR q75 | ssim2 HDR q75 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1064_general_castle-bridge-moat_montjuic-castle- | 49.26 | 79.42 | 77.83 | -1.60 | 77.25 | 78.25 | +1.00 |
| 1065_general_cathedral-model-workshop_sagrada-fa | 49.26 | 88.08 | 88.24 | +0.17 | 86.95 | 87.70 | +0.76 |
| 1229_interiors_lighted-archway-ceiling_casa-batl | 49.26 | 76.52 | 75.25 | -1.26 | 74.32 | 75.38 | +1.06 |
| 1242_interiors_storage-room-clutter_la-pedrera-b | 49.26 | 78.87 | 76.54 | -2.33 | 77.15 | 78.62 | +1.47 |
| 1494_nature_sunset-over-forest_colorado_zfold7_i | 49.26 | 75.92 | 68.22 | -7.70 | 75.46 | 77.22 | +1.75 |
| 1495_nature_lighthouse-on-island_york-maine_ip13 | 49.26 | 78.12 | 75.47 | -2.65 | 77.99 | 79.10 | +1.11 |
| 1520_nature_tulips-in-garden_notre-dame-versaill | 49.26 | 87.27 | 88.31 | +1.04 | 85.77 | 86.36 | +0.59 |
| 1639_food_sashimi-on-plate_colorado_s25u_iso200- | 49.26 | 79.95 | 80.15 | +0.19 | 79.45 | 80.53 | +1.08 |
| 1640_food_birthday-cake-table_colorado_ip16pro_i | 49.26 | 77.38 | 76.53 | -0.85 | 77.38 | 78.49 | +1.11 |

### display boost full (gain map full boost, per image)
| q | base kB | zensim SDR | zensim HDR | Δ | ssim2 SDR | ssim2 HDR | Δ | butter3 SDR | butter3 HDR | butter max SDR | butter max HDR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 35 | 107 | 61.21 | 57.21 | -4.00 | 59.93 | 61.79 | +1.85 | 1.921 | 2.312 | 4.718 | 5.641 |
| 45 | 119 | 64.98 | 61.57 | -3.40 | 63.69 | 65.41 | +1.72 | 1.754 | 2.115 | 4.300 | 5.061 |
| 55 | 135 | 69.23 | 66.54 | -2.69 | 68.00 | 69.49 | +1.49 | 1.554 | 1.872 | 3.885 | 4.615 |
| 65 | 158 | 73.84 | 71.80 | -2.04 | 72.76 | 74.00 | +1.24 | 1.337 | 1.611 | 3.301 | 4.080 |
| 75 | 194 | 80.17 | 78.72 | -1.45 | 79.08 | 80.12 | +1.04 | 1.038 | 1.263 | 2.689 | 3.345 |
| 82 | 220 | 83.05 | 82.01 | -1.04 | 81.94 | 82.74 | +0.80 | 0.915 | 1.117 | 2.451 | 3.015 |
| 88 | 275 | 88.76 | 88.87 | +0.10 | 87.94 | 88.26 | +0.32 | 0.658 | 0.804 | 1.962 | 2.473 |
| 94 | 328 | 89.98 | 90.12 | +0.14 | 89.77 | 89.95 | +0.18 | 0.597 | 0.732 | 1.872 | 2.312 |

- zensim: SDR vs HDR rank agreement across all rows 0.989, within-image mean 0.997; HDR minus SDR median -0.836, 10th/90th pct -4.941 / +0.659
- ssim2: SDR vs HDR rank agreement across all rows 0.998, within-image mean 1.000; HDR minus SDR median +1.064, 10th/90th pct +0.300 / +1.842
- butter3: SDR vs HDR rank agreement across all rows 0.997, within-image mean 1.000; HDR minus SDR median +0.256, 10th/90th pct +0.125 / +0.385
- buttermax: SDR vs HDR rank agreement across all rows 0.990, within-image mean 0.989; HDR minus SDR median +0.641, 10th/90th pct +0.313 / +1.005

| image | gain-map max boost | zensim SDR q75 | zensim HDR q75 | Δ | ssim2 SDR q75 | ssim2 HDR q75 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1064_general_castle-bridge-moat_montjuic-castle- | 49.26 | 79.42 | 77.92 | -1.51 | 77.25 | 77.84 | +0.59 |
| 1065_general_cathedral-model-workshop_sagrada-fa | 49.26 | 88.08 | 88.46 | +0.38 | 86.95 | 87.94 | +1.00 |
| 1229_interiors_lighted-archway-ceiling_casa-batl | 49.26 | 76.52 | 75.44 | -1.08 | 74.32 | 74.98 | +0.66 |
| 1242_interiors_storage-room-clutter_la-pedrera-b | 49.26 | 78.87 | 76.77 | -2.10 | 77.15 | 78.60 | +1.45 |
| 1494_nature_sunset-over-forest_colorado_zfold7_i | 49.26 | 75.92 | 68.63 | -7.29 | 75.46 | 77.13 | +1.66 |
| 1495_nature_lighthouse-on-island_york-maine_ip13 | 49.26 | 78.12 | 75.50 | -2.62 | 77.99 | 79.10 | +1.11 |
| 1520_nature_tulips-in-garden_notre-dame-versaill | 49.26 | 87.27 | 88.43 | +1.16 | 85.77 | 86.23 | +0.46 |
| 1639_food_sashimi-on-plate_colorado_s25u_iso200- | 49.26 | 79.95 | 80.42 | +0.47 | 79.45 | 80.65 | +1.20 |
| 1640_food_birthday-cake-table_colorado_ip16pro_i | 49.26 | 77.38 | 76.93 | -0.46 | 77.38 | 78.58 | +1.20 |
