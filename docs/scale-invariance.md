# Scale-invariance evaluation: zensim, SSIMULACRA2, butteraugli, DSSIM

> **⚠ BANNER 2026-07-18:** protocol reusable; results (2026-04-17) predate Profile A/B + the 372-feature IW pool. Re-run against B if scale-invariance regresses.

This document reports the first-run results from the scale-invariance protocol
defined in [issue #12](https://github.com/imazen/zensimx/issues/12). It answers
the practical question: *if I render the same source at two different resolutions
and apply the same codec/quality, do the metric scores stay comparable?*

## Setup

- **Date**: 2026-04-17
- **Generator**: `coefficient/examples/generate_scale_pyramid` (commit `ab41fac`)
- **Analyzer**: `zensim-validate --scale-invariance` (zensim commit `d2f715f`,
  weights `WEIGHTS_PREVIEW_V0_2`)
- **Sources**: 30 natural photographs (14 from CLIC 2025 training, 16 from
  Adobe FiveK expert C). All landscape, longer-edge ≥ 1920px.
- **Pyramid levels**: 1920×1080, 960×540, 480×270, 240×135 — three octaves of
  pixel count, fixed 16:9 aspect, Lanczos-3 downsampling.
- **Distortions per level**:
  - **Codec**: mozjpeg (effort 4), jpegli (effort 2), zenwebp (default), each
    at q ∈ {30, 50, 70, 85}
  - **Gaussian noise** (control, fixed σ in 0–255 luminance units): σ ∈ {2, 5, 10}
  - **Gaussian blur** (control, fixed σ in pixels): σ ∈ {0.5, 1.0, 2.0}
- **Total**: 30 sources × 4 levels × 18 distortions = **2160 pairs**
- **GPU**: RTX 5070, 82 s end-to-end generation.

The analyzer fits `score = α + β · log₂(pixel_count)` per `(source × codec ×
quality × distortion)` group across the 4 levels, then aggregates β.

## Headline results

| Distortion (all qualities/σ) | Metric | n | median \|β\| / oct | mean signed β | median R² |
|---|---|---:|---:|---:|---:|
| codec | DSSIM | 360 | **0.0001** | -0.0001 | 0.85 |
| codec | butteraugli | 360 | **0.081** | +0.077 | 0.61 |
| codec | zensim | 360 | **0.458** | -0.102 | 0.84 |
| codec | SSIMULACRA2 | 360 | **0.690** | -0.501 | 0.84 |
| gaussian_blur | DSSIM | 90 | 0.001 | -0.002 | 0.96 |
| gaussian_blur | butteraugli | 90 | 0.142 | +0.022 | 0.58 |
| gaussian_blur | zensim | 90 | 1.881 | +2.419 | 0.95 |
| gaussian_blur | SSIMULACRA2 | 90 | 1.861 | +2.305 | 0.92 |
| gaussian_noise | DSSIM | 90 | 0.0001 | +0.0002 | 0.98 |
| gaussian_noise | butteraugli | 90 | 0.112 | +0.123 | 0.90 |
| gaussian_noise | zensim | 90 | 0.890 | -1.139 | 0.99 |
| gaussian_noise | SSIMULACRA2 | 90 | 1.380 | -1.658 | 0.99 |

The full per-quality / per-σ table lives in
[`benchmarks/scale_invariance_2026-04-17.csv`](../benchmarks/scale_invariance_2026-04-17.csv);
per-group fits and the HTML report are in
`/mnt/v/output/zensim/scale-invariance/full/`.

### Codec block normalized to each metric's natural codec range

Raw |β| can mislead when metrics have very different dynamic ranges. Below,
|β|/oct is expressed as a percentage of each metric's IQR over codec rows
(median codec |β| at q ≥ 50, R² ≥ 0.8):

| Metric | codec IQR | median \|β\|/oct | \|β\| / IQR |
|---|---:|---:|---:|
| DSSIM | 0.0038 | 0.0001 | **3.8 %** |
| zensim | 15.0 | 0.61 | **4.1 %** |
| SSIMULACRA2 | 17.4 | 0.87 | **5.0 %** |
| butteraugli | 1.68 | 0.15 | **8.8 %** |

When normalized to its own codec range, **butteraugli has the largest relative
drift per octave**, despite its small absolute |β|. **DSSIM and zensim are
comparably the most scale-stable**; SSIMULACRA2 is moderately worse than zensim;
butteraugli is the worst.

## Drift narrows as quality rises

For codec distortions, all metrics show a monotone decrease in |β| as quality
improves:

| Quality | zensim med \|β\| | SSIMULACRA2 med \|β\| | butteraugli med \|β\| | DSSIM med \|β\| |
|---:|---:|---:|---:|---:|
| q=30 | 1.06 | 1.60 | 0.24 | 0.0005 |
| q=50 | 0.75 | 1.24 | 0.18 | 0.0003 |
| q=70 | 0.63 | 0.88 | 0.13 | 0.0002 |
| q=85 | 0.55 | 0.66 | 0.11 | 0.0001 |

This is the expected pattern: at low quality the per-resolution artifact
profile differs more (block-size mismatches, ringing, chroma-subsample
artifacts), giving the metrics more to disagree about. At high quality the
codec is close to transparent and there's less score variance for the slope to
capture.

## Synthetic controls track physics, not metric bugs

The two control distortion families have parameters chosen to be
*resolution-aware constants* — fixed σ in pixels (blur) or fixed σ in 0–255
luminance units (noise). Neither is *perceptually scale-invariant*: a 1-pixel
blur destroys ~50% of the visible detail in a 240×135 image but is barely
visible at 1920×1080. The metrics correctly track that:

| Control | metric | σ=0.5 | σ=1.0 | σ=2.0 |
|---|---|---:|---:|---:|
| blur | zensim med \|β\| | 0.95 | 2.65 | 4.57 |
| blur | SSIMULACRA2 med \|β\| | 1.05 | 3.30 | 4.84 |
| blur | butteraugli med \|β\| | 0.10 | 0.24 | 0.62 |
| blur | DSSIM med \|β\| | 0.000 | 0.001 | 0.004 |

| Control | metric | σ=2 | σ=5 | σ=10 |
|---|---|---:|---:|---:|
| noise | zensim med \|β\| | 0.26 | 0.94 | 2.37 |
| noise | SSIMULACRA2 med \|β\| | 0.55 | 1.38 | 3.08 |
| noise | butteraugli med \|β\| | 0.07 | 0.15 | 0.26 |
| noise | DSSIM med \|β\| | 0.000 | 0.000 | 0.000 |

**Direction matters and is consistent**: blur scores rise with resolution
(`mean signed β > 0` — fixed-pixel blur is less perceptually destructive at
higher res); noise scores fall with resolution under SSIMULACRA2 / zensim
(`mean signed β < 0` — fixed-amplitude noise is judged more harshly at higher
res, possibly because the metrics evaluate more spatial frequencies where
noise is visible). Butteraugli's noise direction flips positive (scores rise
with res) — different physical model.

DSSIM is almost completely flat on every control, including σ=10 noise. This
is a stronger statement than the codec result: DSSIM is genuinely
scale-invariant under the operations tested, not just stable for codec work.

## What this means for downstream use

For comparing codec output across resolutions:
- **DSSIM**: safe to compare directly across resolutions.
- **zensim**: ~4% relative drift per octave at high quality — usable for
  cross-resolution comparison if you tolerate that or compare same-resolution
  pairs.
- **SSIMULACRA2**: ~5% relative drift; same caveat as zensim, slightly worse.
- **butteraugli**: largest *relative* drift; prefer same-resolution comparisons.

For comparing fixed-σ blur or fixed-amplitude noise across resolutions:
**don't use any of zensim/SSIMULACRA2/butteraugli without first re-thinking
whether the distortion you're applying is meant to be physical (pixels) or
perceptual (relative content)**. The metric isn't lying — your distortion
model is.

## Reproducing this

```bash
# Generate (requires CUDA + coefficient with --features zenwebp)
coefficient/target/release/examples/generate_scale_pyramid \
    -s <source-dir-with-≥1920×1080-images> \
    -o <output-dir> \
    -r <storage-root>

# Analyze
zensim-validate \
    --scale-invariance <output-dir>/scale_pyramid.csv \
    --scale-invariance-out <output-dir>
```

Outputs:
- `scale_pyramid.csv` — raw per-pair metrics from generator
- `scale_pyramid_with_zensim.csv` — enriched with zensim_score column
- `scale_pyramid_fits.csv` — one row per (group × metric) with α, β, R², n
- `scale_pyramid_report.html` — summary table + worst-case slopes

To use a different aspect or pyramid: `--levels 1920x1280,960x640,480x320,240x160`.

## Follow-up A: text and UI content

The original 30 sources were all natural photographs. To check whether the
findings hold on rendered content, we generated 30 synthetic sources at
2048×1366 covering five subdomains:

- 8 articles (body text on white/cream/cool-white themes, 22–36 pt)
- 6 source-code panels (monospaced, syntax-coloured, dark theme)
- 6 dashboard / flat-UI mockups (header bar, sidebar, content cards)
- 6 spreadsheet-like tables (grid + numeric content)
- 4 terminal screenshots (dark BG, monospaced output)

Same matrix (4 levels × 3 codecs × 4 qualities + 6 controls = 18 distortions
per level), 2160 pairs, 73 s.

Results
([benchmarks/scale_invariance_text-ui_2026-04-17.csv](../benchmarks/scale_invariance_text-ui_2026-04-17.csv),
[…html](../benchmarks/scale_invariance_text-ui_2026-04-17.html)):

| Distortion | Metric | Photos med \|β\| | **Text/UI med \|β\|** | Photos med R² | Text/UI med R² |
|---|---|---:|---:|---:|---:|
| codec | DSSIM | 0.0001 | 0.0003 | 0.85 | 0.95 |
| codec | butteraugli | 0.081 | 0.103 | 0.61 | 0.59 |
| codec | zensim | 0.458 | **0.711** | 0.84 | 0.84 |
| codec | SSIMULACRA2 | 0.690 | **0.563** | 0.84 | **0.34** |
| gaussian_blur | DSSIM | 0.001 | 0.001 | 0.96 | 0.32 |
| gaussian_blur | butteraugli | 0.142 | **0.567** | 0.58 | 0.71 |
| gaussian_blur | SSIMULACRA2 | 1.861 | 1.115 | 0.92 | 0.21 |
| gaussian_blur | zensim | 1.881 | 1.351 | 0.95 | 0.56 |
| gaussian_noise | SSIMULACRA2 | 1.380 | **0.690** | 1.00 | 0.96 |
| gaussian_noise | zensim | 0.890 | **0.341** | 1.00 | 0.94 |

The two clearest content-dependent findings:

1. **zensim's codec scale bias is ~55% worse on text/UI than on photos**
   (0.71 vs 0.46 |β|/oct, mean signed +0.78 vs −0.10 — text scores rise with
   resolution, suggesting low-res text artifacts are penalized more heavily).
2. **SSIMULACRA2's relationship to log₂(pixel_count) on text codec runs is
   not even linear** (R² drops from 0.84 to 0.34) — slopes are noisy and the
   median is misleading. Treat ssim2 cross-resolution comparisons on text
   content with extra suspicion regardless of |β|.

Butteraugli's blur drift quadruples on text (0.14 → 0.57 |β|) — fixed-pixel
blur destroys text aggressively on small renders, and butteraugli weights
that more than the codec/photo case suggested.

## Follow-up B: filter sensitivity (spike)

To check whether the choice of downsampling filter biases the result, we
re-ran the same matrix on an 8-source photo subset with three filters:
Lanczos-3 (default), Catmull-Rom (a Mitchell-Netravali variant), and Triangle
(linear). Same 8 sources for all three runs; everything else unchanged.

Codec block, median |β|/oct
([benchmarks/scale_invariance_filter-spike_2026-04-17.csv](../benchmarks/scale_invariance_filter-spike_2026-04-17.csv)):

| Filter | DSSIM | butteraugli | zensim | SSIMULACRA2 |
|---|---:|---:|---:|---:|
| lanczos3 | 0.0001 | 0.095 | 0.497 | 0.767 |
| mitchell | 0.0001 | 0.106 | 0.488 | 0.747 |
| triangle | 0.0001 | 0.109 | 0.440 | 0.765 |

**Filter choice barely matters.** Differences across filters are ≤0.06 in
|β| — comfortably within IQR of an 8-source sample. The 30-source main
result with Lanczos-3 is not sensitive to that choice. (Note: an extreme
filter like `nearest` would likely move things; the filters tested are all
reasonable-quality kernels.)

## Caveats

1. **30 sources per content domain is small.** Per-codec slope distributions
   have IQRs of ~0.4–0.9 for ssim2/zensim under codec; with more sources
   we'd narrow those bounds.
2. **Two content domains tested** (natural photographs + synthetic text/UI).
   Mixed-content sources, animations, and rendered 3D scenes are not
   covered.
3. **Filter sensitivity tested across reasonable kernels only** (Lanczos /
   Mitchell / Triangle). A blocky / nearest-neighbour pyramid would behave
   differently.
4. **No effort tuning.** Codecs ran at default effort. A higher-effort
   encode of the same quality has a different artifact profile and may show
   different |β|.
5. **Embedded weights only.** Whether weight refits with explicit
   resolution balancing improve zensim's scale invariance is the next
   question and is its own follow-up.

## Follow-ups

- ~~Re-run with text/UI sources~~ ✅ done above
- ~~Sensitivity check on downsampling filter~~ ✅ done above
- Decide whether to add `log₂(pixel_count)` as a training signal so future
  weight refits explicitly target scale invariance.
