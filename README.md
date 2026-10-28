# zensim [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zensimx/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zensimx/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zensim?style=flat-square)](https://crates.io/crates/zensim) [![lib.rs](https://img.shields.io/crates/v/zensim?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zensim) [![docs.rs](https://img.shields.io/docsrs/zensim?style=flat-square)](https://docs.rs/zensim) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/zensim?style=flat-square)](#license)

> **This is zensimx, the zensim research repository** (renamed from imazen/zensim on 2026-10-28):
> training, evaluation, experiment records and the research API. The published `zensim` crate is
> developed and released from [imazen/zensim](https://github.com/imazen/zensim), which continues from
> zensim 0.2.7. The `zensim` and `zensim-regress` packages here are `publish = false`.

A perceptual image quality score for codec quality targeting: users choose one
score, and encoders choose the settings to reach it.

Built on the same psychovisual foundations as SSIMULACRA2 and butteraugli — multi-scale SSIM, edge artifacts, detail loss, and high-frequency features in XYB color space — but with trained weights, fused SIMD kernels, and multi-threaded computation.

**Current checkout (2026-10-19):** the user-facing control is one target score.
`codec_target()` returns **B / zensim-b: diff+frag-820 (Rev5)**; D is an explicit fast profile. See the [B promotion record](https://github.com/imazen/zensimx/blob/main/docs/PROFILE_B_DIFF_FRAG_820_2026-10-19.md) for its exact bake, known color-noise inversions and remaining qualification. Negative scores are
valid. The benchmark tables below are historical measurements, not a fresh
qualification of today's profiles. See the [current integration guide](https://github.com/imazen/zensimx/blob/main/docs/CODEC_TARGET_METRIC.md)
for bake identities, actual consumer behavior and the remaining dial requirements.

**Research continuation (2026-10-27):** Mlic recipe comparisons and AIC TRAIN
adaptation are active; no new serving model has been adopted. Agent/operator
state, scientific boundaries and verified run commands are in
[SESSION-RESUME](SESSION-RESUME.md#current-work--october-27-2026).

**Interactive chart exploration**: <https://imazen.github.io/zensim/> — scatter zensim / fast-ssim2 / butteraugli against human MOS across CID22 / KADID / TID / AIC corpora, filter by codec + version, with per-band SROCC tables and step-5 (20-bin) breakdowns.

<!-- crates.io:skip-start -->
## Historical speed measurements

AMD Ryzen 9 7950X 16C/32T (WSL2), synthetic gradient images, no I/O, pre-allocated buffers. zensim and ssimulacra2-rs use rayon (all cores); C++ libjxl, fast-ssim2, and butteraugli-rs are single-threaded. Enabling rayon for fast-ssim2 and butteraugli-rs made them slower at small sizes due to thread-pool overhead, so they're benchmarked single-threaded. Median of 100 samples via criterion.

### SSIMULACRA2 implementations

| Resolution | zensim | zensim (1 thread) | C++ libjxl (FFI) | fast-ssim2 | ssimulacra2-rs |
|------------|-------:|------------------:|-----------------:|-----------:|---------------:|
| 1280x720 | **14 ms** | 39 ms | 249 ms | 111 ms | 545 ms |
| 1920x1080 | **22 ms** | 89 ms | 377 ms | 350 ms | 1,056 ms |
| 3840x2160 | **91 ms** | 366 ms | 1,674 ms | 1,364 ms | 3,980 ms |

### Butteraugli implementations (single-threaded)

| Resolution | C++ libjxl (FFI) | butteraugli-rs |
|------------|----------------:|---------------:|
| 1280x720 | 269 ms | 83 ms |
| 1920x1080 | 647 ms | 154 ms |
| 3840x2160 | 2,688 ms | 906 ms |

Single-threaded zensim is 4x faster than C++ libjxl SSIMULACRA2. Multi-threaded at 4K: 18x.

Reproduce: `cargo bench -p zensim-bench --bench bench_compare` (C++ libjxl FFI requires a local libjxl build; set `LIBJXL_DIR` or let the build script auto-clone it)
<!-- crates.io:skip-end -->

## Correlation with human perception

Quality targeting requires both perceptual agreement and a useful score range.
The tables below preserve the older v0.3/A evaluation; B's dated methodology is
in its [July record](https://github.com/imazen/zensimx/blob/main/benchmarks/profile_b_methodology_2026-07-12.md).
They do not qualify the current profiles against September's stricter dial
requirements. CID22 has also been consulted repeatedly during model selection;
it is not an untouched final test. Current split policy and exceptions are in
[DATA_SPLITS](https://github.com/imazen/zensimx/blob/main/docs/DATA_SPLITS.md).

<!-- crates.io:skip-start -->
<details>
<summary>Historical v0.3/A measurements and score mappings</summary>

These results retain their original measurement scope. In particular, an old
JND-at-60 convention or zero-inversion result on one grid is not a universal
guarantee for today's model. AIC targets below include designed or reconstructed
JND levels; they are not interchangeable with direct human MOS.

Higher SROCC + PLCC + KROCC + PWRC is better; lower OR + Z-RMSE is better.

### CID22 — codec compression artifacts (n=4,292)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| **zensim v0.3** | 0.860 | 0.853 | 0.673 | 0.045 | 0.909 | 0.523 |
| fast-ssim2 (SSIMULACRA2) | **0.890** | **0.888** | **0.706** | 0.042 | **0.935** | **0.460** |
| cvvdp (ColorVideoVDP) | 0.821 | 0.825 | 0.624 | 0.042 | 0.884 | 0.565 |
| iwssim (Wang & Li 2011) | 0.784 | 0.793 | 0.594 | 0.052 | 0.853 | 0.610 |

### AIC-3 CTC — JPEG-AIC compression at JND levels (n=600)

| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
|---|---:|---:|---:|---:|---:|---:|
| zensim v0.3 | 0.776 | 0.788 | 0.607 | 0.042 | 0.854 | 0.616 |
| fast-ssim2 | **0.797** | **0.809** | **0.629** | 0.057 | **0.872** | 0.588 |
| cvvdp | 0.792 | 0.803 | 0.626 | 0.042 | 0.866 | **0.595** |
| iwssim | 0.774 | 0.791 | 0.606 | 0.045 | 0.854 | 0.612 |

### AIC-4 sample — JPEG-AIC reconstructed JND, 6 codecs (n=300)

| Metric | SROCC |
|---|---:|
| zensim v0.3 | 0.928 |
| fast-ssim2 | baselines pending |
| cvvdp | baselines pending |
| iwssim | baselines pending |

The AIC-4 baseline scores for ssim2/cvvdp/iwssim haven't been folded
into our standard panel yet — the raw per-pair metric scores live at
`/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG-AIC_metric_scores.csv`
but the SROCC computation against the reconstructed-JND target isn't
in the panel doc. v0.3's AIC-4 SROCC of **0.928** is from
`bake_verdict` on the 300-pair val parquet at
`canonical-2026-05-18/val/aic4.parquet`.

### CID22 per-band SROCC (10 width-10 bins on the human-MOS scale)

Per `CLAUDE.md` "10-band reporting rule": the primary release gate is the per-band picture, not the aggregate. Below-PJND bands (B3–B5) are the hard ones because the human-MOS scores in those bands are noisy and bunched. Bands B0–B2 have ≤ 1 sample on CID22 and are omitted.

| Band | range | n | **v0.3** | ssim2 | cvvdp | iwssim |
|---|---|--:|---:|---:|---:|---:|
| B3 | [0.30, 0.40) | 57 | 0.051 | **0.134** | 0.148 | 0.096 |
| B4 | [0.40, 0.50) | 266 | 0.230 | 0.289 | 0.260 | 0.210 |
| B5 | [0.50, 0.60) | 615 | 0.273 | **0.389** | 0.290 | 0.193 |
| B6 | [0.60, 0.70) | 836 | 0.287 | **0.417** | 0.336 | 0.210 |
| B7 | [0.70, 0.80) | 1092 | **0.408** | 0.397 | 0.310 | 0.283 |
| B8 | [0.80, 0.90) | 1382 | 0.500 | **0.501** | 0.319 | 0.413 |
| B9 | [0.90, 1.00] | 43 | **0.220** | 0.112 | 0.081 | 0.134 |

Per-band read: ssim2 wins B5–B6 (where most CID22 mass sits); v0.3 wins B7 (good-quality region) and B9 (near-lossless tail). v0.3 essentially matches ssim2 on B8 (the dominant band). cvvdp + iwssim are weakest across every band — they're stronger as aggregate metrics than per-band rank predictors here.

CID22 per-band Z-RMSE (lower better):

| Band | n | **v0.3** | ssim2 | cvvdp | iwssim |
|---|--:|--:|--:|--:|--:|
| B3 | 57 | 0.950 | **0.947** | 0.990 | 0.989 |
| B4 | 266 | 0.959 | **0.947** | 0.962 | 0.965 |
| B5 | 615 | 0.959 | **0.921** | 0.954 | 0.971 |
| B6 | 836 | 0.957 | **0.908** | 0.941 | 0.972 |
| B7 | 1092 | **0.912** | 0.907 | 0.947 | 0.954 |
| B8 | 1382 | 0.866 | 0.866 | 0.947 | 0.909 |
| B9 | 43 | **0.937** | 0.940 | 0.952 | 0.854 |

ssim2 has the tightest Z-RMSE in the mid bands (B4–B6) — this is the ssim2-target training-bias caveat from CLAUDE.md materializing. v0.3 and ssim2 are tied on B7–B8. v0.3 wins the noisy tails (B9 near-lossless).

### Per-corpus headline

- **CID22**: ssim2 wins SROCC by 0.03; v0.3 is second. Note that CLAUDE.md's "SROCC-only verdicts BANNED" caveat applies — older trainers used ssim2-derived targets, which biases SROCC measurements toward ssim2-shaped surfaces. v0.3's Z-RMSE (0.523) trails ssim2's 0.460.
- **AIC-3**: 4-way tie within 0.02 SROCC; ssim2 nominally best.
- **AIC-4**: v0.3 SROCC = 0.928. cvvdp / ssim2 / iwssim baselines on AIC-4 haven't been computed into our panel doc yet (raw scores at `JPEG-AIC_metric_scores.csv`; SROCC against reconstructed-JND target is a TODO).
- **None of the four hits all three holdouts** — v0.3 trades 0.03 CID22 SROCC for full 0-100 dial coverage + JND@60-bit-exact + per-source PJND tracking (the dial properties that matter for codec targeting). See [`docs/CODEC_TARGET_METRIC.md`](docs/CODEC_TARGET_METRIC.md).

v0.2 (default-on linear profile through zensim 0.2.x): 228 linear weights × basic+peak features, trained on 218k concordance-filtered synthetic pairs via Nelder-Mead.

`A`'s May 27 bake used a 372-input MLP (372 → 128 → 64 + per-sample-α head + tanh-output pin) with a monotone 7-knot PCHIP dial spline: 27 KB packed (`v47-strict-QAT`, f16 + zerobias, file `zensim/weights/v47_strict_qat_native_2026-05-27.bin`). Its recorded grid had 0 inversions, 0 above-identity, identity = 97.69. Trained on 5 groups (safesyn 196k + cid22_train 17.6k + kadid 10.1k + tid 3k + konjnd_dense 20.2k) via one QAT-native pass. Evaluation SROCC: CID22 0.866, KADID 0.793, TID 0.793, KonJND 0.419, AIC-3 0.768, AIC-4 0.885; these groups are not all holdouts. Methodology: [`benchmarks/v0_qat_native_methodology_2026-05-27.md`](benchmarks/v0_qat_native_methodology_2026-05-27.md). A is now deprecated and its bake was converted to explicit feature IDs on September 6.

`C`'s original August 5 bake recorded CID22 0.8867, LIVE 0.9604, CSIQ 0.9331 and dial monotonicity 99.3%. Those are historical results from [`PROFILE_C_REPRODUCTION`](docs/PROFILE_C_REPRODUCTION_2026-08-05.md): C changed bakes on August 29 and can now be served through `Zensim::compute`. Its train/serve activity discrepancy was corrected on September 7 with explicit-ID bakes; pixel scores intentionally changed. See the [current integration guide](docs/CODEC_TARGET_METRIC.md).

### v0.2 → v0.3 rough score equivalence

Both profiles span 0..100 but use the dial differently. v0.2's linear formula `100 − 18·|d|^0.7` floor-clamps below moderate distortion (38k of 68k cross-codec pairs land at v0.2 ≤ 5, while v0.3 spreads them across 28..50). v0.3 uses the full 0-100 dial with JND landing at exactly score 60. Rough lookup on 68,788 matched cross-codec pairs (Spearman v0.2 ↔ v0.3 = 0.88):

| v0.2 target | v0.3 median (p25 → p75) | rough quality region |
|--:|--:|---|
| 10 | 52.6 (47.8 → 55.6) | low-q, dial floor |
| 20 | 55.2 (52.4 → 58.2) | sub-PJND |
| 30 | 58.9 (55.0 → 60.5) | approaching JND |
| 40 | 61.0 (58.9 → 63.6) | just past JND |
| 50 | 65.8 (64.3 → 66.9) | mid-PJND |
| 60 | 68.8 (66.4 → 72.5) | comfortable quality |
| 70 | 78.4 (76.5 → 80.3) | good compression |
| 80 | 90.0 (87.6 → 91.1) | near-lossless |
| 90 | 96.6 (95.1 → 97.3) | visually lossless |
| 100 | 100 (exact, byte-identical short-circuit) | lossless |

The mapping is non-linear because v0.2's clamp at low quality compresses the 0-30 range into a single floor; v0.3 differentiates that region. For users targeting "score 70" in v0.2 code, the v0.3 equivalent is roughly **score 78**.

<details>
<summary>Reproduce these numbers</summary>

Download the datasets ([instructions below](#downloading-evaluation-datasets)), then:

```bash
# CID22 — expects CID22_validation_set.csv + original/ and compressed/ dirs
cargo run --release -p zensim-validate -- --dataset ./datasets/cid22 --format cid22

# TID2013 — expects mos_with_names.txt + reference_images/ and distorted_images/
cargo run --release -p zensim-validate -- --dataset ./datasets/tid2013 --format tid2013

# KADID-10k — expects dmos.csv + images/
cargo run --release -p zensim-validate -- --dataset ./datasets/kadid10k --format kadid10k
```

Look for `Raw dist corr: SROCC=...` in the output — that's the raw distance SROCC reported above. The `SROCC (Spearman)` line uses mapped scores, which are lower for KADID and TID due to score clamping at 0 (35% of KADID scores clamp).

</details>
</details>
<!-- crates.io:skip-end -->

## Quick start

```toml
[dependencies]
zensim = "0.3"
```

```rust
use zensim::{Zensim, ZensimProfile, RgbSlice};

// Use the common codec-target profile (currently B for SDR):
let z = Zensim::new(ZensimProfile::codec_target());
// Reproducible experiments also pin the dependency, bake and extractor
// revision; selecting a profile name alone does not freeze its bytes.
// Use the absolute-luminance API and BHdr route for HDR content.
// `src_pixels` / `dst_pixels` are `&[[u8; 3]]` — interleaved, sRGB-encoded
// (gamma, NOT linear) 8-bit RGB. `width`/`height` are `usize`. See "Input
// format" below: getting sRGB-vs-linear wrong silently corrupts every score.
let source = RgbSlice::new(&src_pixels, width, height);
let distorted = RgbSlice::new(&dst_pixels, width, height);
// `compute` returns `Result<ZensimResult, zensim::ZensimError>` — the `?`
// propagates a dimension-mismatch / too-small / too-large error.
let result = z.compute(&source, &distorted)?;
println!("score: {:.2}", result.score()); // 100 = identical, higher = better
```

Also accepts `RgbaSlice` (composited over a noise background), `imgref::ImgRef` (with stride), `ZenpixelsSource` (with `zenpixels` feature), and `StridedBytes` for BGRA, 16-bit, linear float, and wide gamut (Display P3, BT.2020) inputs. See [docs.rs](https://docs.rs/zensim) for the full `ImageSource` trait.

### Input format (read this — wrong input silently corrupts the score)

The score is only meaningful if the pixels you pass match the contract zensim assumes. There is **no format auto-detection** for the `RgbSlice` fast path: if you hand it linear bytes where it expects sRGB, or planar bytes where it expects interleaved, it computes a perfectly valid-looking but wrong score — no error is raised. The contract:

- **Color encoding: sRGB-encoded (gamma), NOT linear.** `RgbSlice` / `RgbaSlice` / the `Srgb8*` and `Srgb16Rgba` `StridedBytes` formats all expect **display-encoded sRGB** values — the bytes a PNG/JPEG decoder gives you. zensim linearizes internally before the XYB conversion. If your data is already linear light, do **not** feed it as sRGB; use `StridedBytes` with `PixelFormat::LinearF32Rgba` (linear 32-bit float RGBA) instead. (Display P3 reuses the sRGB transfer function, so `Srgb8*` formats linearize it correctly; SDR BT.2020 technically wants BT.1886 — for exact results linearize externally and use `LinearF32Rgba`. Set primaries via `StridedBytes::with_color_primaries`.)
- **Channel order: interleaved, not planar.** `RgbSlice` takes `&[[u8; 3]]` laid out `R,G,B, R,G,B, …` (one `[u8; 3]` per pixel), `RgbaSlice` takes `&[[u8; 4]]` as `R,G,B,A, …`. **Planar** input (all R, then all G, then all B) is **not** accepted by these types — you must interleave it first, or describe it some other way. The `[[u8; 3]]` / `[[u8; 4]]` element type also pins it to exactly 3 / 4 bytes per pixel, **tightly packed** (no per-row padding) — for row padding use `StridedBytes` (below).
- **Dimensions: `width` and `height` are `usize`** (not `u32`). Both `RgbSlice::new(data, width, height)` and the underlying `ImageSource::{width,height}` use `usize`.
- **Both images must have identical dimensions**, and each must be non-zero. `compute` returns `ZensimError::DimensionMismatch` if they differ, `ZensimError::ImageTooSmall` if either dimension is `0`. (Since 0.3.0, sub-64px images down to 1×1 are reflect-padded internally and score normally — only empty inputs are rejected.)

The infallible constructors (`RgbSlice::new`, `RgbaSlice::new`, `StridedBytes::new`) **panic** if `data.len()` is too short for `width × height` (or the stride is invalid). For untrusted sizes use the `try_*` variants — `RgbSlice::try_new(data, width, height) -> Result<RgbSlice, ZensimError>` etc. — which return `ZensimError::InvalidDataLength` / `ZensimError::InvalidStride` / `ZensimError::ImageTooLarge` instead of panicking.

### Return type and errors

```rust
pub fn compute(
    &self,
    source: &impl ImageSource,
    distorted: &impl ImageSource,
) -> Result<ZensimResult, zensim::ZensimError>
```

`ZensimError` is a `#[non_exhaustive]` enum (so match it with a `_` arm) — the variants `compute` can return are `DimensionMismatch`, `ImageTooSmall`, and `ImageTooLarge` (dimensions exceed the configured `max_pixels` cap — **120 MP by default** since #49; tighten it with `Zensim::with_max_pixels`, or pass `with_max_pixels(usize::MAX)` to opt out for trusted input — or `width × height` overflows `usize` on 32-bit / wasm32). HDR-flagged sources (`ImageSource::is_hdr` returns `true`) are refused with `HdrInputRequiresPuPath` — score HDR via the PU21 front-end (`Zensim::compute_pu_linear`, fed absolute-luminance linear RGB in cd/m²) instead. On success, `ZensimResult::score()` is the signed similarity score (identity is 100 and severe damage can score below zero); `raw_distance()`, `approx_ssim2()`, `approx_dssim()`, and `approx_butteraugli()` are also available (see "What the score means").

### Strided / padded rows

When rows are not tightly packed (SIMD-aligned padding, a sub-region crop of a larger buffer, decoder output with row guards), use `StridedBytes`, where `stride` is the **byte** distance between the start of consecutive rows:

```rust
use zensim::{StridedBytes, PixelFormat};

// e.g. 8-bit RGB where each row is padded to `row_stride` bytes (≥ width*3):
let src = StridedBytes::new(&bytes, width, height, row_stride, PixelFormat::Srgb8Rgb);
// `try_new(..) -> Result<_, ZensimError>` returns InvalidStride / InvalidDataLength
// instead of panicking. `with_alpha_mode(.., AlphaMode)` / `with_color_primaries(..)`
// set alpha handling and gamut; default alpha mode is `AlphaMode::Unknown`.
let result = z.compute(&src, &dst)?;
```

With the `imgref` feature (on by default), `imgref::ImgRef<'_, rgb::Rgb<u8>>` and `ImgRef<'_, rgb::Rgba<u8>>` implement `ImageSource` directly and honor their pixel stride — pass an `ImgRef` straight to `compute`. (The element type must be `rgb::Rgb<u8>` / `rgb::Rgba<u8>`; the `ImgRef` stride is in pixels.)

### Cancellation

A single `compute` / `compute_with_ref` call is **not** interruptible mid-computation — the `zensim` metric library exposes no `Stop`-token parameter, and a single comparison is typically tens of milliseconds (~22 ms at 1080p). Cancellation is at the granularity of *your* loop: when comparing one reference against many distorted variants (see "Batch comparison"), check your own cancellation flag between `compute_with_ref` calls. The `enough` cooperative-cancellation crate is used by the `zensim-target` codec-targeting CLI (to bound its binary-search loop) and `zensim-regress`, not by the core metric API.

### zenpixels integration

With the `zenpixels` feature, pass any `PixelSlice` or `PixelBuffer` directly:

```toml
[dependencies]
zensim = { version = "0.3", features = ["zenpixels"] }
```

```rust
use zensim::{Zensim, ZensimProfile, ZenpixelsSource};

let source = ZenpixelsSource::try_from_slice(&pixel_slice)?;
let distorted = ZenpixelsSource::try_from_slice(&other_slice)?;
let result = Zensim::new(ZensimProfile::codec_target()).compute(&source, &distorted)?;
```

Format mapping is automatic: RGBX/BGRX becomes opaque, premultiplied alpha is un-premultiplied, color primaries are forwarded. HDR (PQ, HLG) and grayscale are rejected with `UnsupportedFormat`.

## Target-score CLI (`zensim-target`)

The [`zensim-target`](zensim-target/README.md) standalone workspace is the
runtime side of the "user-facing quality dial" goal. Given an input
image and a target zensim score, it picks the codec quality knob via
binary search:

```bash
cargo run --release --manifest-path zensim-target/Cargo.toml --bin zensim-target -- \
    input.png --target 70 --codec zenjpeg --profile codec-target --output out.jpg
```

The CLI and library default to `codec-target` (currently B). Use
`--profile tuner-v4` to reproduce the earlier CLI default. JPEG, WebP, AVIF and PNG are enabled by
default; JXL encode/decode requires `--features zenjxl` before `--`.
The historical May 18 demo matrix at
[`benchmarks/zensim_target_demo_2026-05-18.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/zensim_target_demo_2026-05-18.md):
reported 33 / 36 cells within ±1.5 score units, median 5 iterations. Those
are dated demo results; inspect today's returned `converged` and achieved
score rather than assuming the target was reached.

`zensim-target` is **AGPL-3.0-only** because it links the AGPL zen
codec crates; the core `zensim` library stays MIT/Apache.

## What the score means

100 = identical; higher = more similar. Negative values represent severe
degradation and must remain visible. Byte-identical inputs short-circuit to
100; that does not prove a model's near-identity behavior. A monotone output
spline preserves raw ordering and cannot repair a raw codec-quality inversion.
The historical JND-at-60 convention below is not a universal perceptual
guarantee for current profiles; see the [score contract](docs/CODEC_TARGET_METRIC.md).

Each `ZensimResult` also provides approximate translations to other metrics:

| Method | What it returns |
|--------|-----------------|
| `score()` | Zensim similarity; higher is better, negative values are valid |
| `raw_distance()` | Feature distance before mapping (lower = better) |
| `approx_ssim2()` | SSIMULACRA2 estimate (MAE 4.4 pts, Pearson r = 0.974) |
| `approx_dssim()` | DSSIM estimate (MAE 0.00129, Pearson r = 0.952) |
| `approx_butteraugli()` | Butteraugli estimate (MAE 1.65, Pearson r = 0.713) |

The `mapping` module has bidirectional interpolation tables — including JPEG quality. These are median values from 344k synthetic pairs across 6 codecs (source: `zensim/src/mapping.rs`):

| Zensim | ≈ SSIM2 | ≈ DSSIM | ≈ JPEG quality |
|-------:|--------:|--------:|:---------------|
| 98 | 96.50 | 0.000017 | ~q95 |
| 90 | 89.41 | 0.000278 | ~q60 |
| 80 | 80.51 | 0.001119 | ~q30 |
| 70 | 71.40 | 0.002356 | — |

JPEG quality mapping accuracy is ±7 quality units MAE — individual images vary widely.

## Regression testing

[**zensim-regress**](https://github.com/imazen/zensimx/blob/main/zensim-regress/README.md) tracks pixel output across platforms and dependency updates. Hash-based checksums for fast exact matches; perceptual comparison with forensic evidence when hashes diverge. Amplified diff images, error classification, architecture-specific tolerances, CI manifests, and HTML reports.

```rust
use zensim_regress::checksums::{ChecksumManager, CheckResult};

let mgr = ChecksumManager::new("tests/checksums".as_ref());
let result = mgr.check_pixels("resize", "bicubic", "200x200",
    &pixels, width, height, None).unwrap();
assert!(result.passed(), "{result}");
```

Run with `UPDATE_CHECKSUMS=1` to create baselines. See the [zensim-regress guide](https://github.com/imazen/zensimx/blob/main/zensim-regress/README.md) for the full workflow.

## Batch comparison

Compare one reference against many distorted variants. Precomputing the reference skips redundant XYB conversion and pyramid construction — saves ~25% per comparison at 4K.

```rust
let precomputed = z.precompute_reference(&source)?;
for dst_pixels in &distorted_images {
    let dst = RgbSlice::new(dst_pixels, width, height);
    let result = z.compute_with_ref(&precomputed, &dst)?;
}
```

## How it works

The extractor computes the features declared by the selected bake, scored by
trained weights. The original 228-feature vocabulary included:

- **SSIM** (mean, L2, L4 pooling) — structural similarity in XYB, using ssimulacra2's modified formula (no luminance denominator)
- **Edge artifacts** (mean, L2, L4) — ringing, banding, blockiness
- **Detail loss** (mean, L2, L4) — blur, smoothing, texture destruction
- **MSE** in XYB color space
- **High-frequency features** — energy loss, magnitude loss, energy gain
- **Peak features** — per-feature max and L8-pooled (near-worst-case)

Computed in XYB (cube-root LMS) with O(1)-per-pixel box blur and fused AVX2/AVX-512 SIMD kernels via [archmage](https://crates.io/crates/archmage). Safe scalar fallback on all platforms.

## Profiles

Each `ZensimProfile` bundles weights and score-mapping parameters. A profile
name is not a frozen model: pin the bake and extraction revision when
reproducing scores. Current checkout mapping, checked October 19, 2026:

| Profile | Role and declared features | Bake size |
|---------|----------------------------|----------:|
| `B` (**default** — `codec_target()`) | SDR diff+frag-820 MLP, Rev5; 420 feature IDs / 820 derived inputs | 214,904 B |
| `BHdr` | HDR linear ensemble on absolute-luminance features; 133 IDs | 5,331 B |
| `D` | Fast SDR linear profile + id100/negative-tail spline; 28 IDs | 1,420 B |
| `C` / `CHdr` | SDR/HDR candidates; 667 / 697 declared IDs, canonical activity semantics | 151,785 / 182,826 B |
| `A` (**deprecated**) | Prior MLP profile; 285 IDs | 26,456 B |
| `PreviewV0_1` / `PreviewV0_2` | Original 228-weight linear profiles, retained for compatibility | In-source arrays |

Exact filenames, hashes, serving limitations and evaluation records are in the
[integration guide](https://github.com/imazen/zensimx/blob/main/docs/CODEC_TARGET_METRIC.md).

`ZensimProfile::codec_target()` and `latest_preview()` both return `B` — the canonical production codec-target the zen codecs dial against (the deprecated `latest()` also returns `B`). `A` (the prior default, the v47 MLP) is now `#[deprecated]` and lives behind the default-on `deprecated-profiles` feature — build with `--no-default-features` to drop it. For dynamically loaded candidates, `zensim::BakeScorer` borrows a parsed model and serves the complete head/spline/ensemble/corruption composition; evaluation uses that same surface. Static custom profiles remain available through `ZensimProfile::Custom` and `ProfileParams::builder()`. Results are deterministic for the same input on the same architecture; cross-architecture scores (AVX2 vs scalar vs AVX-512) may differ by small ULP.

**`D` dial-era v2 (2026-09-05).** `D`'s bake changed from `d_sdr_add156_dense_dial_2026-08-31.bin` to `d_sdr_add156_id100_negrich_dial_2026-09-05.bin`. **The forward pass is byte-identical** (both strip to the same weight bytes), so rank and speed do not move — pooled SROCC is bit-identical on 11 of 14 canonical corpora and within a monotone remap's tie residue on the other three (`kadid` −1.3e-7, `live` +5.8e-7, `tid` +7.9e-6); CID22 is 0.863380 before and after. What changes is the **dial**: a perfect copy now reads **100.000** instead of 96.116, the dial-grid top moves 96.05 → 99.38 (p95 95.28 → 95.52) and the bottom −12.20 → −57.17 (p5 9.52 → 8.83), reach 108.25 → 156.55, and the deepest negative-tail probe row −100.0 → −213.1 — negative scores work further out, which is the product contract, not a regression. No grid cell out-scores identity, before or after (0 of 4,424). **Stored `zensim-d` dial values predate this and must be re-read, not rescaled** — the remap is a PCHIP spline, not an affine. Details + gates: [`benchmarks/d_ship_flip_2026-09-05.md`](benchmarks/d_ship_flip_2026-09-05.md).

`ZensimProfile::PreviewV0_1` / `PreviewV0_2` (the linear profiles that shipped in 0.2.7) are RETAINED as first-class, non-deprecated, selectable variants — the 0.3.0 line reverted their removal (commit `493c91cd`) to preserve semver compatibility with 0.2.7; see the CHANGELOG `[Unreleased]` **Restored** entry. `B` is the current diff+frag-820 default for SDR content.

The historical `PreviewV0_4` / `PreviewV0_5*` SOTA-trail variants, `A_Phone`, and `LinearBounded` live in the **`zensim-experimental`** crate (not published), each rebuilt through the `Custom` extension point — e.g. `Zensim::new(zensim_experimental::preview_v0_5_tuner_v4())`. zenpredict (the MLP runtime) is MIT/Apache-2.0 — no AGPL transitive obligation on default builds.

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `avx512` | yes | AVX-512 SIMD paths |
| `threads` | yes | Multi-threaded computation via rayon (disable for wasm / single-threaded) |
| `imgref` | yes | `ImageSource` impls for `imgref::ImgRef<Rgb<u8>>` and `ImgRef<Rgba<u8>>` |
| `candidate-profiles` | yes | C, CHdr and D profile variants |
| `deprecated-profiles` | yes | Deprecated A profile |
| `feature-regime-v2` | yes | Extended feature extraction; distinct from the opt-in feature arithmetic revision 2 |
| `training` | no | Expose metric internals for weight training |
| `classification` | no | Error classification API (`classify()`, `DeltaStats`, `ErrorCategory`) |
| `zenpixels` | no | `ImageSource` adapter for zenpixels `PixelSlice`/`PixelBuffer` |
| `custom-profiles` | no | `ZensimProfile::Custom` + `ProfileParams::builder()` for externally-defined bakes |
| `streaming_strips_oom` | no | Un-ignores the ~500 MB 80 MP streaming OOM-relief integration test |

<!-- crates.io:skip-start -->
## Downloading evaluation datasets

To reproduce the SROCC numbers above, you need the three human-rated datasets. All are freely available for research use.

**TID2013** — [ponomarenko.info/tid2013.htm](https://www.ponomarenko.info/tid2013.htm)

25 reference images, 3,000 distorted (24 distortion types × 5 levels). Download the RAR archive, extract so you have `mos_with_names.txt`, `reference_images/`, and `distorted_images/` in the same directory.

*N. Ponomarenko et al., "Image database TID2013: Peculiarities, results and perspectives," Signal Processing: Image Communication, 2015. [DOI: 10.1016/j.image.2014.10.009](https://doi.org/10.1016/j.image.2014.10.009)*

**KADID-10k** — [database.mmsp-kn.de/kadid-10k-database.html](https://database.mmsp-kn.de/kadid-10k-database.html)

81 reference images, 10,125 distorted (25 distortion types × 5 levels). Download from [OSF](https://osf.io/xkqjh/). Expected structure: `dmos.csv` and `images/` directory in the same parent.

*H. Lin, V. Hosu, D. Saupe, "KADID-10k: A Large-scale Artificially Distorted IQA Database," QoMEX 2019. [DOI: 10.1109/QoMEX.2019.8743252](https://doi.org/10.1109/QoMEX.2019.8743252)*

**CID22** — [cloudinary.com/labs/cid22](https://cloudinary.com/labs/cid22)

49 validation reference images, 4,292 distorted (6 codecs, medium-to-lossless quality). Download the validation set. Expected structure: `CID22_validation_set.csv`, `original/`, and `compressed/` in the same directory. CC BY-SA 4.0.

*Jon Sneyers et al., "CID22: A Large-Scale Subjective Quality Assessment for Lossy Image Compression," 2024.*
<!-- crates.io:skip-end -->

## Workspace

| Crate | Description |
|-------|-------------|
| [`zensim`](https://crates.io/crates/zensim) | Metric library |
| [`zensim-regress`](https://crates.io/crates/zensim-regress) | Visual regression testing ([guide](https://github.com/imazen/zensimx/blob/main/zensim-regress/README.md)) |
| `zensim-experimental` | Historical / research profiles via the `Custom` extension point (unpublished) |
| `zensim-validate` | Evaluation and training CLI (internal) |
| `zensim-bench` | Comparative benchmarks (standalone root, sibling-dep) |
| `zensim-target` | Target-score codec CLI (standalone root, AGPL, sibling-dep) |

## MSRV

Rust 1.93.0 (2024 edition).

## License

[MIT](https://github.com/imazen/zensimx/blob/main/LICENSE-MIT) OR [Apache-2.0](https://github.com/imazen/zensimx/blob/main/LICENSE-APACHE)

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenjxl-decoder] · [jxl-encoder] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenrav1e] · [rav1d-safe] · [zenravif] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] · [zenyuv] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | **zensim** · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] · [zenanalyze-api] |
| Test corpora | [codec-corpus] · [imazen-26] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter] · [zenutils]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zenextras
[zenpdf]: https://github.com/imazen/zenextras
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenravif]: https://github.com/imazen/cavif-rs
[zenavif-parse]: https://github.com/imazen/zenavif
[zenavif-serialize]: https://github.com/imazen/zenavif
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenpipe
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenyuv]: https://github.com/imazen/zenjpeg
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zenpipe
[zenlayout]: https://github.com/imazen/zenpipe
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenanalyze-api]: https://github.com/imazen/zenanalyze
[codec-corpus]: https://github.com/imazen/codec-corpus
[imazen-26]: https://github.com/imazen/imazen-26
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[zenutils]: https://github.com/imazen/zenutils
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go

### Candidate models for research and serving

Use `zensim::BakeScorer` for a caller-owned `zenpredict::Model`. It reuses
prediction state and returns the complete model score, including configured
heads, splines, blends and corruption gates. `compute` scores SDR pixels and
returns the score with its features; `compute_hdr` uses the explicit HDR
encoding; `score_features` scores admitted cached rows. The user still controls
one target score. Model metadata and disposition are model-author settings.

For rectangle steering, `prepare_steering(&source, bin)` binds SDR input;
`prepare_steering_hdr(&source, encoding, bin)` binds native PQ, HLG or absolute
linear HDR input. Both reuse the reference and retain basic/peak signals from
the scoring extraction for subsequent map assembly. The SDR session also serves
bakes that read v2 (f372-f719) and append/append2 (f720-f943) features with the same score and the same map owner
as the cached-reference attribution path; masked/IW (f228-f371) and f944+
reads are refused up front. The HDR route preserves
declared primaries and native precision; it currently refuses fractional
sampling and unsupported feature families. These are input/implementation
contracts, not evidence that an SDR-trained model is calibrated for HDR.
The optional
`with_finite_moment_refinement(true)` improves finite L2/L4/L8 removal estimates
using binned base-image moments; it preserves scalar scores and additive density.
It is disabled by default and adds preparation cost; see the
[measured runtime and qualification](benchmarks/recovery_completion_2026-09-15.md).
See the [accuracy evidence](benchmarks/finite_moments_2026-09-14.md) and
[updated preparation cost](benchmarks/aligned_bins_2026-09-14.md)
before enabling it. Map coverage and small TRAIN checks do not qualify native
encoder improvements.

New candidates must be evaluated through this same Rust surface. See
[the workflow](docs/WAVE_PLAYBOOK.md), [feature identity](docs/FEATURE_SET_IDS.md)
and [evaluation contract](docs/FULL_EVAL.md). The `serve_custom_bake` example
loads models without leaked bytes or a static loader. Formula revision is
validated. Rev5 models select their declared arithmetic without a process
environment pin. Raw research requests and older wide models retain their
process-revision admission guards (`ZENSIM_FORMULA_REV`).

`ZENSIM_FORMULA_REV=5` selects the `localwin` revision for bakes and research
requests reading only basic (f0–155), peaks (f156–227), and v2 (f372–719).
Explicit raw extraction requests for other families are refused, including
masked/IW and SDR/PU extended-feature requests. With `training`,
`compute_all_features` computes all 576 supported slots independently of
the bake's serving subset. Rev5 mean-offset metadata is zero on scalar and
basic-only prepared steering paths. Requests for supported slots can retain
a wider storage layout. Rev5 uses local f32 blur windows,
sixteen fixed accumulator lanes, fused expressions, and stable central moments.
Identical inputs retain computed reference-only features while difference and
similarity slots have exact identities. Prepared SDR and HDR steering support Rev5.
`ZENSIM_NEIGHBOUR_EXACT=1` enables exact coarse-scale v2 refinement through local
window replay and ordered stable-moment merges. Sampled SDR bakes use canonical
resize reductions; the strip and bounded-v2 entries route through the canonical
owner. See the
[Rev5 contract](benchmarks/rev5_spec_2026-10-04.md) and
[implementation gates](benchmarks/rev5_WORKLOG.md) and
[entry/refinement panel](benchmarks/rev5_entries_2026-10-04.md). Rev5 model retraining and
quiet-box performance qualification remain separate follow-up work.
