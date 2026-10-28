# Streaming 372-feature CPU path — plan

**Goal**: support 80MP images on the production CPU 372-feature path without OOM.

## Current state (2026-05-22)

The "streaming" infrastructure already exists at the strip-WITHIN-plane level:
- `streaming::process_scale_bands` does row-strip H-blur → fused V-blur+features
- `fused::fused_vblur_features_ssim` / `_edge` uses `#[arcane]` + magetypes correctly
- `iw_pool::compute_iw_weights` (full-plane allocs) is NOT in the production path —
  the streaming path uses `simd_ops::build_iw_weight_and_mse` (per-strip)

**The OOM at 80MP is not from IW**, it's from:

1. `convert_source_to_xyb` allocates full dst XYB planes (3 × W × H × 4 bytes)
2. `PrecomputedReference` holds the full ref pyramid (1.33 × W × H × 3 × 4 bytes)
3. Each scale's downscale happens IN-PLACE on the full plane

For 80MP (8000 × 10000):
- dst planes: 3 × 80M × 4 = **960 MB**
- PrecomputedReference (4 scales): 1.33 × 80M × 3 × 4 = **1.28 GB**
- Plus scale buffers (`pool::ScaleBuffers`): another ~250 MB
- **Total per-pair peak: ~2.5 GB**
- 16 rayon workers × 2.5 GB = **40 GB**. OOMs at 32 GB RAM, marginal at 64 GB.

## Architectural fix

**Y-strip aggregation**: process the source image in horizontal strips with
overlap rows for the blur stencil. Each strip runs the existing pipeline
(unchanged inner kernels). Aggregate per-scale features across strips.

### Strip geometry

- `strip_inner = 256` scale-0 rows (the "contributing" rows)
- `strip_margin = 128` scale-0 rows (overlap for blur stencil at scale 3)
- Each strip processes `(strip_inner + 2 * strip_margin) = 512` scale-0 rows
- At scale 3 (1/8 dims): 64 inner + 32 margin = 96 rows total per strip
- Blur radius at scale 3: 5 rows → 11-row stencil → margin must be ≥ 5 rows
  at every scale. 128 rows at scale 0 = 16 rows at scale 3 → satisfies.

Strip count for 80MP (h=10000): 40 strips of 256 inner rows + edges.

### Per-strip memory

512 rows × 8000 cols × 4 bytes × 3 channels × 2 sides × 1.33 (pyramid)
≈ **125 MB per pair per strip**.

16 workers × 125 MB = **2 GB peak**. Fits comfortably.

### Aggregation math

`ScaleStats` fields are pooled means (`mean_d`, `root4_d`, etc.), not raw sums.
To aggregate:
- For mean stats: `agg_mean = sum(tile_mean × tile_count) / total_count`
- For root-power stats: `agg_rootp = (sum(tile_rootp^p × tile_count) / total_count)^(1/p)`
- For max stats: `agg_max = max(tile_max)`
- `tile_count` = product of strip dims at that scale, accounting for blur-edge skip

Per scale at scale `s`: `tile_count = strip_inner_rows[s] × (padded_w[s] - 2*blur_radius)`.

### Per-strip codepath

1. Extract `(y_full_start, y_full_end)` strip of source bytes via row-copy.
2. Wrap in `RgbSlice` (or LinearF32Rgba etc. — match input format).
3. Run existing `compute_multiscale_stats_streaming_with_ref_borrowed` on the
   strip's sub-source.
4. Capture `ScaleStats` per scale.
5. Reverse-pool: `tile_mean × tile_count = tile_sum`. Accumulate into
   global per-scale sum buffers.
6. After last strip: re-pool to produce the final `ScaleStats`.

## Implementation phases

### Phase 1: top-level strip iterator + aggregation (this branch)

- New function `Zensim::compute_with_ref_streaming_strips(ref, dist, strip_inner, strip_margin)`
- Internal: iterate strips, copy rows into sub-ImageSource, run existing
  pipeline per strip, aggregate.
- Tests: byte-equivalence to `compute_with_ref` on a 1080p test pair
  (within numeric tolerance < 1e-6 per feature).
- Cost: ~600 LOC + tests.

### Phase 2: avoid the per-strip row-copy

- Wrap `ImageSource` with a `StripView` adapter that reports
  `width = full_width`, `height = strip_height`, `row(y) = parent.row(y + offset)`.
- Eliminates the per-strip ~6 MB copy.

### Phase 3: magetypes audit of inner kernels

- Inventory which inner kernels do or don't use the magetypes idiom.
- Hot paths to confirm: `build_iw_weight_and_mse`, `ssim_channel_extended`,
  `edge_diff_channel_masked`.
- Profile per-kernel cost on a 1080p strip; promote any non-vectorized to
  `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`.

### Phase 4: streaming PrecomputedReference

- Currently `PrecomputedReference` holds the full ref pyramid. With strip
  aggregation, we ONLY need the ref's strip rows for each strip — so we can
  build the ref XYB pyramid strip-by-strip in lockstep with the dist.
- This requires changing `PrecomputedReference` to support "fill for strip
  Y range" rather than "all rows" — substantial API change.

## Acceptance criteria

| Test | Pass condition |
|---|---|
| `streaming_strips_matches_full` | All 372 features within `1e-6` rel error vs full path on 99 safesyn pairs |
| `streaming_strips_oom_80mp` | Process a synthetic 80MP pair with `--max-memory 4GB` set without OOM |
| `streaming_strips_throughput` | At least 1× the throughput of the full path on safesyn 99 pairs |
| `single_threaded_throughput` | At least 1.2× the single-threaded throughput vs full path |

## Tracking

- imazen/zensimx#40 (acumen) — Mode B-lite work intersects here
- imazen/zensim — new issue for this work (TBD)
