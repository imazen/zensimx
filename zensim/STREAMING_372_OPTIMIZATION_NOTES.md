# Streaming-372 optimization roadmap

After Phase 1 (Y-strip aggregator) cherry-pick from `feat/acumen-foundation`,
the **byte-exact 1e-6 gate is locked** (worst rel error observed: 6.83e-14
machine epsilon on the 99-pair safesyn test). What's left is perf + memory
shape, not correctness.

This doc enumerates the optimization followups in order of expected impact.
Each is sized to land as its own commit with the byte-exact gate as the
acceptance criterion.

## Memory-cost model at 80 MP (8000 × 10000)

Measured peak RSS: 5.73 GB.

Per-strip allocation breakdown (40 strips at strip_inner=256, strip_margin=128):

| Source | Bytes per strip | Total at 80 MP |
|---|--:|--:|
| `precomputed_ref_slice_rows` 3 planes × 4 scales (memcpy of full strip rows) | **65.8 MB** | **2632 MB** memcpy + 480 Vec allocations |
| `dst_planes` `[Vec::new(); 3]` (grows in `convert_source_to_xyb_into`) | ~16 MB | 640 MB allocations |
| `ScaleAccumulators` per-scale buffers | ~16 KB | 640 KB (negligible) |
| `strip_meta` `Vec<(usize, usize, usize, usize)>` | 1280 bytes | (one-shot) |

`precomputed_ref_slice_rows` is the dominant cost: ~2.6 GB of per-pair memcpy
plus 480 mallocs. At 30 GB/s system bandwidth that's ~87 ms of pure memcpy
on the wall.

## Optimization 1: Zero-copy `PrecomputedReferenceView<'a>` (HIGHEST IMPACT)

**Status**: DONE — commit `15d3d70` ✓
**Estimated win**: ~10–15% throughput at 80 MP, ~1 GB peak RSS reduction
(eliminates the 65 MB strip copies × 16 rayon workers concurrent footprint),
better L3 locality (parent pyramid stays in cache across strips).

### Why a view, not scratch reuse

Thread-local scratch reuse would amortize the mallocs but keep the memcpy.
The memcpy is the dominant cost. A borrowed view eliminates both.

### Architecture

Add a tiny accessor trait `MultiScaleRef` that both owned + view types
implement:

```rust
pub(crate) trait MultiScaleRef {
    fn num_scales(&self) -> usize;
    fn ref_width(&self) -> usize;
    fn ref_height(&self) -> usize;
    /// Returns `(planes, padded_width, height)` for the given scale.
    fn scale(&self, idx: usize) -> ([&[f32]; 3], usize, usize);
}
```

Two impls:

1. `impl MultiScaleRef for PrecomputedReference` — `scale(idx)` returns
   `([&planes[0], &planes[1], &planes[2]], w, h)` from the owned Vecs.

2. `pub(crate) struct PrecomputedReferenceView<'a>` — holds
   `scales: Vec<([&'a [f32]; 3], usize, usize)>`. `impl MultiScaleRef for
   PrecomputedReferenceView<'a>` returns the slices directly.

Replace `precomputed_ref_slice_rows` with a `slice_rows_view` method on
`PrecomputedReference` that returns a `PrecomputedReferenceView<'_>`
borrowing into the owned ref's planes — zero copies, zero allocs in the hot
loop. (The view itself allocates one small Vec for `scales` metadata, but
that's ~150 bytes × num_scales=4 = 600 bytes, vs 65 MB per strip today.)

### Cascade: consumer signature changes

The strip kernel `compute_multiscale_accums_streaming_with_ref_borrowed`
becomes generic on `impl MultiScaleRef`. Three internal consumers must
accept `[&[f32]; 3]` instead of `&[Vec<f32>; 3]`:

- `compute_xyb_mean_offset` (5 call sites in streaming.rs)
- `compute_xyb_mean_offset_range` (2 call sites)
- `process_scale_bands_into_accum` (7 call sites)

All callers convert via `[&plane[0][..], &plane[1][..], &plane[2][..]]` —
mechanical change, ~30 lines across the file. No cross-module call sites
(audited; all 14 are streaming.rs-local).

### Acceptance

- `streaming::tests::strip_aggregator_byte_exact_safesyn_99` passes at
  `1e-6 rel / 1e-9 abs` (load-bearing).
- `streaming::tests::strip_aggregator_byte_exact_single_pair` passes.
- All 4 integration tests in `tests/streaming_strips.rs` pass.
- New micro-bench (added in this commit): `precomputed_ref_view_perf`
  measures view-build vs copy-build for a 1080p ref + 10 strips. Expected:
  ~1000× speedup on the view-build step alone.

## Optimization 2: Per-rayon-worker scratch for `dst_planes`

**Status**: DONE — commit `a8d9c84` ✓
**Estimated win**: ~640 MB allocation churn → ~48 MB resident (16 workers
× 3 plane Vec). ~5-10% throughput on small-strip sizes where alloc
overhead dominates.

Currently each strip closure does `let mut dst_planes: [Vec<f32>; 3] =
std::array::from_fn(|_| Vec::new())` and the planes grow inside
`convert_source_to_xyb_into` to `padded_width × strip_height × 4 bytes`.

After this commit, dst_planes comes from a per-worker scratch via rayon's
`with_init`/`try_for_each_with` pattern (or a `thread_local!` keyed on the
strip kernel's invocation). First strip on each worker allocates;
subsequent strips reuse via `.truncate()` + `.resize()`.

### Acceptance

- Same byte-exact tests pass.
- Allocator-count macro-bench (new): show ~480 → ~48 allocations per 80 MP
  pair.

## Optimization 3: Phase 4 — streaming `PrecomputedReference`

**Status**: NOT shipping — analysis shows it's the wrong fix.

**Estimated win on inspection**: ~10% peak RSS at 80 MP (1.28 GB pyramid
→ ~800 MB strip-pyramid × 16 workers). For 1080p the ref pyramid is
~16 MB; per-strip pyramid × 16 workers is ~256 MB — actually WORSE.

The two strip paths already cover the use cases cleanly:
- `compute_with_ref_streaming_strips` — batched encoder loop (1 ref × N
  distorted): builds the full pyramid once, scores N distorted against
  it via zero-copy strip views (Optimization 1 covered this).
- `compute_streaming_strips` — single-pair memory-bounded: builds a
  per-strip pyramid from the source SubsetView, never materializes the
  full pyramid. THIS is what the 80 MP OOM test uses.

Phase 4 would converge these — make the with-ref path also build
strip-by-strip. But that means the BATCH case rebuilds the pyramid per
distorted candidate, losing the amortization across calls. Net: slower
for batches, only saves peak RSS for unusual single-call workloads
that would call the with-ref path with N=1.

If you have a workload that truly needs both batched reuse AND
bounded peak (very large image, very many distorted), the fix is a
caller-side discipline: process strips of the corpus in batches, with
each batch building its own per-strip pyramid. That's a caller-side
pattern, not a metric-side architectural change.

Currently `PrecomputedReference::new` builds the full ref XYB pyramid up
front (1.28 GB at 80 MP). With strip aggregation, we only need each
strip's rows from each scale — so build the ref pyramid strip-by-strip in
lockstep with dist.

This is the substantial Phase 4 API change in the original plan doc. It
requires:
- A `RefStripBuilder` (or similar) that maintains pyramid state across
  strips.
- Strip-local XYB conversion of ref rows in lockstep with dist.
- Validate same `1e-6` byte-exact gate.

Lower priority than Optimizations 1 + 2 because the alloc churn matters
more for throughput than peak RSS for the foreseeable use case (per-pair
batched scoring, not single 80 MP one-offs).

## Optimization 4: SIMD-tier audit of inner kernels

**Status**: DONE — commits `0fb528e` (AVX-512 v4x siblings) + `148c65c`
(per-rayon-worker ScaleBuffers reuse) ✓

**Measured wins**: 1.5-2.7× wall-time speedup across all geometries.
On safesyn 99-pair, 16-core Zen 4 (AVX-512 host):

| Geometry | Pre-Opt 4 | Post-Opt 4 | Speedup |
|---|---:|---:|---:|
| Small  256×1024 1T full      | 3.34s | 1.47s | **2.28×** |
| Small  256×1024 par strip    | 1.10s | 0.40s | **2.75×** |
| Medium 1024×2048 1T full     | 2.03s | 0.98s | **2.08×** |
| Medium 1024×2048 par buff    | 0.48s | 0.18s | **2.67×** |

What landed:

1. **AVX-512 v4x siblings for 10 hot inner kernels**. Hand-rolled
   `_v4` kernels already used `f32x16` ops (16-wide), which polyfill
   as 2× AVX2 256-bit on V4 hardware. Adding sibling `_v4x` functions
   with `X64V4xToken` makes the same source compile to single AVX-512
   512-bit ops — halving instruction count + reducing register
   pressure. Touched files: `blur.rs`, `color.rs`, `fused.rs`. The
   `incant!` dispatcher tier lists now include `v4x` before `v4`.

2. **Per-rayon-worker `ScaleBuffers` reuse**. `process_band` closure
   inside `process_scale_bands_into_accum` was creating fresh
   `ScaleBuffers` (7 × `Vec<f32>` zero-fill) per call. Post-v4x the
   memset cost rose to 9.24% (was 5.06% — fixed compute got 2×
   faster, making memset a larger share). The fix: map_init pattern
   gives each rayon worker its own `ScaleBuffers::empty()`; the first
   band call grows via new `ensure_capacity` method; subsequent calls
   are no-ops. Memset post-fix dropped to 3.74%.

3. **Profile-driven**: identified hot loops via `cargo flamegraph` on
   a 1080p strip-path bench harness (`zensim/examples/streaming_bench.rs`).
   Top 6 symbols now collectively ~96% of wall-time, all `_v4x` SIMD
   kernels. Remaining non-SIMD time is rayon coordination (~4.4%) and
   memset for initial allocations (~3.7%) — both at the threshold
   where further optimization needs harder structural changes.

`cargo read magetypes` confirmed the kernel idiom: `#[magetypes(define(...),
v4, v3, neon, wasm128, scalar)]` for one-source-many-platforms SIMD. The
existing streaming infra already uses this through `archmage` (see
`fused::fused_vblur_features_ssim`, `simd_ops::build_iw_weight_and_mse`,
etc.).

Audit any remaining `#[inline]`-without-target_feature loops in the strip
critical path:

- `compute_xyb_mean_offset_range` — scalar fallback if no `#[arcane]` /
  `#[magetypes]`.
- `ScaleAccumulators::merge` — pure scalar today; probably fine since it
  runs once per strip per scale (not in the hot inner loop).
- Per-strip plane-copy in `precomputed_ref_slice_rows` — moot if
  Optimization 1 ships.

Profile first with `perf stat` or `flamegraph` to confirm which kernels
are actually hot. Don't optimize on speculation.

## Cache-locality observations

- `STRIP_INNER = 32` (inline-band processing) sits at Zen 4's 1 MB L2
  boundary. Already tuned. Don't bump this.
- `strip_inner = 256` (outer Y-strip aggregator) is L3-bound at full image
  width. This is correct — outer strips are for memory-budget control,
  not L2 fitting.
- The view-refactor (Optimization 1) helps L3 reuse: instead of copying
  parent planes into a freshly-allocated strip Vec (cold pages), the strip
  kernel reads parent planes directly (warm pages from prior strips on the
  same worker).

## Test infrastructure notes

The `1e-6 rel / 1e-9 abs` gate on 99 safesyn pairs catches all 372
features. Worst observed during Phase 1: **6.83e-14 rel** (machine
epsilon). Any optimization that drifts this gate is wrong.

Two test modes:
- Default: `cargo test --release --features threads streaming::tests` —
  uses tiny synthetic pairs (256×1024) for fast CI.
- 80 MP OOM: `cargo test --release --features threads,streaming_strips_oom
  streaming_strips_oom_80mp` — gates on peak RSS ≤ 8 GB.

## Throughput baseline (Phase 1, on this branch)

Measured on safesyn 99-pair set, 16-core Zen 4:

### Before Optimizations 1 + 2 (cherry-pick state, commit `ead257b`)

| Geometry | Parallel (strip vs full) | Single-threaded |
|---|---:|---:|
| 256×1024 | 1.75× | 0.87× |
| 1024×2048 | 0.87× | 0.89× |

### After Optimizations 1 + 2 (view refactor + scratch reuse, commit `a8d9c84`)

| Geometry | Parallel strip/full | Parallel buffered/full | 1T strip/full |
|---|---:|---:|---:|
| 256×1024 | 1.35× | 1.04× | **1.60×** ✓ |
| 1024×2048 | 1.07× | 0.92× | **1.49×** ✓ |

**Single-threaded target ≥ 1.2× MET on both geometries** — the
1T throughput improvement directly proves the view refactor + scratch
reuse eliminated the per-strip alloc + memcpy overhead that was
the Phase 1 1T bottleneck. The plan doc had targeted Optimization 3
(streaming PrecomputedReference) for this win, but it's already done
with the simpler view + scratch combination.

The parallel 256×1024 strip/full ratio moved from 1.75× → 1.35× —
slight regression at very small workloads. Hypothesis: `map_init`'s
per-worker scratch initialization cost is amortized poorly when each
worker only processes 1–2 strips. Worth follow-up to detect-small +
fall back to per-strip Vec::new (probably ~5% gain at this size; not
worth doing until profiling confirms `map_init` is the cause).

## How to execute the next commit

Pick Optimization 1. It's the highest-impact single change.

```sh
cd ~/work/zen/zensimx
git checkout feat/streaming-372-phase1   # this branch
# implement the trait + view, cascade signatures (~300 LOC)
cargo test --release --features threads -p zensim --lib \
    streaming::tests::strip_aggregator_byte_exact   # MUST pass
cargo test --release --features threads -p zensim --test streaming_strips
git commit -m "perf(streaming): zero-copy PrecomputedReferenceView for strips"
git push
```

If the byte-exact test fails at any point during the refactor, stop and
trace which scale + which feature diverged. The aggregator math is exact
under shared `Vec<f32>` vs slice-into-shared-Vec, so any divergence is a
slicing bug — most likely off-by-one in `start_off`/`end_off` at scale
boundaries.
