//! Attribution-density steering maps — `custom-profiles` API.
//!
//! `BakeScorer::compute_with_ref_and_attribution` binds a candidate's
//! complete score, declared feature extraction and sensitivities to the
//! retained, binned map owner. Its result reports unsupported spatial terms
//! and discontinuous corruption gating. The basic-block construction below
//! underlies both that route and the earlier caller-supplied-gradient APIs;
//! the full route also covers v2, append and append2 integrands. The candidate
//! route additionally covers the 36 L8 terms in f156-227.
//! [`ScoredAttribution::refinement_gain`] adds hard maxima through a separate
//! non-additive rectangle estimate. Its coverage report is distinct from
//! density coverage; see that method's frozen-signal ownership contract.
//!
//! Builds a per-pixel **attribution density** `D(x, y)` for a scalar model's
//! gradient `s_k = ∂score/∂f_k` over the BASIC feature block (f0-155: 13 slots
//! × 3 channels × 4 scales, scale-major `scale*39 + ch*13 + slot`), plus a
//! summed-area table for O(1) arbitrary-rectangle queries.
//!
//! # Why this exists (E-M6..E-M9, `benchmarks/coherent089_seeded_frontier_2026-07-27.md`)
//!
//! The per-pixel *signal fold* (`DiffmapWeighting::ModelSensitivity`) blends
//! scales by gradient mass and renormalizes per scale, so a model whose mass
//! sits on coarse-scale MSE degrades to a 1/8-resolution map — block ranking
//! (M3) collapses to 0.1-0.3 and inverts at 128 px, while the per-block
//! gradient attribution `Σ_k s_k·Δf_k(block)` (M2) stays 0.999-1.000 at every
//! block size. This module productizes the M2 mechanism: each feature's
//! per-pixel contribution is laid down with its **TRUE integrand and ABSOLUTE
//! normalization** (no per-scale renormalization, no mass blend — `s_k`
//! carries the weight), so a rectangle sum over any block is the first-order
//! prediction of the score gain from refining that block.
//!
//! # The contract
//!
//! [`AttributionResult::query_rect`] over block `B` approximates
//! `Σ_k s_k · Δf_k(B)` — the linearized scalar-score gain from re-encoding
//! `B` at reference quality — for ANY axis-aligned rectangle, at O(1) cost.
//! This is the codec-facing steering surface for variable partitions (AV1
//! 4-128 px recursive, JXL variable DCT, HEVC CTU splits): the codec queries
//! its own partition geometry directly instead of consuming a fixed-grid map.
//!
//! # Integrands (verified against `finalize()` / `fused_vblur_features_ssim`)
//!
//! Per (scale, channel), the 13 basic slots pool per-pixel signals as:
//!
//! | slot | feature | pooling | per-pixel signal `v` |
//! |---|---|---|---|
//! | 0/1/2 | ssim mean/4th/2nd | mean / p4 / p2 | `sd` (SSIM error) |
//! | 3/4/5 | art mean/4th/2nd | mean / p4 / p2 | `max(0, ed)`, `ed = (1+|d−μ2|)/(1+|s−μ1|) − 1` |
//! | 6/7/8 | det mean/4th/2nd | mean / p4 / p2 | `max(0, −ed)` |
//! | 9 | mse | mean | `(s − d)²` |
//! | 10 | hf_energy_loss | `max(0, 1 − varDst/varSrc)` | `(s−μ1)²`, `(d−μ2)²` |
//! | 11 | hf_mag_loss | `max(0, 1 − madDst/madSrc)` | `\|s−μ1\|`, `\|d−μ2\|` |
//! | 12 | hf_energy_gain | `max(0, varDst/varSrc − 1)` | same as 10 |
//!
//! - **Mean-pooled** (slots 0,3,6,9): `d_k(i) = v_i / N` — the pixel's exact
//!   share of the feature. Full-plane sum = the feature value exactly.
//! - **p-pooled** (slots 1,2,4,5,7,8; `f = (mean(v^p))^{1/p}`):
//!   `d_k(i) = (1/p) · (v_i^p / N) · M^{(1−p)/p}` with `M = mean(v^p)`.
//!   The `1/p` factor makes the block sum the **removal-consistent first
//!   order**: zeroing block `B`'s members changes `M` by `−ΔM_B` linearly, so
//!   `Δf ≈ −(1/p)·ΔM_B·M^{1/p−1}`. The raw value-weighted gradient
//!   `v_i·∂f/∂v_i = (v_i^p/N)·M^{(1−p)/p}` (Euler form, sums to `f`) is the
//!   `t→0` path derivative of *scaling* the block and over-weights p-pooled
//!   slots by ×p relative to their true block-removal effect — wrong relative
//!   weighting against mean slots, so the `1/p` form is used. Per-slot block
//!   RANKING is identical either way (same `M` for all blocks); only the
//!   cross-slot mix differs.
//!   Candidate L8 slots (offsets 3/4/5 in each six-slot peak cell) use the
//!   same formula with p=8 and the same SSIM/artifact/detail signals. Their
//!   historical `p95` field names do not denote quantiles. Canonical f32
//!   eighth powers are multiplied by f64 coefficients before map rounding.
//! - **hf ratio slots** (10-12): exact first-order integrands of the clamped
//!   ratio, gated on the clamp state (`varSrc > 1e-10` matching `finalize`,
//!   loss XOR gain active). With `e_i = (s_i−μ1_i)² − (d_i−μ2_i)²`:
//!   slot 10 active ⇒ `d_10(i) = e_i / Σe_src²`; slot 12 active ⇒
//!   `d_12(i) = −e_i / Σe_src²`; slot 11 active ⇒
//!   `d_11(i) = (|s_i−μ1_i| − |d_i−μ2_i|) / Σ|s−μ1|`. These are SIGNED —
//!   e.g. a ringing block inside a globally-blurred image genuinely
//!   *increases* `hf_energy_loss` when refined, and the density carries that.
//!   Full-plane sums equal the (unclamped-side) feature values exactly.
//!
//! Sign convention matches `ModelSensitivity`: the map weight is `−s_k`
//! (score-oriented models have negative sensitivity on error signals →
//! positive map weight = "refining here gains score"). The map is **signed**.
//!
//! Coarse scales are upsampled **sum-preservingly**: a scale-`s` pixel's
//! density spreads uniformly over its `2^s × 2^s` full-resolution footprint
//! divided by the footprint area (nearest-neighbor ÷ area, NOT bilinear), so
//! rectangle sums reproduce coarse-plane sums exactly.
//!
//! # Honest approximations / blind spots
//!
//! 1. **Hard maxima, masked/IW pools and f944+ have no integrands here.**
//!    The candidate result reports locally active unsupported IDs, including
//!    extraction variants whose matching retained integrands are unavailable.
//!    Reference-only features and SDR highlight structural zeros contribute
//!    exactly zero. The candidate covers L8 (three of each six f156-227 slots)
//!    and supported f372-943 terms. Legacy caller-gradient APIs still ignore
//!    all f156-371. An additive density cannot exactly represent finite max
//!    removal: removing either of two tied maxima has no effect, but removing
//!    both does. Coverage of L8 does not resolve that max-term limitation.
//! 2. **Blur bleed** (C2b MEASURED): refining block `B` also changes signals
//!    within the blur radius outside `B`. The pure-window-supported signals
//!    (ssim `d`; v2 contrast/texture) ARE spread over their blur window via
//!    the sum-preserving box spread (`blur::box_spread_sum_preserving`,
//!    clipped-window per-source-normalized convention) — measured NEUTRAL on
//!    the 8-cell gate (±0.003). Residual-form signals (art/det/hf/mscn) stay
//!    pixel-allocated: the 50/50 pixel/window split was measured and
//!    REGRESSED all 8 cells (−0.01..−0.08). That result does not reject a
//!    pixel derivative contracted with the actual repair direction. An
//!    unweighted residual gradient sums to zero, but its repair contraction
//!    generally does not. Reflect-101 also requires the transpose `K^T`,
//!    not plain `K`, at boundaries. Synthetic tests establish this distinction;
//!    finite repair fidelity and native utility remain unproven. See
//!    `benchmarks/pixel_adjoint_2026-09-14.md` and the historical C2b results.
//! 2a. **Lattice-step features (v2 BLOCKINESS) are exact for rectangle repairs.** A rectangle repair changes the
//!    steps it cuts by a one-sided amount, not half of the step's excess, so the density gives each step end its own
//!    one-sided share and the result keeps the per-step pair terms (`LatticeScale`); `query_rect` and
//!    `block_sums` add the correction for the steps a rectangle edge cuts. Exact when the edges are aligned to
//!    the pyramid scale's footprint (`2^s` source pixels) and the scale's planes lie inside the image; other
//!    edges use the convention `-(I/2)|a_p - a_q|` on the covered fractions of the two step ends. With a sampling geometry, or on a
//!    sink that does not carry the terms, the symmetric 50/50 split remains. `density()` is the pixel-allocated
//!    view and does not include the edge terms.
//! 3. **SIMD-padding columns** (padded width − width) carry feature mass that
//!    the trimmed map cannot attribute (≤ ~3 % of columns; near-zero signal
//!    since both planes zero-pad identically).
//! 4. **p-root curvature** beyond first order for large removals, and **hf
//!    clamp state** assumed fixed under refinement.
//! 5. The internal V-blur banding replicates the production 32-row band
//!    layout, so per-pixel signals and pooled scalars are bit-compatible with
//!    the production feature extractor (`sum-preservation` tests below).

use crate::ZensimError;
use crate::blur::fused_blur_h_ssim;
use crate::det_math::DetRoots;
use crate::fused::{StripChannelAccum, fused_vblur_features_ssim};
use crate::metric::{
    FEATURES_PER_CHANNEL_BASIC, MIN_PYRAMID_DIM, check_within_max_pixels, config_from_params,
    reflect_pad_to_min, validate_pair, validate_ref_match,
};
use crate::source::ImageSource;
use crate::streaming::{
    MultiScaleRef, PrecomputedReference, convert_source_to_xyb, downscale_3_planes,
};
use archmage::autoversion;

/// Production band height for the V-blur running-sum chains (must equal
/// `streaming::STRIP_INNER` so the per-band accumulator init points — and
/// therefore every per-pixel signal and pooled scalar — are bit-compatible
/// with the production feature extractor).
const BAND_ROWS: usize = 32;

// ── Feature-layout block boundaries (half-open ends) ─────────────────────
//
// These name the `s`-slice bounds that
// [`crate::Zensim::compute_attribution_density_full`] cuts the model
// gradient on. They exist because the append2 block (`f924-943`) was
// silently dropped for ten months by a hard-coded `min(len, 924)` bound —
// found 2026-08-04 by the coherence study, and the reason the block was
// classified "non-decomposable" when 8 of its 20 slots are exactly
// decomposable. A named bound plus the per-width coverage test
// (`attribution_covers_expected_slots_per_width`) is the anti-recurrence
// guard: adding a block means adding a constant AND a row to that test.

#[cfg(feature = "feature-regime-v2")]
mod layout_ends {
    /// End of the basic block (`f0-155`) = start of the v1 pooled block.
    /// Used by the candidate coverage report and the per-width coverage gate.
    pub(crate) const BLOCK_END_BASIC: usize = 156;
    /// End of the v1 peak block (max and L8, f156-227).
    pub(crate) const BLOCK_END_V1_PEAKS: usize = 228;
    /// End of the v1 peak/masked/IW pooled block (`f156-371`) = start of v2.
    /// Max/masked/IW remain unsupported (module blind spot 1).
    pub(crate) const BLOCK_END_V1_POOLS: usize = 372;
    /// End of the v2 block (`f372-719`) = start of the append block.
    pub(crate) const BLOCK_END_V2: usize = 720;
    /// End of the append block (`f720-923`) = start of append2.
    pub(crate) const BLOCK_END_APPEND: usize = 924;
    /// End of the append2 block (`f924-943`). Anything beyond (the f944+ CSFW
    /// block) has no attribution integrand yet and is deliberately not sliced —
    /// the coverage test pins that too.
    pub(crate) const BLOCK_END_APPEND2: usize = 944;
}
#[cfg(feature = "feature-regime-v2")]
use layout_ends::*;

/// Per-pixel (or binned) attribution density + summed-area table.
///
/// Produced by [`crate::Zensim::compute_attribution_density`]. See the
/// [module docs](self) for the mechanism and the steering contract.
///
/// The stored density is the `f32` view (visualization / export); the SAT is
/// built from the internal `f64` accumulation *before* `f32` rounding, so
/// [`query_rect`](Self::query_rect) sums are `f64`-accurate. A result built
/// via [`from_density`](Self::from_density) instead derives the SAT from the
/// given `f32` values.
///
/// # Subsampled (binned) results
///
/// The `*_binned` compute entry points fold the per-pixel accumulation into a
/// `bin × bin` grid before the SAT and `f32` view are built, shrinking
/// *retained* memory and SAT-build cost by `bin²` (the transient full-
/// resolution `f64` accumulation canvas still exists during the compute — the
/// per-pixel integrands are exact either way). Because every steering
/// consumer reads *integrals* (rect sums), binning does not approximate the
/// queries codec partitions actually make: [`query_rect`](Self::query_rect)
/// stays **exact** for bin-aligned rectangles (and at image edges), and
/// [`block_sums`](Self::block_sums) stays exact whenever `bin` divides the
/// block size. Unaligned interior rectangles are answered by area-weighted
/// interpolation (mass within a bin treated as uniform) — a first-order
/// approximation whose error is bounded by the mass in the partially-covered
/// boundary bins. Pick `bin` as a power of two that divides the codec's
/// partition alignment (4/8/16 for AV1, 8 for JXL var-DCT) and every real
/// query stays exact.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct AttributionResult {
    /// Grid-resolution signed density, row-major `grid_w × grid_h`, in
    /// per-pixel units (each value is its bin's mean; for `bin == 1` this is
    /// the per-pixel density unchanged).
    density: Vec<f32>,
    /// Summed-area table over bin SUMS, `(grid_w+1) × (grid_h+1)`,
    /// `sat[v*(grid_w+1)+u]` = Σ density-mass over grid `[0,u) × [0,v)`.
    sat: Vec<f64>,
    width: usize,
    height: usize,
    /// Grid step in pixels (`1` = per-pixel, the default construction).
    bin: usize,
    grid_w: usize,
    grid_h: usize,
    /// Exact boundary terms of lattice-step features (BLOCKINESS): the pair interaction of steps cut by a
    /// rectangle edge. Empty for every result that has none; see [`LatticeScale`].
    lattice: Vec<LatticeScale>,
}

impl AttributionResult {
    /// Build from an externally assembled density plane (row-major
    /// `width × height`), e.g. the basic-block density plus a v2 fold map.
    /// The SAT is derived from the given `f32` values.
    ///
    /// # Panics
    ///
    /// Panics if `density.len() != width * height`.
    pub fn from_density(density: Vec<f32>, width: usize, height: usize) -> Self {
        Self::from_density_with_sat_scratch(density, width, height, Vec::new())
    }

    /// Internal `from_density` variant reusing a caller-provided SAT
    /// buffer's capacity (the stale single-pass session's recycle path —
    /// avoids a fresh multi-MB allocation + page-fault storm per codec
    /// iteration). An empty `sat_scratch` reproduces
    /// [`from_density`](Self::from_density) exactly.
    fn from_density_with_sat_scratch(
        density: Vec<f32>,
        width: usize,
        height: usize,
        mut sat_scratch: Vec<f64>,
    ) -> Self {
        assert_eq!(density.len(), width * height, "density len != width*height");
        build_sat_into(|i| density[i] as f64, width, height, &mut sat_scratch);
        Self {
            density,
            sat: sat_scratch,
            width,
            height,
            bin: 1,
            grid_w: width,
            grid_h: height,
            lattice: Vec::new(),
        }
    }

    /// Internal: build from the f64 accumulation canvas (SAT keeps f64 truth).
    pub(crate) fn from_f64_canvas(canvas: Vec<f64>, width: usize, height: usize) -> Self {
        Self::from_f64_values(&canvas, width, height)
    }

    fn from_f64_values(canvas: &[f64], width: usize, height: usize) -> Self {
        Self::from_f64_values_parallel(canvas, width, height, false)
    }

    fn from_f64_values_parallel(
        canvas: &[f64],
        width: usize,
        height: usize,
        parallel: bool,
    ) -> Self {
        debug_assert_eq!(canvas.len(), width * height);
        let sat = build_sat_values(canvas, width, height, parallel);
        let density = canvas.iter().map(|&v| v as f32).collect();
        Self {
            density,
            sat,
            width,
            height,
            bin: 1,
            grid_w: width,
            grid_h: height,
            lattice: Vec::new(),
        }
    }

    /// Internal: fold the f64 accumulation canvas into a `bin × bin` grid,
    /// then build the SAT + `f32` view at grid resolution. `bin == 1`
    /// delegates to [`from_f64_canvas`](Self::from_f64_canvas) so the
    /// per-pixel construction stays bit-identical to the unbinned path.
    ///
    /// The SAT holds bin SUMS (integrals are what queries consume); the
    /// `f32` density holds bin MEANS over each bin's *real* pixel count
    /// (edge bins are clipped), keeping the exported view in per-pixel
    /// units so visualization scales are comparable across `bin` choices.
    /// Level-1 reference implementation (fold an existing full-resolution
    /// canvas): production paths now accumulate bin-side (Level 2), so this
    /// survives as the independent reference the `binned_l2_matches_l1_fold`
    /// equivalence gates compare against.
    #[cfg_attr(not(test), allow(dead_code))]
    fn from_f64_canvas_binned(canvas: Vec<f64>, width: usize, height: usize, bin: usize) -> Self {
        assert!(bin > 0, "bin must be non-zero");
        if bin == 1 {
            return Self::from_f64_canvas(canvas, width, height);
        }
        debug_assert_eq!(canvas.len(), width * height);
        let grid_w = width.div_ceil(bin);
        let grid_h = height.div_ceil(bin);
        let mut sums = vec![0.0f64; grid_w * grid_h];
        for y in 0..height {
            let row = &canvas[y * width..(y + 1) * width];
            let grid_row = &mut sums[(y / bin) * grid_w..(y / bin + 1) * grid_w];
            for (cell, chunk) in grid_row.iter_mut().zip(row.chunks(bin)) {
                *cell += chunk.iter().sum::<f64>();
            }
        }
        Self::from_bin_sums(sums, width, height, bin)
    }

    /// Internal: build directly from bin SUMS (the Level-2 sink product) —
    /// SAT over sums, `f32` view = bin means over real clipped pixel counts.
    fn from_bin_sums(sums: Vec<f64>, width: usize, height: usize, bin: usize) -> Self {
        Self::from_bin_values(&sums, width, height, bin)
    }

    fn from_bin_values(sums: &[f64], width: usize, height: usize, bin: usize) -> Self {
        Self::from_bin_values_parallel(sums, width, height, bin, false)
    }

    fn from_bin_values_parallel(
        sums: &[f64],
        width: usize,
        height: usize,
        bin: usize,
        parallel: bool,
    ) -> Self {
        debug_assert!(bin > 0);
        if bin == 1 {
            // Every bin has one real pixel, including image boundaries.
            // Dividing by its area is exactly division by one.
            return Self::from_f64_values_parallel(sums, width, height, parallel);
        }
        let grid_w = width.div_ceil(bin);
        let grid_h = height.div_ceil(bin);
        debug_assert_eq!(sums.len(), grid_w * grid_h);
        let sat = build_sat_values(sums, grid_w, grid_h, parallel);
        let density = sums
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let real_w = bin.min(width - (i % grid_w) * bin);
                let real_h = bin.min(height - (i / grid_w) * bin);
                (v / (real_w * real_h) as f64) as f32
            })
            .collect();
        Self {
            density,
            sat,
            width,
            height,
            bin,
            grid_w,
            grid_h,
            lattice: Vec::new(),
        }
    }

    /// The signed density in per-pixel units, row-major
    /// [`grid_width()`](Self::grid_width) `×` [`grid_height()`](Self::grid_height)
    /// (`== width × height` for the default `bin == 1`; one value per bin —
    /// the bin's mean — for binned results). Positive = refining here is
    /// predicted to raise the scalar score.
    pub fn density(&self) -> &[f32] {
        &self.density
    }

    /// Image width in pixels (actual, not SIMD-padded).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Grid step in pixels: `1` for per-pixel results, the requested bin for
    /// results from the `*_binned` entry points.
    pub fn bin(&self) -> usize {
        self.bin
    }

    /// Width of the stored grid (`ceil(width / bin)`).
    pub fn grid_width(&self) -> usize {
        self.grid_w
    }

    /// Height of the stored grid (`ceil(height / bin)`).
    pub fn grid_height(&self) -> usize {
        self.grid_h
    }

    /// Sum of the density over the half-open PIXEL rectangle
    /// `[x0, x1) × [y0, y1)`, in O(1) via the summed-area table. Coordinates
    /// are clamped to the image; empty or inverted rectangles return `0.0`.
    ///
    /// For a codec partition block `B`, this is the first-order prediction of
    /// the scalar-score gain from re-encoding `B` at reference quality (see
    /// the [module docs](self) for the exact semantics and approximations).
    ///
    /// On a binned result the answer is exact when the rectangle's edges are
    /// bin-aligned or clamped at the image boundary; unaligned interior edges
    /// are answered by area-weighted interpolation (uniform-mass-within-bin —
    /// see the [struct docs](Self)).
    pub fn query_rect(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let x1 = x1.min(self.width);
        let y1 = y1.min(self.height);
        let x0 = x0.min(x1);
        let y0 = y0.min(y1);
        let edge_terms = self.lattice_correction(x0, y0, x1, y1);
        if self.bin == 1 {
            let w1 = self.grid_w + 1;
            return self.sat[y1 * w1 + x1] - self.sat[y0 * w1 + x1] - self.sat[y1 * w1 + x0]
                + self.sat[y0 * w1 + x0]
                + edge_terms;
        }
        let u0 = self.grid_coord(x0, self.width, self.grid_w);
        let u1 = self.grid_coord(x1, self.width, self.grid_w);
        let v0 = self.grid_coord(y0, self.height, self.grid_h);
        let v1 = self.grid_coord(y1, self.height, self.grid_h);
        self.sat_at(u1, v1) - self.sat_at(u0, v1) - self.sat_at(u1, v0)
            + self.sat_at(u0, v0)
            + edge_terms
    }

    /// Internal: attach finalized lattice-step boundary terms.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // BLOCKINESS lattice terms are produced by the v2 walk
    pub(crate) fn with_lattice(mut self, lattice: Vec<LatticeScale>) -> Self {
        self.lattice = lattice;
        self
    }

    /// Sum of the exact lattice-step boundary terms for the (already clamped) rectangle; `0.0` when the result
    /// carries none, so every pre-existing result is unchanged bit for bit (`x + 0.0 == x`).
    fn lattice_correction(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        self.lattice
            .iter()
            .map(|l| l.correction(x0, y0, x1, y1))
            .sum::<f64>()
    }

    /// Pixel coordinate → (grid node index, fractional advance into the next
    /// cell). The fraction is measured against the cell's REAL pixel extent
    /// (edge cells are clipped), so `x == limit` lands exactly on the last
    /// node and full-image / edge-clamped queries stay exact.
    fn grid_coord(&self, x: usize, limit: usize, grid: usize) -> (usize, f64) {
        if x >= limit {
            return (grid, 0.0);
        }
        let idx = x / self.bin;
        let start = idx * self.bin;
        let real = self.bin.min(limit - start);
        (idx, (x - start) as f64 / real as f64)
    }

    /// Continuous SAT lookup at fractional grid coordinates via bilinear
    /// interpolation of the four surrounding nodes. For a piecewise-constant
    /// (per-cell-uniform) density this is *exactly* the integral over
    /// `[0, u) × [0, v)`; a zero fraction reproduces the node value exactly
    /// (`a + 0.0 * (b - a) == a`, signed zeros included).
    fn sat_at(&self, (ix, fx): (usize, f64), (iy, fy): (usize, f64)) -> f64 {
        let w1 = self.grid_w + 1;
        let xa = (ix + 1).min(self.grid_w);
        let ya = (iy + 1).min(self.grid_h);
        let s00 = self.sat[iy * w1 + ix];
        let s10 = self.sat[iy * w1 + xa];
        let s01 = self.sat[ya * w1 + ix];
        let s11 = self.sat[ya * w1 + xa];
        let top = s00 + fx * (s10 - s00);
        let bot = s01 + fx * (s11 - s01);
        top + fy * (bot - top)
    }

    /// Per-block density sums on a fixed `block × block` grid (row-major,
    /// `ceil(w/block) × ceil(h/block)`, edge blocks clipped) — the same
    /// layout as the `diffmap_block_coherence` harness. Convenience over
    /// [`query_rect`](Self::query_rect); arbitrary partitions should query
    /// their own rectangles.
    ///
    /// On a binned result these sums are exact whenever
    /// [`bin()`](Self::bin) divides `block` (every interior boundary is
    /// bin-aligned; edge blocks clamp at the image boundary, which is always
    /// exact).
    ///
    /// # Panics
    ///
    /// Panics if `block == 0`.
    pub fn block_sums(&self, block: usize) -> Vec<f64> {
        assert!(block > 0, "block size must be non-zero");
        let bx = self.width.div_ceil(block);
        let by = self.height.div_ceil(block);
        let mut out = Vec::with_capacity(bx * by);
        for byi in 0..by {
            for bxi in 0..bx {
                let x0 = bxi * block;
                let y0 = byi * block;
                out.push(self.query_rect(x0, y0, x0 + block, y0 + block));
            }
        }
        out
    }
}

/// Standard SAT recurrence from an index-addressed f64 source.
fn build_sat(get: impl Fn(usize) -> f64, width: usize, height: usize) -> Vec<f64> {
    let mut sat = Vec::new();
    build_sat_into(get, width, height, &mut sat);
    sat
}

/// [`build_sat`] into a reusable buffer: only the guard row and guard
/// column are re-zeroed — every other element is written by the
/// recurrence before any read, so recycled contents are never observable.
/// Bitwise-identical to a fresh build.
fn build_sat_into(get: impl Fn(usize) -> f64, width: usize, height: usize, sat: &mut Vec<f64>) {
    let w1 = width + 1;
    let len = w1 * (height + 1);
    if sat.len() != len {
        sat.clear();
        sat.resize(len, 0.0);
    } else {
        sat[..w1].fill(0.0);
        for y in 1..=height {
            sat[y * w1] = 0.0;
        }
    }
    for y in 0..height {
        let row = y * width;
        let sat_prev = y * w1;
        let sat_row = (y + 1) * w1;
        let (previous, current) = sat.split_at_mut(sat_row);
        let above = &previous[sat_prev + 1..sat_row];
        let current = &mut current[1..w1];
        let mut run = 0.0f64;
        for (x, (cell, &prior)) in current.iter_mut().zip(above).enumerate() {
            run += get(row + x);
            *cell = run + prior;
        }
    }
}

// Row prefix sums are independent. The vertical pass keeps each cell's
// original addition (row prefix + preceding row), with no tree reduction.
fn build_sat_values(values: &[f64], width: usize, height: usize, parallel: bool) -> Vec<f64> {
    #[cfg(feature = "threads")]
    if parallel {
        use rayon::prelude::*;
        let threads = rayon::current_num_threads();
        const ROWS: usize = 64;
        if threads > 1 && height.div_ceil(ROWS) >= threads {
            let stride = width + 1;
            let mut sat = vec![0.0; stride * (height + 1)];
            sat[stride..]
                .par_chunks_mut(stride * ROWS)
                .enumerate()
                .for_each(|(band, rows)| {
                    for (r, row) in rows.chunks_mut(stride).enumerate() {
                        let input = &values[(band * ROWS + r) * width..][..width];
                        let mut run = 0.0;
                        for (cell, &v) in row[1..].iter_mut().zip(input) {
                            run += v;
                            *cell = run;
                        }
                    }
                });
            for y in 1..=height {
                let (above, current) = sat.split_at_mut(y * stride);
                for (cell, &prior) in current[1..stride]
                    .iter_mut()
                    .zip(&above[(y - 1) * stride + 1..])
                {
                    *cell += prior;
                }
            }
            return sat;
        }
    }
    #[cfg(not(feature = "threads"))]
    let _ = parallel;
    build_sat(|i| values[i], width, height)
}

/// Per-(scale, channel) combine coefficients — everything deferred until the
/// channel's pooled sums are known. Each field multiplies the per-pixel term
/// shown in the module-docs table; the `−s_k` sign fold and `1/N` are baked in.
#[derive(Clone, Copy, Default)]
struct SlotCoeffs {
    c_sd: f64,
    c_sd4: f64,
    c_sd2: f64,
    c_art: f64,
    c_art4: f64,
    c_art2: f64,
    c_det: f64,
    c_det4: f64,
    c_det2: f64,
    c_mse: f64,
    /// Coefficient on the SIGNED plane `e = (s−μ1)² − (d−μ2)²` (slots 10/12,
    /// mutually exclusive by the global clamp).
    c_hfe: f64,
    /// Coefficient on the SIGNED plane `|s−μ1| − |d−μ2|` (slot 11).
    c_hfm: f64,
}

impl SlotCoeffs {
    /// Derive coefficients from the model gradient `s` (basic layout, global
    /// index `base_k + slot`), the channel's raw pooled sums, and the scale's
    /// pixel count `n`.
    fn derive(s: &[f64], base_k: usize, acc: &StripChannelAccum, n: f64) -> Self {
        // Gradient lookup: 0 beyond the slice (short/basic-only s vectors).
        let g = |slot: usize| s.get(base_k + slot).copied().unwrap_or(0.0);
        let inv_n = 1.0 / n;
        // p-pooled deferred scalar: (1/p) · M^{(1−p)/p} / N, zero-guarded like
        // `finalize` (M = 0 ⇒ feature 0 ⇒ no gradient; x^{1/p} has an infinite
        // slope at 0 which the removal-consistent form never evaluates).
        let p_coeff = |sk: f64, sum_pow: f64, p: f64| -> f64 {
            let m = sum_pow * inv_n;
            if m > 0.0 {
                -sk * inv_n * (1.0 / p) * m.powf((1.0 - p) / p)
            } else {
                0.0
            }
        };
        let var_src = acc.hf_sq_src * inv_n;
        let var_dst = acc.hf_sq_dst * inv_n;
        let mad_src = acc.hf_abs_src * inv_n;
        let mad_dst = acc.hf_abs_dst * inv_n;
        // hf energy: slot 10 (loss) active when varDst < varSrc, slot 12
        // (gain) when varDst > varSrc — never both (`finalize` clamps the
        // other side to 0). d10 = e/Σsrc², d12 = −e/Σsrc²; fold −s_k.
        //
        // The GAIN half differentiates whichever form the SCORE is using — a
        // density that differentiates a form nobody is running describes a
        // metric nobody is running. The coefficient comes from the same owner
        // as the value, in the accumulator's own units, so the shipped arm's
        // spelling (`1/Σsrc²`) is preserved BIT-for-BIT rather than re-derived
        // through the pooled means.
        let c_hfe = if var_src > crate::hf_gain_form::VAR_SRC_FLOOR {
            if var_dst < var_src {
                -g(10) / acc.hf_sq_src
            } else if var_dst > var_src {
                g(12)
                    * crate::hf_gain_form::hf_energy_gain_d_sum_dst_sq(
                        crate::hf_gain_form::active_gain_form(),
                        acc.hf_sq_src,
                        acc.hf_sq_dst,
                        inv_n,
                    )
            } else {
                0.0
            }
        } else {
            0.0
        };
        let c_hfm = if mad_src > crate::hf_gain_form::VAR_SRC_FLOOR && mad_dst < mad_src {
            -g(11) / acc.hf_abs_src
        } else {
            0.0
        };
        Self {
            c_sd: -g(0) * inv_n,
            c_sd4: p_coeff(g(1), acc.ssim_d4, 4.0),
            c_sd2: p_coeff(g(2), acc.ssim_d2, 2.0),
            c_art: -g(3) * inv_n,
            c_art4: p_coeff(g(4), acc.edge_art4, 4.0),
            c_art2: p_coeff(g(5), acc.edge_art2, 2.0),
            c_det: -g(6) * inv_n,
            c_det4: p_coeff(g(7), acc.edge_det4, 4.0),
            c_det2: p_coeff(g(8), acc.edge_det2, 2.0),
            c_mse: -g(9) * inv_n,
            c_hfe,
            c_hfm,
        }
    }
}

/// Mirror of `ScaleAccumulators::finalize` for the 13 basic slots of one
/// channel — used by the sum-preservation tests to compare the density's
/// full-plane sums against the exact feature values its planes pooled to.
fn basic13_from_acc(a: &StripChannelAccum, n: f64) -> [f64; 13] {
    let inv = 1.0 / n;
    let var_src = a.hf_sq_src * inv;
    let var_dst = a.hf_sq_dst * inv;
    let mad_src = a.hf_abs_src * inv;
    let mad_dst = a.hf_abs_dst * inv;
    let root_form = crate::det_math::active_root_form();
    [
        a.ssim_d * inv,
        (a.ssim_d4 * inv).max(0.0).quarter_root(root_form),
        (a.ssim_d2 * inv).max(0.0).sqrt(),
        a.edge_art * inv,
        (a.edge_art4 * inv).max(0.0).quarter_root(root_form),
        (a.edge_art2 * inv).max(0.0).sqrt(),
        a.edge_det * inv,
        (a.edge_det4 * inv).max(0.0).quarter_root(root_form),
        (a.edge_det2 * inv).max(0.0).sqrt(),
        a.mse * inv,
        crate::hf_gain_form::hf_energy_loss(var_src, var_dst),
        crate::hf_gain_form::hf_mag_loss(mad_src, mad_dst),
        crate::hf_gain_form::hf_energy_gain(
            crate::hf_gain_form::active_gain_form(),
            var_src,
            var_dst,
        ),
    ]
}

/// Field-wise merge for per-band accumulators (the fused kernel returns one
/// per band; sums add, maxes max). Band-index order matches the production
/// sequential merge, keeping f64 addition order identical.
fn merge_acc(a: &mut StripChannelAccum, b: &StripChannelAccum) {
    a.ssim_d += b.ssim_d;
    a.ssim_d4 += b.ssim_d4;
    a.ssim_d2 += b.ssim_d2;
    a.edge_art += b.edge_art;
    a.edge_art4 += b.edge_art4;
    a.edge_art2 += b.edge_art2;
    a.edge_det += b.edge_det;
    a.edge_det4 += b.edge_det4;
    a.edge_det2 += b.edge_det2;
    a.mse += b.mse;
    a.hf_sq_src += b.hf_sq_src;
    a.hf_sq_dst += b.hf_sq_dst;
    a.hf_abs_src += b.hf_abs_src;
    a.hf_abs_dst += b.hf_abs_dst;
    a.ssim_d8 += b.ssim_d8;
    a.edge_art8 += b.edge_art8;
    a.edge_det8 += b.edge_det8;
    a.ssim_max = a.ssim_max.max(b.ssim_max);
    a.edge_art_max = a.edge_art_max.max(b.edge_art_max);
    a.edge_det_max = a.edge_det_max.max(b.edge_det_max);
}

/// Exact boundary terms for one pyramid scale's lattice-step feature (v2 BLOCKINESS).
///
/// The feature is a sum over steps between plane pixels `p`, `q` that are adjacent across an 8-pixel lattice
/// line, of `E = excess(|d_q - d_p|, |s_q - s_p|)`. A rectangle repair copies reference pixels into the pixels
/// it covers, so a step with both ends inside loses all of `E`, a step with one end inside changes to
/// `E_p = excess(|d_q - s_p|, ..)` (p repaired) or `E_q = excess(|s_q - d_p|, ..)` (q repaired), and a step with
/// both ends outside is unchanged. With `g_p = E - E_p`, `g_q = E - E_q` and the pair term `I = E_p + E_q - E`
/// (all multiplied by the feature's map coefficient) the repair gain is `g_p`, `g_q`, or `E = g_p + g_q + I`.
/// The density gives `p` the share `g_p + I/2` and `q` the share `g_q + I/2`, so a rectangle containing both
/// ends is exact by additivity; a rectangle that cuts the step needs `-I/2` per cut step. This stores `I` per
/// step as prefix sums along each lattice line so a rectangle's four edges are answered in O(1).
///
/// Coverage convention matches the density's own sum-preserving upsample: a plane pixel's mass spreads
/// uniformly over its `2^s x 2^s` footprint, so a rectangle covers a fraction `a_p`, `a_q` of the two ends and the
/// correction is `-(I/2) |a_p - a_q|`: `-I/2` for a one-sided aligned cut, `0` when both or neither end is covered
/// (also when both are covered equally but partially, where the density's own area weighting already applies).
/// For footprint-aligned edges (`a` in {0, 1}) this is exact; for other edges it is the stated convention.
#[derive(Clone, Debug, Default)]
pub(crate) struct LatticeScale {
    scale: usize,
    /// Plane dimensions (columns, rows) of this scale.
    ws: usize,
    hs: usize,
    /// Prefix sums over rows for each vertical lattice line `X = 8k` (`k >= 1`): `pv[(k-1)*(hs+1) + y]`.
    pv: Vec<f64>,
    /// Prefix sums over columns for each horizontal lattice line `Y = 8m`: `ph[(m-1)*(ws+1) + x]`.
    ph: Vec<f64>,
    finalized: bool,
}

#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // BLOCKINESS lattice terms are produced by the v2 walk
impl LatticeScale {
    pub(crate) fn new(scale: usize, ws: usize, hs: usize) -> Self {
        Self {
            scale,
            ws,
            hs,
            ..Self::default()
        }
    }

    fn lines_v(&self) -> usize {
        self.ws.saturating_sub(1) / LATTICE_STEP
    }

    fn lines_h(&self) -> usize {
        self.hs.saturating_sub(1) / LATTICE_STEP
    }

    /// Record `value` (the pair term times the coefficient) for the step between columns `x-1` and `x` of row `y`.
    pub(crate) fn add_vertical_step(&mut self, x: usize, y: usize, value: f64) {
        debug_assert!(
            x.is_multiple_of(LATTICE_STEP)
                && x >= LATTICE_STEP
                && x < self.ws
                && y < self.hs
                && !self.finalized
        );
        if self.pv.is_empty() {
            self.pv = vec![0.0; self.lines_v() * (self.hs + 1)];
        }
        let line = x / LATTICE_STEP - 1;
        self.pv[line * (self.hs + 1) + y + 1] += value;
    }

    /// Record `value` for the step between rows `y-1` and `y` of column `x`.
    pub(crate) fn add_horizontal_step(&mut self, y: usize, x: usize, value: f64) {
        debug_assert!(
            y.is_multiple_of(LATTICE_STEP)
                && y >= LATTICE_STEP
                && y < self.hs
                && x < self.ws
                && !self.finalized
        );
        if self.ph.is_empty() {
            self.ph = vec![0.0; self.lines_h() * (self.ws + 1)];
        }
        let line = y / LATTICE_STEP - 1;
        self.ph[line * (self.ws + 1) + x + 1] += value;
    }

    /// Turn the per-step values into prefix sums (`p[i]` = sum of steps `0..i`).
    pub(crate) fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        for line in self.pv.chunks_mut(self.hs + 1) {
            for i in 1..line.len() {
                line[i] += line[i - 1];
            }
        }
        for line in self.ph.chunks_mut(self.ws + 1) {
            for i in 1..line.len() {
                line[i] += line[i - 1];
            }
        }
        self.finalized = true;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pv.is_empty() && self.ph.is_empty()
    }

    /// Fraction of footprint `[i f, (i+1) f)` covered by `[lo, hi)`.
    fn coverage(lo: usize, hi: usize, f: usize, i: usize) -> f64 {
        let a = lo.max(i * f);
        let b = hi.min((i + 1) * f);
        b.saturating_sub(a) as f64 / f as f64
    }

    /// `-(|cp - cq| / 2) * sum_r I_r c_r` over the rows/columns `0..len` of one lattice line whose prefix sums
    /// are `prefix`, where `c_r` is the rectangle's cross-axis coverage of row/column `r` (footprint factor `f`)
    /// and `cp`, `cq` are the coverages of the line's two sides. Only the (at most two) end rows are partial.
    fn line_sum(
        prefix: &[f64],
        len: usize,
        lo: usize,
        hi: usize,
        f: usize,
        cp: f64,
        cq: f64,
    ) -> f64 {
        if lo >= hi {
            return 0.0;
        }
        let first = lo / f;
        let last = ((hi - 1) / f).min(len.saturating_sub(1));
        if first > last {
            return 0.0;
        }
        let step = |r: usize| prefix[r + 1] - prefix[r];
        let mut covered = 0.0;
        for r in [first, last] {
            covered += step(r) * Self::coverage(lo, hi, f, r);
            if first == last {
                break;
            }
        }
        // Rows strictly between the end rows are fully covered.
        if last > first + 1 {
            covered += prefix[last] - prefix[first + 1];
        }
        -0.5 * (cp - cq).abs() * covered
    }

    /// The boundary correction for the rectangle `[x0, x1) x [y0, y1)` in source pixels.
    pub(crate) fn correction(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        if !self.finalized || x0 >= x1 || y0 >= y1 {
            return 0.0;
        }
        let f = 1usize << self.scale;
        let mut total = 0.0;
        // Candidate lattice lines are the ones next to a rectangle edge column; everything else has both ends
        // fully covered or fully uncovered and contributes exactly zero.
        let candidates = |lo: usize, hi: usize, limit: usize| -> [Option<usize>; 4] {
            let first = lo / f;
            let last = (hi - 1) / f;
            let mut out = [None; 4];
            for (slot, c) in out.iter_mut().zip([first, first + 1, last, last + 1]) {
                if c >= LATTICE_STEP && c.is_multiple_of(LATTICE_STEP) && c < limit {
                    *slot = Some(c);
                }
            }
            if out[2] == out[0] || out[2] == out[1] {
                out[2] = None;
            }
            if out[3] == out[0] || out[3] == out[1] || out[3] == out[2] {
                out[3] = None;
            }
            out
        };
        for x in candidates(x0, x1, self.ws).into_iter().flatten() {
            let (cp, cq) = (
                Self::coverage(x0, x1, f, x - 1),
                Self::coverage(x0, x1, f, x),
            );
            if cp == cq {
                continue;
            }
            let line = x / LATTICE_STEP - 1;
            if let Some(prefix) = self
                .pv
                .get(line * (self.hs + 1)..(line + 1) * (self.hs + 1))
            {
                total += Self::line_sum(prefix, self.hs, y0, y1, f, cp, cq);
            }
        }
        for y in candidates(y0, y1, self.hs).into_iter().flatten() {
            let (cp, cq) = (
                Self::coverage(y0, y1, f, y - 1),
                Self::coverage(y0, y1, f, y),
            );
            if cp == cq {
                continue;
            }
            let line = y / LATTICE_STEP - 1;
            if let Some(prefix) = self
                .ph
                .get(line * (self.ws + 1)..(line + 1) * (self.ws + 1))
            {
                total += Self::line_sum(prefix, self.ws, x0, x1, f, cp, cq);
            }
        }
        total
    }
}

/// Per-computation collector of [`LatticeScale`] buffers, one per pyramid scale, owned by the pass-B caller.
/// `logical` is the image the result will be queried against: only steps whose two plane pixels have their whole
/// source footprint inside it are recorded (the others keep the legacy symmetric split).
#[derive(Clone, Debug, Default)]
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // BLOCKINESS lattice terms are produced by the v2 walk
pub(crate) struct LatticeSet {
    logical: (usize, usize),
    enabled: bool,
    scales: Vec<LatticeScale>,
}

#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // BLOCKINESS lattice terms are produced by the v2 walk
impl LatticeSet {
    pub(crate) fn reset(&mut self, logical_w: usize, logical_h: usize, enabled: bool) {
        self.logical = (logical_w, logical_h);
        self.enabled = enabled;
        self.scales.clear();
    }

    /// The buffer and usable plane columns/rows for `scale`, or `None` when exact terms are not available
    /// (disabled, e.g. by a sampling geometry). Give it back with [`Self::put`].
    pub(crate) fn take(
        &mut self,
        scale: usize,
        ws: usize,
        hs: usize,
    ) -> Option<(LatticeScale, usize, usize)> {
        if !self.enabled {
            return None;
        }
        let f = 1usize << scale;
        let limits = (ws.min(self.logical.0 / f), hs.min(self.logical.1 / f));
        // Always a fresh buffer: every `put` buffer already holds recorded steps, and results sum their buffers,
        // so a second `take` for the same scale would stay additive instead of double counting.
        Some((LatticeScale::new(scale, ws, hs), limits.0, limits.1))
    }

    pub(crate) fn put(&mut self, buffer: LatticeScale) {
        if !buffer.is_empty() {
            self.scales.push(buffer);
        }
    }

    pub(crate) fn finish(&mut self) -> Vec<LatticeScale> {
        let mut out = std::mem::take(&mut self.scales);
        for l in &mut out {
            l.finalize();
        }
        out
    }
}

/// Spacing of the BLOCKINESS lattice in plane pixels (`feature_v2::BLOCK_LATTICE`).
const LATTICE_STEP: usize = 8;

/// Level-2 bin accumulator: the `bin × bin` grid the per-scale attribution
/// mass folds into DIRECTLY, so the full-resolution canvas (and its trim
/// copy) never exists for `bin > 1`. Bins are defined over the LOGICAL image
/// — mass landing in the padded-compute margin is clipped at fold time,
/// reproducing the old canvas-trim semantics exactly.
///
/// Sums here differ from the Level-1 fold-of-full-canvas only by float
/// reassociation (each coarse footprint's per-bin overlap is applied as one
/// multiply instead of per-fine-pixel adds) — gated at 1e-9 rel by the
/// `binned_l2_matches_l1_fold` tests. `bin == 1` callers never construct
/// this type: the per-pixel paths are byte-identical to the pre-Level-2 code.
pub(crate) struct BinAccum {
    bins: Vec<f64>,
    parallel: bool,
    bin: usize,
    gw: usize,
    width: usize,
    height: usize,
    /// Lattice-step boundary terms collected by pass B (see [`LatticeScale`]).
    lattice: Vec<LatticeScale>,
}

impl BinAccum {
    pub(crate) fn new(width: usize, height: usize, bin: usize) -> Self {
        assert!(bin > 0, "bin must be non-zero");
        let gw = width.div_ceil(bin);
        let gh = height.div_ceil(bin);
        Self {
            bins: vec![0.0; gw * gh],
            parallel: false,
            bin,
            gw,
            width,
            height,
            lattice: Vec::new(),
        }
    }

    #[cfg(feature = "feature-regime-v2")]
    fn new_reusing(
        width: usize,
        height: usize,
        bin: usize,
        scratch: &mut Vec<f64>,
        parallel: bool,
    ) -> Self {
        assert!(bin > 0, "bin must be non-zero");
        let gw = width.div_ceil(bin);
        let mut bins = std::mem::take(scratch);
        bins.resize(gw * height.div_ceil(bin), 0.0);
        bins.fill(0.0);
        Self {
            bins,
            parallel,
            bin,
            gw,
            width,
            height,
            lattice: Vec::new(),
        }
    }

    #[cfg(feature = "feature-regime-v2")]
    fn into_result_reusing(mut self, scratch: &mut Vec<f64>) -> AttributionResult {
        for lattice in &mut self.lattice {
            lattice.finalize();
        }
        let mut result = AttributionResult::from_bin_values_parallel(
            &self.bins,
            self.width,
            self.height,
            self.bin,
            self.parallel,
        );
        result.lattice = self.lattice;
        *scratch = self.bins;
        result
    }

    /// Research perceptual integration of a per-pixel map
    /// ([`crate::map_paint::smooth_sum_preserving`]); a binned map is left as is.
    #[cfg(feature = "custom-profiles")]
    pub(crate) fn smooth_per_pixel(&mut self, sigma: f64) {
        if self.bin == 1 {
            crate::map_paint::smooth_sum_preserving(&mut self.bins, self.gw, self.height, sigma);
        }
    }

    /// The logical image size queries are made against.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // BLOCKINESS lattice terms are produced by the v2 walk
    pub(crate) fn logical_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Attach the lattice-step boundary terms pass B collected for this accumulation.
    #[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))] // BLOCKINESS lattice terms are produced by the v2 walk
    pub(crate) fn set_lattice(&mut self, lattice: Vec<LatticeScale>) {
        self.lattice = lattice;
    }

    /// Fold one scale plane (coarse dims `sw × sh`, sum-preserving upsample
    /// factor `factor = 2^scale`) into the bins, clipped to the logical
    /// image. Each coarse pixel's mass spreads uniformly over its
    /// `factor × factor` fine footprint; a bin receives
    /// `value / factor² × |footprint ∩ bin ∩ image|`.
    fn add_scale_plane(&mut self, plane: PlaneRef<'_>, sw: usize, sh: usize, factor: usize) {
        if self.bin == 1 {
            self.add_pixel_plane(plane, sw, sh, factor);
            return;
        }
        if factor == 1 {
            // Fast path: row fold with logical clip (sw ≥ width when padded).
            debug_assert!(sw >= self.width && sh >= self.height);
            for y in 0..self.height {
                let row_bins =
                    &mut self.bins[(y / self.bin) * self.gw..(y / self.bin + 1) * self.gw];
                let base = y * sw;
                for (cell, x0) in row_bins.iter_mut().zip((0..self.width).step_by(self.bin)) {
                    let x1 = (x0 + self.bin).min(self.width);
                    let mut run = 0.0f64;
                    for x in x0..x1 {
                        run += plane.get(base + x);
                    }
                    *cell += run;
                }
            }
            return;
        }
        let inv_area = 1.0 / ((factor * factor) as f64);
        if self.bin.is_multiple_of(factor) {
            // Every coarse footprint lies in one bin. Keep each bin's
            // source-row/source-column addition chain and the original
            // multiply order, including underflow and signed zero handling.
            let coarse_bin = self.bin / factor;
            let full_cols = (self.width / factor).min(sw);
            for sy in 0..sh.min(self.height.div_ceil(factor)) {
                let oy = (self.height - sy * factor).min(factor) as f64;
                let area = oy * factor as f64;
                let row_bins =
                    &mut self.bins[(sy / coarse_bin) * self.gw..(sy / coarse_bin + 1) * self.gw];
                for (bx, cell) in row_bins.iter_mut().enumerate() {
                    let first = bx * coarse_bin;
                    let end = (first + coarse_bin).min(full_cols);
                    for sx in first..end {
                        let v = plane.get(sy * sw + sx) * inv_area;
                        if v != 0.0 {
                            *cell += v * area;
                        }
                    }
                }
                let right = self.width % factor;
                if right != 0 && full_cols < sw {
                    let v = plane.get(sy * sw + full_cols) * inv_area;
                    if v != 0.0 {
                        row_bins[full_cols / coarse_bin] += v * (oy * right as f64);
                    }
                }
            }
            return;
        }
        for sy in 0..sh {
            let y0 = sy * factor;
            if y0 >= self.height {
                break;
            }
            let y1 = (y0 + factor).min(self.height);
            let base = sy * sw;
            for sx in 0..sw {
                let x0 = sx * factor;
                if x0 >= self.width {
                    break;
                }
                let x1 = (x0 + factor).min(self.width);
                let v = plane.get(base + sx) * inv_area;
                if v == 0.0 {
                    continue;
                }
                let mut by = y0 / self.bin;
                while by * self.bin < y1 {
                    let oy = (y1.min((by + 1) * self.bin) - y0.max(by * self.bin)) as f64;
                    let mut bx = x0 / self.bin;
                    while bx * self.bin < x1 {
                        let ox = (x1.min((bx + 1) * self.bin) - x0.max(bx * self.bin)) as f64;
                        self.bins[by * self.gw + bx] += v * (oy * ox);
                        bx += 1;
                    }
                    by += 1;
                }
            }
        }
    }

    /// Pixel bins need no footprint/bin intersection or integer division.
    /// Each destination receives one addition per scale, in the same scale
    /// order as the general fold. Keep its zero handling and multiply order.
    fn add_pixel_plane(&mut self, plane: PlaneRef<'_>, sw: usize, sh: usize, factor: usize) {
        let width = self.width;
        if width == 0 {
            return;
        }
        if factor == 1 {
            debug_assert!(sw >= width && sh >= self.height);
        }
        let rows = self.height.min(sh * factor);
        let run = |first: usize, bins: &mut [f64]| {
            add_pixel_band(plane, sw, width, factor, first, bins);
        };
        let _ = self.parallel;
        // Keep an underfilled fold on the serial SIMD path. Starting fewer
        // independent bands than workers adds a pool rendezvous without
        // supplying one band per worker; the arithmetic stays identical.
        #[cfg(feature = "threads")]
        if self.parallel
            && rows > 64
            && rayon::current_num_threads() > 1
            && rows.div_ceil(64) >= rayon::current_num_threads()
        {
            use rayon::prelude::*;
            self.bins[..rows * width]
                .par_chunks_mut(64 * width)
                .enumerate()
                .for_each(|(band, bins)| run(band * 64, bins));
            return;
        }
        run(0, &mut self.bins[..rows * width]);
    }

    pub(crate) fn add_scale_plane_f64(
        &mut self,
        plane: &[f64],
        sw: usize,
        sh: usize,
        factor: usize,
    ) {
        self.add_scale_plane(PlaneRef::F64(plane), sw, sh, factor);
    }

    pub(crate) fn add_scale_plane_f32(
        &mut self,
        plane: &[f32],
        sw: usize,
        sh: usize,
        factor: usize,
    ) {
        self.add_scale_plane(PlaneRef::F32(plane), sw, sh, factor);
    }

    /// [`Self::add_scale_plane_f32`] under the research paint mode
    /// ([`crate::map_paint`]): at `bin == 1` with a source-resolution guide,
    /// `PaintMode::Guided` splits each coarse pixel over its footprint in
    /// proportion to the guide; otherwise the nearest-neighbour fold.
    #[cfg(feature = "custom-profiles")]
    pub(crate) fn add_scale_plane_painted_f32(
        &mut self,
        plane: &[f32],
        sw: usize,
        sh: usize,
        scale: usize,
        paint: crate::map_paint::PaintMode,
        guide0: Option<&[f64]>,
    ) {
        if scale > 0
            && self.bin == 1
            && paint == crate::map_paint::PaintMode::Guided
            && let Some(g) = guide0.filter(|g| g.len() == self.width * self.height)
        {
            let full =
                crate::map_paint::paint_guided(plane, sw, sh, scale, g, self.width, self.height);
            self.add_scale_plane(PlaneRef::F32(&full), self.width, self.height, 1);
            return;
        }
        self.add_scale_plane(PlaneRef::F32(plane), sw, sh, 1usize << scale);
    }

    fn into_result(self) -> AttributionResult {
        let mut lattice = self.lattice;
        for l in &mut lattice {
            l.finalize();
        }
        let mut result =
            AttributionResult::from_bin_sums(self.bins, self.width, self.height, self.bin);
        result.lattice = lattice;
        result
    }
}

// Independent destination rows retain the original per-pixel addition order.
// Dispatch widens only these elementwise operations; no reduction is reordered.
#[autoversion]
fn add_pixel_band(
    plane: PlaneRef<'_>,
    sw: usize,
    width: usize,
    factor: usize,
    first: usize,
    bins: &mut [f64],
) {
    for (offset, row) in bins.chunks_exact_mut(width).enumerate() {
        let base = ((first + offset) / factor) * sw;
        let source = match plane {
            PlaneRef::F32(src) => PlaneRef::F32(&src[base..base + sw]),
            PlaneRef::F64(src) => PlaneRef::F64(&src[base..base + sw]),
        };
        match factor {
            1 => add_pixel_unit_row(source, row),
            2 => add_pixel_scaled_row::<2>(source, row),
            4 => add_pixel_scaled_row::<4>(source, row),
            8 => add_pixel_scaled_row::<8>(source, row),
            _ => {
                let inv_area = 1.0 / (factor * factor) as f64;
                for (sx, pixels) in row.chunks_mut(factor).enumerate().take(sw) {
                    let v = source.get(sx) * inv_area;
                    if v != 0.0 {
                        for cell in pixels {
                            *cell += v;
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn add_pixel_unit_row(plane: PlaneRef<'_>, row: &mut [f64]) {
    match plane {
        PlaneRef::F32(src) => {
            for (cell, &v) in row.iter_mut().zip(src) {
                *cell += 0.0 + f64::from(v);
            }
        }
        PlaneRef::F64(src) => {
            for (cell, &v) in row.iter_mut().zip(src) {
                *cell += 0.0 + v;
            }
        }
    }
}

#[inline(always)]
fn add_pixel_scaled_row<const FACTOR: usize>(plane: PlaneRef<'_>, row: &mut [f64]) {
    let inv_area = 1.0 / (FACTOR * FACTOR) as f64;
    let len = match plane {
        PlaneRef::F32(v) => v.len(),
        PlaneRef::F64(v) => v.len(),
    };
    for (sx, pixels) in row.chunks_mut(FACTOR).enumerate().take(len) {
        let v = plane.get(sx) * inv_area;
        if v != 0.0 {
            for cell in pixels {
                *cell += v;
            }
        }
    }
}

/// Source-plane dtype adapter for [`BinAccum`] — one fold algorithm for the
/// f64 (standalone) and f32 (fused) accumulation pipelines.
#[derive(Clone, Copy)]
enum PlaneRef<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

impl PlaneRef<'_> {
    #[inline(always)]
    fn get(&self, i: usize) -> f64 {
        match self {
            PlaneRef::F64(p) => p[i],
            PlaneRef::F32(p) => p[i] as f64,
        }
    }
}

/// Where per-scale attribution mass lands: the full-resolution canvas
/// (per-pixel results — the pre-Level-2 code path, byte-identical) or the
/// Level-2 bin accumulator (`bin > 1`).
pub(crate) enum AttrSinkF64<'a> {
    Canvas(&'a mut [f64]),
    Bins(&'a mut BinAccum),
}

/// f32 twin of [`AttrSinkF64`] for the fused pipelines.
pub(crate) enum AttrSinkF32<'a> {
    Canvas(&'a mut [f32]),
    Bins(&'a mut BinAccum),
}

/// Sum-preserving power-of-2 upsample-add: each scale pixel's value spreads
/// uniformly over its `factor × factor` footprint ÷ area. With floor-halved
/// pyramid dims the footprint never exceeds the canvas, so rectangle sums
/// over the canvas reproduce coarse-plane sums exactly (÷ by powers of two
/// is exact in f64); the `min` clamps are defensive only.
fn upsample_add_sum_preserving(
    scale_plane: &[f64],
    sw: usize,
    sh: usize,
    canvas: &mut [f64],
    cw: usize,
    ch: usize,
    factor: usize,
) {
    if factor == 1 {
        debug_assert_eq!(sw, cw);
        for (c, &v) in canvas.iter_mut().zip(scale_plane.iter()) {
            *c += v;
        }
        return;
    }
    let inv_area = 1.0 / ((factor * factor) as f64);
    for sy in 0..sh {
        let y0 = sy * factor;
        let y1 = (y0 + factor).min(ch);
        let src_row = sy * sw;
        for sx in 0..sw {
            let v = scale_plane[src_row + sx] * inv_area;
            let x0 = sx * factor;
            let x1 = (x0 + factor).min(cw);
            for row in canvas[y0 * cw..].chunks_mut(cw).take(y1.saturating_sub(y0)) {
                for slot in &mut row[x0..x1] {
                    *slot += v;
                }
            }
        }
    }
}

/// One channel of one scale, banded exactly like the production pipeline:
/// whole-plane fused H-blur, then per-32-row-band fused V-blur + feature
/// accumulation with `±blur_radius` overlap slices. Fills `sd`/`mu1`/`mu2`
/// (full scale planes) and returns the merged accumulator.
fn process_channel_banded(
    src_c: &[f32],
    dst_c: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    bufs: &mut ChannelBuffers,
) -> StripChannelAccum {
    let n = w * h;
    fused_blur_h_ssim(
        &src_c[..n],
        &dst_c[..n],
        &mut bufs.h_mu1[..n],
        &mut bufs.h_mu2[..n],
        &mut bufs.h_ssq[..n],
        &mut bufs.h_s12[..n],
        w,
        h,
        radius,
    );

    let mut acc = StripChannelAccum::zero();
    let overlap = radius; // blur_passes == 1 (validated by the caller)
    let mut y = 0usize;
    while y < h {
        let inner_end = (y + BAND_ROWS).min(h);
        let top = y.saturating_sub(overlap);
        let bot = (inner_end + overlap).min(h);
        let off = top * w;
        let band_n = (bot - top) * w;
        let band = fused_vblur_features_ssim(
            &bufs.h_mu1[off..off + band_n],
            &bufs.h_mu2[off..off + band_n],
            &bufs.h_ssq[off..off + band_n],
            &bufs.h_s12[off..off + band_n],
            &src_c[off..off + band_n],
            &dst_c[off..off + band_n],
            w,
            bot - top,
            y - top,
            inner_end - y,
            radius,
            &mut bufs.mu1[off..off + band_n],
            &mut bufs.mu2[off..off + band_n],
            true,
            &mut bufs.sd[off..off + band_n],
            true,
            // No sigma side-output here: the attribution walk re-derives what
            // it needs; the v1-pool replay is the only sigma consumer.
            &mut [],
            &mut [],
            false,
            // The attribution walk emits no append/append2 slots, so it never
            // needs the free raw moments.
            crate::fused::FreeExtrasWork::default(),
            crate::fused::ExtPoolsWork::default(),
            &[],
        );
        merge_acc(&mut acc, &band);
        y = inner_end;
    }
    acc
}

/// Reusable per-scale scratch (sized once at scale-0 dims, sliced per scale).
struct ChannelBuffers {
    h_mu1: Vec<f32>,
    h_mu2: Vec<f32>,
    h_ssq: Vec<f32>,
    h_s12: Vec<f32>,
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    sd: Vec<f32>,
}

impl ChannelBuffers {
    fn new(n: usize) -> Self {
        Self {
            h_mu1: vec![0.0; n],
            h_mu2: vec![0.0; n],
            h_ssq: vec![0.0; n],
            h_s12: vec![0.0; n],
            mu1: vec![0.0; n],
            mu2: vec![0.0; n],
            sd: vec![0.0; n],
        }
    }
}

/// Exact-integrand combine (f64) for one channel: all seven signals
/// re-derived from the SAME planes the pooled features consumed. Mass is
/// routed by SUPPORT class (C2b bleed allocation, MEASURED variant:
/// window-only spread):
///  - ssim terms → the blur-window plane (per-pixel value is a pure
///    function of K-blurred planes);
///  - art/det/hf terms → the pixel plane (residual form `d_i − (K∗d)_i`;
///    the 50/50 pixel/window split was measured and REGRESSED all 8 gate
///    cells, and the pure `I − K` adjoint allocates zero net mass — both
///    recorded in the C2b benchmark section);
///  - mse → the pixel plane (no blur in its signal).
fn basic_combine_channel(
    bufs: &ChannelBuffers,
    src_c: &[f32],
    dst_c: &[f32],
    n: usize,
    co: &SlotCoeffs,
    id_slice: &mut [f64],
    bl_slice: &mut [f64],
    // Revision 7: the v1 edge ratio as `(x − y)/(1 + y)`.
    stable7: bool,
) {
    for i in 0..n {
        let sd = bufs.sd[i] as f64;
        let sv = src_c[i] as f64;
        let dv = dst_c[i] as f64;
        let m1 = bufs.mu1[i] as f64;
        let m2 = bufs.mu2[i] as f64;
        let d1 = (sv - m1).abs();
        let d2 = (dv - m2).abs();
        let ed = if stable7 {
            (d2 - d1) / (1.0 + d1)
        } else {
            (1.0 + d2) / (1.0 + d1) - 1.0
        };
        let art = ed.max(0.0);
        let det = (-ed).max(0.0);
        let sd2 = sd * sd;
        let a2 = art * art;
        let dt2 = det * det;
        let pd = sv - dv;
        let e_hf = d1 * d1 - d2 * d2;
        let win_term = co.c_sd * sd + co.c_sd4 * (sd2 * sd2) + co.c_sd2 * sd2;
        let res_term = co.c_art * art
            + co.c_art4 * (a2 * a2)
            + co.c_art2 * a2
            + co.c_det * det
            + co.c_det4 * (dt2 * dt2)
            + co.c_det2 * dt2
            + co.c_hfe * e_hf
            + co.c_hfm * (d1 - d2);
        let px_term = co.c_mse * (pd * pd);
        id_slice[i] += px_term + res_term;
        bl_slice[i] += win_term;
    }
}

/// Core builder: multi-scale pyramid walk over (reference pyramid, distorted
/// XYB planes), exact-integrand combine per (scale, channel), sum-preserving
/// upsample into a full-resolution f64 canvas at padded-compute dims.
///
/// Returns `(canvas, basic_features)` where `basic_features` mirrors the
/// production basic block (13 × 3 × scales, scale-major) — consumed by the
/// sum-preservation tests.
///
/// **Takes `&impl MultiScaleRef`, not `&PrecomputedReference`** (fold-engine
/// lane, stage 4). The predecessor lane recorded "attribution's basic canvas
/// is buffered-native" as retirement blocker 4; read at source that is true of
/// the concrete TYPE and not of the WALK — this builder and
/// [`build_attribution_into_sink`] call only `crate::blur`, never
/// `compute_multiscale_stats_streaming`, `process_scale_bands`, or any other
/// walk function. The generic signature is that statement made structural: the
/// canvas needs a pyramid to read, nothing more, and it survives the deletion
/// of the buffered walk provided the pyramid cache does.
fn build_attribution_canvas(
    pyramid: &impl MultiScaleRef,
    dst_planes: [Vec<f32>; 3],
    comp_pw: usize,
    comp_h: usize,
    num_scales: usize,
    radius: usize,
    parallel: bool,
    s: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut canvas = vec![0.0f64; comp_pw * comp_h];
    let own_features = build_attribution_into_sink(
        pyramid,
        dst_planes,
        comp_pw,
        comp_h,
        num_scales,
        radius,
        parallel,
        s,
        &mut AttrSinkF64::Canvas(&mut canvas),
    );
    (canvas, own_features)
}

/// [`build_attribution_canvas`] generalized over the mass sink: the Canvas
/// arm is the byte-identical pre-Level-2 path; the Bins arm folds each
/// scale's density straight into the Level-2 accumulator so no
/// full-resolution canvas exists.
#[allow(clippy::too_many_arguments)]
fn build_attribution_into_sink(
    pyramid: &impl MultiScaleRef,
    mut dst_planes: [Vec<f32>; 3],
    comp_pw: usize,
    comp_h: usize,
    num_scales: usize,
    radius: usize,
    parallel: bool,
    s: &[f64],
    sink: &mut AttrSinkF64<'_>,
) -> Vec<f64> {
    const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
    let mut own_features = Vec::with_capacity(num_scales * FPC * 3);
    let n0 = comp_pw * comp_h;
    let mut bufs3: [ChannelBuffers; 3] = [
        ChannelBuffers::new(n0),
        ChannelBuffers::new(n0),
        ChannelBuffers::new(n0),
    ];
    let mut chan_id: [Vec<f64>; 3] = [vec![0.0; n0], vec![0.0; n0], vec![0.0; n0]];
    let mut chan_win: [Vec<f64>; 3] = [vec![0.0; n0], vec![0.0; n0], vec![0.0; n0]];
    let mut scale_density = vec![0.0f64; comp_pw * comp_h];
    let mut spread_plane = vec![0.0f64; comp_pw * comp_h];
    let mut spread_tmp: Vec<f64> = Vec::new();

    let mut w = comp_pw;
    let mut h = comp_h;
    for scale in 0..num_scales {
        if w < 8 || h < 8 {
            break;
        }
        let (src_planes, sw, sh) = pyramid.scale(scale);
        assert_eq!(w, sw, "attribution: width mismatch at scale {scale}");
        assert_eq!(h, sh, "attribution: height mismatch at scale {scale}");
        let n = sw * sh;
        let n_f = n as f64;
        scale_density[..n].fill(0.0);
        spread_plane[..n].fill(0.0);

        // Per-channel work is independent (own blur buffers, own output
        // planes) — channel-parallel under `threads` (C2b Part 2).
        let mut accs: [Option<StripChannelAccum>; 3] = [None, None, None];
        {
            let [b0, b1, b2] = &mut bufs3;
            let [i0, i1, i2] = &mut chan_id;
            let [w0p, w1p, w2p] = &mut chan_win;
            let run = |c: usize,
                       buf: &mut ChannelBuffers,
                       idp: &mut Vec<f64>,
                       winp: &mut Vec<f64>|
             -> StripChannelAccum {
                idp[..n].fill(0.0);
                winp[..n].fill(0.0);
                let src_c = &src_planes[c][..n];
                let dst_c = &dst_planes[c][..n];
                let acc = process_channel_banded(src_c, dst_c, sw, sh, radius, buf);
                let base_k = scale * FPC * 3 + c * FPC;
                let co = SlotCoeffs::derive(s, base_k, &acc, n as f64);
                basic_combine_channel(
                    buf,
                    src_c,
                    dst_c,
                    n,
                    &co,
                    &mut idp[..n],
                    &mut winp[..n],
                    crate::ssim_form::active_revision().stable_forms(),
                );
                acc
            };
            #[cfg(feature = "threads")]
            if parallel {
                let ((a0, (a1, a2)), ()) = rayon::join(
                    || {
                        rayon::join(
                            || run(0, b0, i0, w0p),
                            || rayon::join(|| run(1, b1, i1, w1p), || run(2, b2, i2, w2p)),
                        )
                    },
                    || (),
                );
                accs = [Some(a0), Some(a1), Some(a2)];
            }
            if accs[0].is_none() {
                accs = [
                    Some(run(0, b0, i0, w0p)),
                    Some(run(1, b1, i1, w1p)),
                    Some(run(2, b2, i2, w2p)),
                ];
            }
        }
        for acc in accs.iter().flatten() {
            own_features.extend_from_slice(&basic13_from_acc(acc, n_f));
        }
        for c in 0..3 {
            for i in 0..n {
                scale_density[i] += chan_id[c][i];
                spread_plane[i] += chan_win[c][i];
            }
        }

        // One sum-preserving spread per scale for the window-supported mass,
        // then fold into the pixel plane.
        crate::blur::box_spread_sum_preserving(
            &mut spread_plane[..n],
            sw,
            sh,
            radius,
            &mut spread_tmp,
        );
        for (d, s) in scale_density[..n].iter_mut().zip(spread_plane[..n].iter()) {
            *d += *s;
        }

        match sink {
            AttrSinkF64::Canvas(canvas) => upsample_add_sum_preserving(
                &scale_density[..n],
                sw,
                sh,
                canvas,
                comp_pw,
                comp_h,
                1usize << scale,
            ),
            AttrSinkF64::Bins(accum) => {
                accum.add_scale_plane_f64(&scale_density[..n], sw, sh, 1usize << scale)
            }
        }

        if scale + 1 < num_scales {
            let (nw, nh) = downscale_3_planes(&mut dst_planes, w, h, parallel);
            w = nw;
            h = nh;
        }
    }
    own_features
}

impl crate::metric::Zensim {
    /// Attribution-density steering map for a scalar model's gradient
    /// (`custom-profiles` research API — see the [module docs](self) for the
    /// mechanism, the exact per-slot integrands, and the honest
    /// approximations list).
    ///
    /// `s` is the model's sensitivity `∂score/∂f_k` over the BASIC feature
    /// block, laid out scale-major (`scale*39 + ch*13 + slot`, i.e. `f0-155`
    /// of the production vector; entries beyond index 155 are ignored, so
    /// passing a full 372/720-wide gradient slice is safe). Gradients are the
    /// caller's job (e.g. central differences through the bake runtime, as in
    /// `examples/diffmap_block_coherence.rs`) — no gradient computation
    /// happens here.
    ///
    /// The result's [`query_rect`](AttributionResult::query_rect) over a
    /// block approximates the first-order score gain from refining that
    /// block; [`block_sums`](AttributionResult::block_sums) gives the fixed
    /// grid the coherence harness scores as `M3a`.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] for zero-dimension inputs,
    /// [`ZensimError::DimensionMismatch`] / [`ZensimError::ImageTooLarge`]
    /// per the usual pair validations, and
    /// [`ZensimError::ModelForwardFailed`] if the active profile uses
    /// `blur_passes != 1` (the banded fused kernels this density reuses are
    /// single-pass; every shipped profile is).
    pub fn compute_attribution_density(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<AttributionResult, ZensimError> {
        validate_pair(source, distorted)?;
        let precomputed = self.precompute_reference(source)?;
        self.compute_attribution_density_with_ref(&precomputed, distorted, s)
    }

    /// [`compute_attribution_density`](Self::compute_attribution_density)
    /// against a precomputed reference — the encoder-loop form (precompute
    /// once, re-query per candidate encode).
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    pub fn compute_attribution_density_with_ref(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<AttributionResult, ZensimError> {
        self.compute_attribution_density_with_ref_binned(precomputed, distorted, s, 1)
    }

    /// [`compute_attribution_density`](Self::compute_attribution_density)
    /// with the result folded to a `bin × bin` grid — `bin²` less retained
    /// memory and SAT-build work, exact for bin-aligned queries (see the
    /// [`AttributionResult`] subsampling docs). `bin == 1` is bit-identical
    /// to the per-pixel entry point.
    ///
    /// # Panics
    ///
    /// Panics if `bin == 0`.
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    pub fn compute_attribution_density_binned(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s: &[f64],
        bin: usize,
    ) -> Result<AttributionResult, ZensimError> {
        validate_pair(source, distorted)?;
        let precomputed = self.precompute_reference(source)?;
        self.compute_attribution_density_with_ref_binned(&precomputed, distorted, s, bin)
    }

    /// [`compute_attribution_density_with_ref`](Self::compute_attribution_density_with_ref)
    /// with the result folded to a `bin × bin` grid (the encoder-loop form).
    ///
    /// # Panics
    ///
    /// Panics if `bin == 0`.
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    pub fn compute_attribution_density_with_ref_binned(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        bin: usize,
    ) -> Result<AttributionResult, ZensimError> {
        assert!(bin > 0, "bin must be non-zero");
        if bin == 1 {
            let (canvas, width, height) = self.basic_canvas_trimmed(precomputed, distorted, s)?;
            return Ok(AttributionResult::from_f64_canvas(canvas, width, height));
        }
        Ok(self
            .basic_bins(precomputed, distorted, s, bin)?
            .into_result())
    }

    /// FULL-coverage attribution density (task #67 C2a): the BASIC block
    /// density plus exact-integrand densities for the v2 (`f372-719`),
    /// append (`f720-923`) and append2 (`f924-943`) blocks — each included
    /// when `s` extends that far — built by
    /// [`crate::feature_v2::compute_v2_append_attribution`]'s replication of
    /// the production kernels. `s` is the raw full-layout gradient
    /// (`∂score/∂f_k`; 372-, 720-, 924- or 944-wide). The `f156-371`
    /// peak/masked/iw block is not spatializable and is ignored, as in the
    /// harness's structural-zero handling.
    ///
    /// Block bounds are the named [`BLOCK_END_*`](BLOCK_END_BASIC) constants
    /// so a future regime bump cannot silently drop a block the way the
    /// hard-coded `924` bound dropped append2 (found 2026-08-04; campaign
    /// appendix E). `attribution_covers_expected_slots_per_width` is the
    /// guard that fails if the covered slot set ever drifts.
    ///
    /// Remaining documented approximations: first-order integrands
    /// throughout, blur bleed unmodeled, finalize clamps treated as inert
    /// (BLOCKINESS lattice steps are exact for footprint-aligned rectangle
    /// repairs, see the module docs, item 2a), reference-only slots (fragility, grad-src-mean, append2 luma-mean-ref) exactly zero,
    /// and the append2 HDR highlight bins structurally zero on this SDR
    /// route. See the module docs here and the `feature_v2` attribution
    /// section.
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_attribution_density_full(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<AttributionResult, ZensimError> {
        self.compute_attribution_density_full_binned(source, distorted, s, 1)
    }

    /// [`compute_attribution_density_full`](Self::compute_attribution_density_full)
    /// with the result folded to a `bin × bin` grid (see the
    /// [`AttributionResult`] subsampling docs). `bin == 1` is bit-identical
    /// to the per-pixel entry point.
    ///
    /// # Panics
    ///
    /// Panics if `bin == 0`.
    ///
    /// # Errors
    ///
    /// Same as [`compute_attribution_density`](Self::compute_attribution_density).
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_attribution_density_full_binned(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        s: &[f64],
        bin: usize,
    ) -> Result<AttributionResult, ZensimError> {
        assert!(bin > 0, "bin must be non-zero");
        validate_pair(source, distorted)?;
        let precomputed = self.precompute_reference(source)?;
        // Half-open [start, end) slice of `s` for one block, empty when `s`
        // does not reach the block. One helper for all three so a new block
        // is one more line, not one more chance to mistype a bound.
        let block = |start: usize, end: usize| -> Option<&[f64]> {
            (s.len() > start).then(|| &s[start..s.len().min(end)])
        };
        // NB: the v2 slice starts at the END of the v1 pooled block (372),
        // not at the end of the basic block — `f156-371` is skipped, not
        // spatialized.
        let s_v2: &[f64] = block(BLOCK_END_V1_POOLS, BLOCK_END_V2).unwrap_or(&[]);
        let s_append: Option<&[f64]> = block(BLOCK_END_V2, BLOCK_END_APPEND);
        let s_append2: Option<&[f64]> = block(BLOCK_END_APPEND, BLOCK_END_APPEND2);
        let want_v2 = !s_v2.is_empty() || s_append.is_some() || s_append2.is_some();

        if bin == 1 {
            let (mut canvas, width, height) =
                self.basic_canvas_trimmed(&precomputed, distorted, s)?;
            let mut lattice = Vec::new();
            if want_v2 {
                let v2a = crate::feature_v2::compute_v2_append_attribution(
                    source,
                    distorted,
                    s_v2,
                    s_append,
                    s_append2,
                    self.max_pixels(),
                    self.parallel(),
                )?;
                debug_assert_eq!((v2a.width, v2a.height), (width, height));
                for (c, v) in canvas.iter_mut().zip(v2a.density.iter()) {
                    *c += *v;
                }
                lattice = v2a.lattice;
            }
            return Ok(
                AttributionResult::from_f64_canvas(canvas, width, height).with_lattice(lattice)
            );
        }

        // Level-2: one BinAccum receives BOTH blocks' per-scale mass — no
        // full-resolution basic canvas, v2 density plane, or trim copies.
        let mut accum = self.basic_bins(&precomputed, distorted, s, bin)?;
        if want_v2 {
            crate::feature_v2::compute_v2_append_attribution_into_bins(
                source,
                distorted,
                s_v2,
                s_append,
                s_append2,
                self.max_pixels(),
                self.parallel(),
                &mut accum,
            )?;
        }
        Ok(accum.into_result())
    }

    /// Shared validation + input prep for the basic attribution builders:
    /// pair checks, blur-config gate, XYB conversion of the distorted at
    /// padded-compute dims. Returns
    /// `(dst_planes, comp_pw, comp_h, num_scales, radius, width, height)`.
    #[allow(clippy::type_complexity)]
    fn basic_attr_prep(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
    ) -> Result<([Vec<f32>; 3], usize, usize, usize, usize, usize, usize), ZensimError> {
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels())?;
        let config = config_from_params(params, self.parallel());
        crate::metric::check_profile_route(params, &config)?;
        crate::metric::validate_cached_revision(precomputed, config.revision())?;
        if config.blur_passes != 1 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "attribution density requires blur_passes == 1 (all shipped profiles)",
            });
        }

        let width = distorted.width();
        let height = distorted.height();
        // Compute dims come from the (possibly reflect-padded) reference
        // pyramid; a sub-64px distorted is reflect-padded to match, and the
        // original image stays in the top-left for the trim/clip below.
        let (_, comp_pw, comp_h) = precomputed.scale(0);
        let dst_planes = if width < MIN_PYRAMID_DIM || height < MIN_PYRAMID_DIM {
            let padded = reflect_pad_to_min(distorted);
            convert_source_to_xyb(&padded, comp_pw, self.parallel(), config.revision())
        } else {
            convert_source_to_xyb(distorted, comp_pw, self.parallel(), config.revision())
        };
        let num_scales = config.num_scales.min(precomputed.num_scales());
        Ok((
            dst_planes,
            comp_pw,
            comp_h,
            num_scales,
            config.blur_radius,
            width,
            height,
        ))
    }

    /// Shared basic-block canvas builder (f64, trimmed to the logical
    /// image): the C1 path and the full-coverage path both start here.
    fn basic_canvas_trimmed(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<(Vec<f64>, usize, usize), ZensimError> {
        let (dst_planes, comp_pw, comp_h, num_scales, radius, width, height) =
            self.basic_attr_prep(precomputed, distorted)?;
        let (canvas, _own_features) = build_attribution_canvas(
            precomputed,
            dst_planes,
            comp_pw,
            comp_h,
            num_scales,
            radius,
            self.parallel(),
            s,
        );

        // Trim the padded-compute canvas to the logical image (top-left).
        let trimmed = if comp_pw == width && comp_h == height {
            canvas
        } else {
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height {
                out.extend_from_slice(&canvas[y * comp_pw..y * comp_pw + width]);
            }
            out
        };
        Ok((trimmed, width, height))
    }

    /// Level-2 sibling of [`basic_canvas_trimmed`](Self::basic_canvas_trimmed):
    /// the per-scale mass folds straight into a [`BinAccum`] — no
    /// full-resolution canvas, no trim copy (the fold clips to the logical
    /// image). Only called with `bin > 1`.
    fn basic_bins(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        bin: usize,
    ) -> Result<BinAccum, ZensimError> {
        debug_assert!(bin > 1);
        let (dst_planes, comp_pw, comp_h, num_scales, radius, width, height) =
            self.basic_attr_prep(precomputed, distorted)?;
        let mut accum = BinAccum::new(width, height, bin);
        build_attribution_into_sink(
            precomputed,
            dst_planes,
            comp_pw,
            comp_h,
            num_scales,
            radius,
            self.parallel(),
            s,
            &mut AttrSinkF64::Bins(&mut accum),
        );
        Ok(accum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SplitMode::Amplitude` carries every p-pool of the basic slots (the L4 and L2 pools of SSIM, artifact and
    /// detail loss) and the three L8 slots of a peak cell as mean pools of their members: no power term is left,
    /// and each member's linear coefficient is the sum of its pools' `−s/N`. The exact split keeps its power terms.
    #[test]
    fn amplitude_split_folds_p_pools_into_the_linear_terms() {
        use crate::map_paint::SplitMode;
        let n = 256.0;
        let stats = crate::metric::ScaleStats {
            ssim: [0.02, 0.3, 0.0, 0.0, 0.0, 0.0],
            ssim_2nd: [0.2, 0.0, 0.0],
            edge: [0.0, 0.25, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            edge_2nd: [0.1, 0.05, 0.0, 0.0, 0.0, 0.0],
            ssim_p95: [0.4, 0.0, 0.0],
            art_p95: [0.35, 0.0, 0.0],
            det_p95: [0.3, 0.0, 0.0],
            ..Default::default()
        };
        let g: [f64; 13] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let form = crate::hf_gain_form::HfGainForm::RatioExcess;
        let co = |split| SlotCoeffs::from_scale_stats(&g, 0, &stats, 0, n, 0.0, 0.0, form, split);
        let (exact, amp) = (co(SplitMode::Exact), co(SplitMode::Amplitude));
        for v in [
            exact.c_sd4,
            exact.c_sd2,
            exact.c_art4,
            exact.c_art2,
            exact.c_det4,
            exact.c_det2,
        ] {
            assert!(v != 0.0, "the exact split keeps its power terms");
        }
        assert_eq!(
            [
                amp.c_sd4, amp.c_sd2, amp.c_art4, amp.c_art2, amp.c_det4, amp.c_det2
            ],
            [0.0; 6]
        );
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * b.abs();
        assert!(close(amp.c_sd, -(1.0 + 2.0 + 3.0) / n), "{}", amp.c_sd);
        assert!(close(amp.c_art, -(4.0 + 5.0 + 6.0) / n), "{}", amp.c_art);
        assert!(close(amp.c_det, -(7.0 + 8.0 + 9.0) / n), "{}", amp.c_det);
        assert_eq!(amp.c_mse, exact.c_mse);
        let mut peaks = [0.0; 6];
        peaks[3..].copy_from_slice(&[2.0, 3.0, 5.0]);
        assert!(
            l8_coefficients(&peaks, 0, 0, &stats, n, SplitMode::Exact)
                .iter()
                .all(|v| *v != 0.0)
        );
        assert_eq!(
            l8_coefficients(&peaks, 0, 0, &stats, n, SplitMode::Amplitude),
            [0.0; 3]
        );
        assert_eq!(
            l8_amplitude_fold(&peaks, 0, 0, n),
            [-2.0 / n, -3.0 / n, -5.0 / n]
        );
    }
    use crate::profile::ProfileParams;
    use crate::{RgbSlice, Zensim, ZensimProfile};
    use std::sync::OnceLock;

    std::thread_local! {
        pub(super) static MAX_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        pub(super) static MAX_REFERENCE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    #[cfg(feature = "feature-regime-v2")]
    fn prepared_max_projection_preserves_public_outputs() {
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let models: Vec<_> = if let Ok(paths) = std::env::var("ZENSIM_MAX_PROJECTION_MODELS") {
            paths
                .split(',')
                .map(|path| crate::mlp::Model::from_bytes(&std::fs::read(path).unwrap()).unwrap())
                .collect()
        } else {
            let recipe = serde_json::json!({
                "schema_hash":1,"scaler_mean":vec![0.0;228],"scaler_scale":vec![1.0;228],
                "layers":[{"in_dim":228,"out_dim":1,"activation":"identity","dtype":"f32",
                    "weights":(0..228).map(|i| -0.001 * (1 + i % 7) as f64).collect::<Vec<_>>(),
                    "biases":[100.0]}]
            });
            let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
            vec![crate::mlp::Model::from_bytes(&bytes).unwrap()]
        };
        let weights = vec![1.0 / models.len() as f64; models.len()];
        let mut comparisons = 0;
        for (w, h) in [(17, 9), (97, 131), (256, 257)] {
            let src: Vec<_> = (0..w * h)
                .map(|i| [(i % 251) as u8, (i * 7 % 239) as u8, (i * 13 % 233) as u8])
                .collect();
            let mut dst = src.clone();
            for i in (0..dst.len()).step_by(11) {
                dst[i] = [0, 255, 0];
            }
            let rs = RgbSlice::new(&src, w, h);
            let ds = RgbSlice::new(&dst, w, h);
            for parallel in [false, true] {
                for finite in [false, true] {
                    let scorer = || {
                        crate::BakeScorer::ensemble(&models, Some(&weights))
                            .unwrap()
                            .with_parallel(parallel)
                            .with_finite_moment_refinement(finite)
                    };
                    let mut old = scorer();
                    let mut new = scorer();
                    let scalar = new.compute(&rs, &ds, None).unwrap();
                    let mut old = old.prepare_steering(&rs, 8).unwrap();
                    let mut new = new.prepare_steering(&rs, 8).unwrap();
                    for image in [&ds, &rs, &ds] {
                        let before = MAX_REFERENCE_CALLS.with(std::cell::Cell::get);
                        MAX_REFERENCE.with(|flag| flag.set(true));
                        let a = old.compute(image, None);
                        MAX_REFERENCE.with(|flag| flag.set(false));
                        let a = a.unwrap();
                        let b = new.compute(image, None).unwrap();
                        if !b.result().is_identical() {
                            assert!(MAX_REFERENCE_CALLS.with(std::cell::Cell::get) > before);
                        }
                        assert_eq!(a.result().score().to_bits(), b.result().score().to_bits());
                        assert_eq!(bits(a.result().features()), bits(b.result().features()));
                        assert_eq!(bits(a.sensitivities()), bits(b.sensitivities()));
                        assert_eq!(a.unsupported_feature_ids(), b.unsupported_feature_ids());
                        assert_eq!(
                            a.unsupported_refinement_feature_ids(),
                            b.unsupported_refinement_feature_ids()
                        );
                        if !b.result().is_identical() {
                            assert_eq!(bits(scalar.features()), bits(b.result().features()));
                            assert_eq!(scalar.score().to_bits(), b.result().score().to_bits());
                        }
                        for (x0, y0, x1, y1) in (0..h)
                            .step_by(8)
                            .flat_map(|y| {
                                (0..w)
                                    .step_by(8)
                                    .map(move |x| (x, y, (x + 8).min(w), (y + 8).min(h)))
                            })
                            .chain([
                                (0, 0, w, h),
                                (1, 1, w - 1, h - 1),
                                (0, 0, 1, 1),
                                (w, h, w, h),
                            ])
                        {
                            assert_eq!(
                                a.attribution().query_rect(x0, y0, x1, y1).to_bits(),
                                b.attribution().query_rect(x0, y0, x1, y1).to_bits()
                            );
                            assert_eq!(
                                a.refinement_gain(x0, y0, x1, y1).to_bits(),
                                b.refinement_gain(x0, y0, x1, y1).to_bits()
                            );
                            comparisons += 1;
                        }
                    }
                }
            }
        }
        eprintln!(
            "public max projection parity: {} model members, {comparisons} rectangle comparisons",
            models.len()
        );
    }

    #[test]
    fn max_projection_matches_pixel_reference_across_sampling_and_masks() {
        let mut contracts = vec![None];
        for kernel in ["triangle", "mitchell", "robidouxsharp"] {
            for (version, channels, ratio) in [
                ("v1", "xyb", "3/2"),
                ("v1", "xyb", "2"),
                ("v1", "xyb", "3"),
                ("v1", "y", "3/2"),
                ("v1", "y", "2"),
                ("v1", "y", "3"),
                ("v2", "xyb", "1,2,4,8"),
                ("v2", "xyb", "1,3,5,7"),
                ("v2", "xyb", "1,2,3,5"),
            ] {
                contracts.push(Some(format!("{version}:{channels}:{kernel}:{ratio}")));
            }
        }
        let mut cases = 0;
        for contract in contracts {
            let sampling = contract.as_ref().map(|contract| {
                let recipe = serde_json::json!({
                    "schema_hash":1,"scaler_mean":[0.0],"scaler_scale":[1.0],
                    "metadata":[{"key":crate::sampling::KEY,"type":"utf8","text":contract}],
                    "layers":[{"in_dim":1,"out_dim":1,"activation":"identity",
                        "dtype":"f32","weights":[1.0],"biases":[0.0]}]
                });
                let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
                crate::sampling::Sampling::from_model(&model)
                    .unwrap()
                    .unwrap()
            });
            for (w, h) in [(1, 1), (17, 9), (97, 83), (129, 131)] {
                let pixels = vec![[0u8; 3]; w * h];
                let source = RgbSlice::new(&pixels, w, h);
                let pre = sampling.map_or_else(
                    || test_zensim().precompute_reference(&source).unwrap(),
                    |s| s.reference(&source, false, crate::ssim_form::active_revision()),
                );
                for scale in 0..4 {
                    let (_, sw, sh) = pre.scale(scale);
                    let mut ret = crate::streaming::AttrScaleRetention::new(sw * sh);
                    let mut src: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; sw * sh]);
                    let mut dst = src.clone();
                    for c in 0..3 {
                        for i in 0..sw * sh {
                            // Zeros, repeated maxima, both edge branches, and
                            // non-binary fractions expose SIMD/reduction drift.
                            src[c][i] = ((i * 17 + c * 3) % 29) as f32 / 31.0;
                            dst[c][i] = ((i * 7 + c * 13) % 31) as f32 / 29.0;
                            ret.mu1[c][i] = ((i * 11 + c) % 23) as f32 / 27.0;
                            ret.mu2[c][i] = ((i * 3 + c) % 19) as f32 / 23.0;
                            ret.sd[c][i] = if c == 0 { 0.0 } else { (i % 11) as f32 / 13.0 };
                        }
                    }
                    for mask in 0..8 {
                        let mut sensitivities = [0.0; 72];
                        let mut expected = 0;
                        for c in 0..3 {
                            for slot in 0..3 {
                                if mask & (1 << ((slot + c) % 3)) != 0 {
                                    sensitivities[(scale * 3 + c) * 6 + slot] =
                                        (c as f64 - 0.5) * 0.7;
                                    expected += 1;
                                }
                            }
                        }
                        let mut maps = Vec::new();
                        // The callee compares every output bit with the old
                        // per-pixel path under cfg(test), including public tests.
                        retain_max_removals(
                            &mut maps,
                            &sensitivities,
                            scale,
                            w,
                            h,
                            sw,
                            sh,
                            src.each_ref().map(Vec::as_slice),
                            dst.each_ref().map(Vec::as_slice),
                            &BasicPlaneView::from(&ret),
                            pre.sampling_geometry.as_ref(),
                        );
                        assert_eq!(maps.len(), expected);
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 3584);
        eprintln!("max projection exact reference cases: {cases}");
    }

    #[test]
    fn max_rectangles_match_explicit_reflected_source_footprints() {
        let mut queries = 0;
        for (w, h) in [(1, 1), (3, 5), (17, 23), (65, 71), (97, 83)] {
            // Coordinate-coded pixels let the actual reflection owner supply
            // an independent expansion, rather than copying its index formula.
            let src: Vec<[u8; 3]> = (0..h)
                .flat_map(|y| (0..w).map(move |x| [x as u8, y as u8, 0]))
                .collect();
            let rs = RgbSlice::new(&src, w, h);
            let extended = reflect_pad_to_min(&rs);
            let pre = test_zensim().precompute_reference(&rs).unwrap();
            let cuts = |n: usize| {
                let mut v = if n <= 5 {
                    (0..=n).collect()
                } else {
                    vec![0, 1, 7, 8, 16, n / 2, n - 1, n]
                };
                v.retain(|x| *x <= n);
                v.sort_unstable();
                v.dedup();
                v
            };
            let xc = cuts(w);
            let yc = cuts(h);
            for scale in 0..4 {
                let (_, sw, sh) = pre.scale(scale);
                let factor = 1 << scale;
                let xs: Vec<Vec<usize>> = (0..sw)
                    .map(|x| {
                        (x * factor..(x + 1) * factor)
                            .map(|i| {
                                usize::from(extended.row_bytes(0)[3 * i.min(extended.width() - 1)])
                            })
                            .collect()
                    })
                    .collect();
                let ys: Vec<Vec<usize>> = (0..sh)
                    .map(|y| {
                        (y * factor..(y + 1) * factor)
                            .map(|i| {
                                usize::from(extended.row_bytes(i.min(extended.height() - 1))[1])
                            })
                            .collect()
                    })
                    .collect();
                let xb = max_axis_footprints(w, sw, scale);
                let yb = max_axis_footprints(h, sh, scale);
                for (expanded, bounds) in xs.iter().zip(&xb).chain(ys.iter().zip(&yb)) {
                    assert_eq!(
                        (
                            *expanded.iter().min().unwrap(),
                            *expanded.iter().max().unwrap()
                        ),
                        *bounds
                    );
                }
                let signals: Vec<f32> = (0..sw * sh)
                    .map(|i| ((i * 13 + i / sw * 7) % 11) as f32)
                    .collect();
                let global = signals.iter().copied().fold(0.0_f32, f32::max);
                let mut map = MaxRemoval::new(156, -1.0, w, h);
                for y in 0..sh {
                    for x in 0..sw {
                        map.add(signals[y * sw + x], xb[x], yb[y]);
                    }
                }
                map.finish();
                assert_eq!(map.global(), f64::from(global));
                assert_eq!(
                    map.feature_drop(0, 0, usize::MAX, usize::MAX),
                    f64::from(global)
                );
                assert_eq!(map.feature_drop(usize::MAX, 0, w, h), 0.0);
                assert_eq!(map.feature_drop(0, usize::MAX, w, h), 0.0);
                for &x0 in &xc {
                    for &x1 in &xc {
                        if x1 < x0 {
                            continue;
                        }
                        let remove_x: Vec<bool> = xs
                            .iter()
                            .map(|v| v.iter().all(|x| x0 <= *x && *x < x1))
                            .collect();
                        for &y0 in &yc {
                            for &y1 in &yc {
                                if y1 < y0 {
                                    continue;
                                }
                                let remove_y: Vec<bool> = ys
                                    .iter()
                                    .map(|v| v.iter().all(|y| y0 <= *y && *y < y1))
                                    .collect();
                                let mut outside = 0.0_f32;
                                for y in 0..sh {
                                    for x in 0..sw {
                                        if !remove_x[x] || !remove_y[y] {
                                            outside = outside.max(signals[y * sw + x]);
                                        }
                                    }
                                }
                                assert_eq!(
                                    map.feature_drop(x0, y0, x1, y1),
                                    f64::from(global) - f64::from(outside),
                                    "{w}x{h} scale{scale} ({x0},{y0})..({x1},{y1})"
                                );
                                queries += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(queries > 10000);
        eprintln!("max source-footprint rectangle queries: {queries}");
    }

    #[test]
    fn max_ties_are_non_additive_and_mismatches_keep_coverage_missing() {
        let mut map = MaxRemoval::new(156, -1.0, 3, 1);
        for (x, v) in [4.0, 2.0, 4.0].into_iter().enumerate() {
            map.add(v, (x, x), (0, 0));
        }
        map.finish();
        assert_eq!(map.feature_drop(0, 0, 1, 1), 0.0);
        assert_eq!(map.feature_drop(1, 0, 3, 1), 0.0);
        assert_eq!(map.feature_drop(0, 0, 3, 1), 4.0);
        #[cfg(feature = "feature-regime-v2")]
        {
            let mut features = vec![0.0; 228];
            features[156] = 4.0;
            let mut maps = vec![map];
            assert_eq!(bind_max_removals(&mut maps, &features, &[156, 228]), [228]);
            features[156] = f64::from_bits(4.0_f64.to_bits() + 1);
            assert_eq!(
                bind_max_removals(&mut maps, &features, &[156, 228]),
                [156, 228]
            );
            assert!(maps.is_empty());
        }
    }

    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn candidate_coverage_distinguishes_variants_reference_only_and_missing_integrands() {
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let wanted = SlotSet::from_slots([0, 156, 159, 227, 228, 377, 924, 926, 927, 944]);
        let mut plan = Plan::derive(&wanted, 960).unwrap();
        let mut sensitivities = vec![0.; 960];
        for id in [0, 156, 159, 227, 228, 377, 924, 926, 927, 944] {
            sensitivities[id] = -1.;
        }
        let (spatial, missing) = candidate_map_sensitivities(&plan, &sensitivities);
        assert_eq!(missing, [156, 228, 944]);
        assert_eq!(spatial[159], -1.);
        assert_eq!(spatial[227], -1.);
        assert_eq!(spatial[377], -1.);
        assert_eq!(spatial[924], -1.);
        assert_eq!(spatial[926], 0.); // Reference-only luma.
        assert_eq!(spatial[927], 0.); // SDR highlight structural zero.
        plan.compute.append2_dst_activity = true;
        let (_, missing) = candidate_map_sensitivities(&plan, &sensitivities);
        assert_eq!(missing, [156, 228, 924, 944]);
        plan.compute.v2_blocks = false;
        let (spatial, missing) = candidate_map_sensitivities(&plan, &sensitivities);
        assert_eq!(missing, [156, 228, 377, 924, 944]);
        assert_eq!(spatial.len(), 228); // No stale v2 retention is consulted.
        assert_eq!(spatial[159], -1.);
        assert_eq!(spatial[227], -1.);
    }

    #[test]
    #[cfg(feature = "feature-regime-v2")]
    fn recycled_bin_storage_preserves_live_results_and_geometry_changes() {
        let mut scratch = Vec::new();
        let mut live = Vec::new();
        for (w, h, bin) in [
            (31, 17, 1),
            (129, 65, 8),
            (32, 9, 1),
            (17, 31, 3),
            (31, 17, 1),
        ] {
            let canvas: Vec<f64> = (0..w * h)
                .map(|i| {
                    if i % 7 == 0 {
                        -0.0
                    } else {
                        (i as f64 - 173.0) / 113.0
                    }
                })
                .collect();
            let mut fresh = BinAccum::new(w, h, bin);
            fresh.add_scale_plane_f64(&canvas, w, h, 1);
            let expected = fresh.into_result();
            let mut accum = BinAccum::new_reusing(w, h, bin, &mut scratch, false);
            accum.add_scale_plane_f64(&canvas, w, h, 1);
            let got = accum.into_result_reusing(&mut scratch);
            assert_eq!(
                got.density.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected
                    .density
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                got.sat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.sat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            live.push((got, expected));
            for (got, expected) in &live {
                assert_eq!(
                    got.query_rect(3, 2, 19, 13).to_bits(),
                    expected.query_rect(3, 2, 19, 13).to_bits()
                );
            }
        }
    }

    #[test]
    #[cfg(all(feature = "feature-regime-v2", feature = "threads"))]
    fn pixel_bin_row_parallelism_preserves_bits() {
        for threads in [1, 8, 16] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for (w, h) in [
                        (31usize, 97usize),
                        (127, 129),
                        (64, 257),
                        (129, 17),
                        (127, 513),
                        (129, 1025),
                    ] {
                        let mut serial = BinAccum::new(w, h, 1);
                        let mut scratch = Vec::new();
                        let mut parallel = BinAccum::new_reusing(w, h, 1, &mut scratch, true);
                        for factor in [1, 2, 4, 8, 3] {
                            let sw = w.div_ceil(factor);
                            let sh = h.div_ceil(factor);
                            let values: Vec<f32> = (0..sw * sh)
                                .map(|i| {
                                    if i % 7 == 0 {
                                        -0.0
                                    } else {
                                        (i as f32 - 73.0) / 113.0
                                    }
                                })
                                .collect();
                            serial.add_scale_plane_f32(&values, sw, sh, factor);
                            parallel.add_scale_plane_f32(&values, sw, sh, factor);
                            assert_eq!(
                                serial.bins.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                parallel
                                    .bins
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                });
        }
    }

    #[test]
    fn fused_l8_matches_separate_passes_with_tails_and_signed_coefficients() {
        for n in [1, 7, 8, 15, 16, 17, 63, 129, 1025] {
            let plane = |seed: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i * 37 + seed * 53) % 257) as f32 / 128.0)
                    .collect()
            };
            let planes = [plane(0), plane(1), plane(2), plane(3), plane(4)];
            let [sd, src, dst, mu1, mu2] = planes.each_ref().map(Vec::as_slice);
            for co in [[0.0; 12], core::array::from_fn(|i| (i as f32 - 5.0) / 13.0)] {
                for l8 in [[1.0, -0.75, 0.5], [1e30, -1e-30, 0.0], [0.0; 3]] {
                    let mut id = plane(5);
                    let mut win = plane(6);
                    let mut expected_id = id.clone();
                    let mut expected_win = win.clone();
                    fused_combine_plane_f32(
                        sd,
                        src,
                        dst,
                        mu1,
                        mu2,
                        co,
                        &mut expected_id,
                        &mut expected_win,
                        false,
                    );
                    fused_combine_l8_f32(
                        sd,
                        src,
                        dst,
                        mu1,
                        mu2,
                        l8,
                        &mut expected_id,
                        &mut expected_win,
                    );
                    combine_basic_and_l8::<true>(
                        sd, src, dst, mu1, mu2, co, l8, &mut id, &mut win, false,
                    );
                    for (a, b) in id
                        .iter()
                        .chain(&win)
                        .zip(expected_id.iter().chain(&expected_win))
                    {
                        assert_eq!(a.to_bits(), b.to_bits(), "length {n}");
                    }
                }
            }
        }
    }

    #[test]
    fn l8_reconstructs_each_canonical_feature() {
        let (w, h) = (128, 128);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim().with_parallel(false);
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let canonical = z.compute_with_ref(&pre, &ds).unwrap();
        for cell in 0..12 {
            for slot in 3..6 {
                let k = cell * 6 + slot;
                let mut s_peaks = [0.0; 72];
                s_peaks[k] = if k % 2 == 0 { -1.0 } else { 0.75 };
                let mut canvas = vec![0.0; w * h];
                let (result, _, _) = z
                    .fused_basic_into(
                        &pre,
                        &ds,
                        &[],
                        &s_peaks,
                        None,
                        None,
                        &mut AttrSinkF32::Canvas(&mut canvas),
                    )
                    .unwrap();
                assert_eq!(result.features(), canonical.features());
                assert_eq!(result.score().to_bits(), canonical.score().to_bits());
                let expected = -s_peaks[k] * canonical.features()[156 + k] / 8.0;
                let actual: f64 = canvas.iter().map(|v| f64::from(*v)).sum();
                if (actual - expected).abs() > 2e-5 * expected.abs().max(1e-12) {
                    panic!("f{}: {actual} != {expected}", 156 + k);
                }
            }
        }
    }

    #[test]
    fn candidate_map_skipping_matches_the_full_legacy_map() {
        if !crate::ssim_form::run_at_revision(
            "3",
            "attribution::tests::candidate_map_skipping_matches_the_full_legacy_map",
            "LEAN-MAP-PARITY",
        ) {
            return;
        }
        for (w, h) in [(17, 9), (127, 97), (256, 193)] {
            let (src, dst) = test_pair(w, h);
            let (rs, ds) = (
                crate::RgbSlice::new(&src, w, h),
                crate::RgbSlice::new(&dst, w, h),
            );
            for parallel in [false, true] {
                let z = crate::Zensim::new(crate::profile::historical_b_for_test())
                    .with_parallel(parallel);
                let pre = z.precompute_reference(&rs).unwrap();
                for mode in 0..4 {
                    let mut s = vec![0.0; 156];
                    for (id, v) in s.iter_mut().enumerate() {
                        if (mode == 0 || id % 13 < 10)
                            && (mode < 2 || !matches!(id,0..13|26..39))
                            && (mode < 3 || matches!(id % 13, 0..=2 | 9))
                        {
                            *v = -(id as f64 + 1.0) / 157.0;
                        }
                    }
                    let run = |revision| {
                        let mut bins = BinAccum::new(w, h, 8);
                        z.fused_basic_into_at_revision(
                            &pre,
                            &ds,
                            &s,
                            &[],
                            None,
                            None,
                            &mut AttrSinkF32::Bins(&mut bins),
                            revision,
                            None,
                            8,
                            None,
                            None,
                            crate::map_paint::MapPaint::default(),
                        )
                        .unwrap();
                        bins.into_result()
                    };
                    let full = run(None);
                    let lean = run(Some(crate::feature_defs::FormulaRevision::Rev3));
                    for y in (0..h).step_by(8) {
                        for x in (0..w).step_by(8) {
                            assert_eq!(
                                full.query_rect(x, y, x + 17, y + 19).to_bits(),
                                lean.query_rect(x, y, x + 17, y + 19).to_bits(),
                                "{w}x{h} mode{mode} parallel{parallel}"
                            );
                        }
                    }
                }
            }
        }
        println!("LEAN-MAP-PARITY");
    }

    #[test]
    fn cached_hf_gain_coefficients_follow_the_served_formula() {
        use crate::hf_gain_form::{HfGainForm, hf_energy_gain};
        // Differentiate the actual pooled feature by perturbing its raw sum.
        // This fails for the former unconditional 1/sum_src derivative.
        for form in [
            HfGainForm::RatioExcess,
            HfGainForm::SaturatingExcess,
            HfGainForm::BoundedExcess,
            HfGainForm::Log1pExcess,
        ] {
            for ratio in [1.1, 2.0, 100.0] {
                let (src, n) = (0.25, 128.0);
                let dst = src * ratio;
                let stats = crate::metric::ScaleStats {
                    hf_sq_dst_sum: [dst; 3],
                    hf_energy_gain: [hf_energy_gain(form, src / n, dst / n); 3],
                    ..Default::default()
                };
                let mut sensitivities = [0.0; 156];
                sensitivities[12] = -2.5;
                let co = SlotCoeffs::from_scale_stats(
                    &sensitivities,
                    0,
                    &stats,
                    0,
                    n,
                    src,
                    0.5,
                    form,
                    crate::map_paint::SplitMode::Repair,
                );
                let eps = dst * 1e-5;
                let fd = -2.5
                    * (hf_energy_gain(form, src / n, (dst + eps) / n)
                        - hf_energy_gain(form, src / n, (dst - eps) / n))
                    / (2.0 * eps);
                assert!(
                    (co.c_hfe - fd).abs() <= 1e-8 * fd.abs().max(1e-6),
                    "{form:?} {ratio}: {} != {fd}",
                    co.c_hfe
                );
                if form == HfGainForm::RatioExcess {
                    assert_eq!(co.c_hfe, -2.5 / src);
                }
            }
        }
    }

    #[test]
    fn l8_finite_moment_removal_and_near_zero_coefficients() {
        // Independent closed-form moment interventions, NOT pixel edits:
        // the latter also move neighboring means and need codec experiments.
        for amplitude in [0.0_f32, 3e-6, 0.1, 1.0] {
            let values: Vec<f32> = (0..64).map(|i| amplitude * (i + 1) as f32 / 64.0).collect();
            let powers: Vec<f64> = values
                .iter()
                .map(|v| {
                    let v2 = v * v;
                    let v4 = v2 * v2;
                    f64::from(v4 * v4)
                })
                .collect();
            let total: f64 = powers.iter().sum();
            let root = (total / 64.0).powf(0.125);
            let stats = crate::metric::ScaleStats {
                ssim_p95: [root; 3],
                ..Default::default()
            };
            let mut s = [0.0; 72];
            s[3] = -256.0;
            let co = l8_coefficients(&s, 0, 0, &stats, 64.0, crate::map_paint::SplitMode::Repair);
            let mut id = [0.0; 64];
            let mut win = [0.0; 64];
            fused_combine_l8_f32(
                &values, &[0.0; 64], &[0.0; 64], &[0.0; 64], &[0.0; 64], co, &mut id, &mut win,
            );
            assert!(win.iter().all(|v| v.is_finite()));
            assert_eq!(id, [0.0; 64]);
            if amplitude == 0.0 {
                assert_eq!(win, [0.0; 64]);
                continue;
            }
            if amplitude == 3e-6 {
                // A f32 coefficient would overflow and poison the map even
                // though its final density is representable and nonzero.
                assert!(co[0] > f64::from(f32::MAX));
                assert!(win.iter().any(|v| *v > 0.0));
            }
            for end in [1, 16, 32, 63, 64] {
                let fraction = powers[..end].iter().sum::<f64>() / total;
                let predicted: f64 = win[..end].iter().map(|v| f64::from(*v)).sum();
                let expected = 256.0 * root * fraction / 8.0;
                assert!((predicted - expected).abs() <= 2e-5 * expected.abs().max(1e-12));
                let exact = 256.0 * root * (1.0 - (1.0 - fraction).powf(0.125));
                // Concavity bounds: first order underpredicts finite removal;
                // at full removal the true effect is eight times larger.
                let tol = 2e-5 * (256.0 * root).max(1e-12);
                assert!(predicted <= exact + tol);
                assert!(exact <= 8.0 * predicted + tol);
                let epsilon = 1e-5;
                let derivative =
                    256.0 * root * (1.0 - (1.0 - epsilon * fraction).powf(0.125)) / epsilon;
                assert!((derivative - predicted).abs() <= 2e-5 * (256.0 * root).max(1e-12));
            }
        }
    }

    #[test]
    fn aligned_bin_fold_matches_independent_footprint_intersections() {
        for (width, height) in [(0, 0), (0, 4), (4, 0), (1, 1), (13, 19), (32, 24), (17, 8)] {
            for factor in [2, 3, 4, 8] {
                for bin in [1, 2, 3, 4, 6, 8, 16] {
                    let (sw, sh) = (width / factor + 3, height / factor + 3);
                    let values64 = [
                        0.0,
                        -0.0,
                        f64::from_bits(1),
                        1e100,
                        -7.0,
                        -1e100,
                        9.125,
                        1e-100,
                    ];
                    let values32 = [
                        0.0,
                        -0.0,
                        f32::from_bits(1),
                        1e20,
                        -7.0,
                        -1e20,
                        9.125,
                        1e-20,
                    ];
                    let p64: Vec<f64> = (0..sw * sh).map(|i| values64[i % 8]).collect();
                    let p32: Vec<f32> = (0..sw * sh).map(|i| values32[i % 8]).collect();
                    for plane in [PlaneRef::F64(&p64), PlaneRef::F32(&p32)] {
                        let mut actual = BinAccum::new(width, height, bin);
                        for (i, x) in actual.bins.iter_mut().enumerate() {
                            *x = i as f64 * 0.125 - 2.0;
                        }
                        let mut expected = actual.bins.clone();
                        // Independent geometry reference: visit every source
                        // footprint and intersect it with every output bin.
                        // Unlike the fast path, no alignment is assumed.
                        for _ in 0..2 {
                            for sy in 0..sh {
                                for sx in 0..sw {
                                    let v =
                                        plane.get(sy * sw + sx) * (1.0 / (factor * factor) as f64);
                                    if v == 0.0 {
                                        continue;
                                    }
                                    for (i, cell) in expected.iter_mut().enumerate() {
                                        let (bx, by) = (i % actual.gw, i / actual.gw);
                                        let ox = ((sx + 1) * factor)
                                            .min((bx + 1) * bin)
                                            .min(width)
                                            .saturating_sub((sx * factor).max(bx * bin));
                                        let oy = ((sy + 1) * factor)
                                            .min((by + 1) * bin)
                                            .min(height)
                                            .saturating_sub((sy * factor).max(by * bin));
                                        if ox != 0 && oy != 0 {
                                            *cell += v * (oy as f64 * ox as f64);
                                        }
                                    }
                                }
                            }
                            actual.add_scale_plane(plane, sw, sh, factor);
                            for (i, (a, b)) in actual.bins.iter().zip(&expected).enumerate() {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "{width}x{height} factor={factor} bin={bin} cell={i}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pixel_bin_unit_factor_matches_frozen_row_fold_bits() {
        for (width, height, stride) in [(0, 4, 3), (4, 0, 7), (1, 1, 4), (13, 19, 16)] {
            let values = [
                0.0_f64,
                -0.0,
                f64::from_bits(1),
                1e100,
                -7.0,
                -1e100,
                9.125,
                1e-100,
            ];
            let plane: Vec<_> = (0..stride * height)
                .map(|i| values[i % values.len()])
                .collect();
            let mut actual = BinAccum::new(width, height, 1);
            actual.bins.fill(-0.0);
            let mut expected = actual.bins.clone();
            for _ in 0..2 {
                // The frozen factor-one fold starts each source-row bin
                // integral at +0, even when its only sample is -0.
                for y in 0..height {
                    for x in 0..width {
                        expected[y * width + x] += 0.0 + plane[y * stride + x];
                    }
                }
                actual.add_scale_plane_f64(&plane, stride, height, 1);
                assert_eq!(
                    actual.bins.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn finite_moment_correction_matches_removed_raw_powers() {
        for p in [2_u32, 4, 8] {
            let values: Vec<f64> = (1..=64).map(|i| f64::from(i) / 64.0).collect();
            let powers: Vec<f64> = values.iter().map(|x| x.powi(p as i32)).collect();
            let total: f64 = powers.iter().sum();
            let root = (total / 64.0).powf(1.0 / f64::from(p));
            let map = MomentRemoval {
                sensitivity: -7.0,
                root,
                power: p,
                total,
                mass: AttributionResult::from_bin_sums(powers.clone(), 64, 1, 1),
            };
            for end in [0, 1, 16, 63, 64] {
                let removed: f64 = powers[..end].iter().sum();
                let next = ((total - removed).max(0.0) / 64.0).powf(1.0 / f64::from(p));
                let first_order = 7.0 * root * removed / (f64::from(p) * total);
                let actual = first_order + map.correction(0, 0, end, 1);
                assert!((actual - 7.0 * (root - next)).abs() < 2e-12);
            }
            assert_eq!(finite_root_extra(0.0, p), 0.0);
            assert!(finite_root_extra(1e-12, p).abs() < 1e-23);
            assert_eq!(finite_root_extra(1.0, p), 1.0 - 1.0 / f64::from(p));
        }
    }

    #[test]
    fn l8_bins_preserve_aligned_queries_on_padded_and_tiny_images() {
        let s_peaks: Vec<f64> = (0..72)
            .map(|k| if k % 6 >= 3 { -1.0 } else { 0.0 })
            .collect();
        for (w, h) in [(1, 1), (17, 23), (97, 83), (128, 160)] {
            let (src, dst) = test_pair(w, h);
            let z = test_zensim();
            let rs = RgbSlice::new(&src, w, h);
            let ds = RgbSlice::new(&dst, w, h);
            let pre = z.precompute_reference(&rs).unwrap();
            let (_, cw, ch) = pre.scale(0);
            let mut canvas = vec![0.0; cw * ch];
            let (canonical, _, _) = z
                .fused_basic_into(
                    &pre,
                    &ds,
                    &[],
                    &s_peaks,
                    None,
                    None,
                    &mut AttrSinkF32::Canvas(&mut canvas),
                )
                .unwrap();
            let mut trimmed = Vec::new();
            for y in 0..h {
                trimmed.extend_from_slice(&canvas[y * cw..y * cw + w]);
            }
            let full = AttributionResult::from_density(trimmed, w, h);
            let tolerance = 2e-5 * full.query_rect(0, 0, w, h).abs().max(1e-12);
            for bin in [1, 8, 16] {
                let mut accum = BinAccum::new(w, h, bin);
                let (result, _, _) = z
                    .fused_basic_into(
                        &pre,
                        &ds,
                        &[],
                        &s_peaks,
                        None,
                        None,
                        &mut AttrSinkF32::Bins(&mut accum),
                    )
                    .unwrap();
                assert_eq!(result.features(), canonical.features());
                assert_eq!(result.score().to_bits(), canonical.score().to_bits());
                let map = accum.into_result();
                assert!(map.density().iter().all(|v| v.is_finite()));
                for y in (0..h).step_by(bin) {
                    for x in (0..w).step_by(bin) {
                        let (x1, y1) = ((x + bin).min(w), (y + bin).min(h));
                        let delta = map.query_rect(x, y, x1, y1) - full.query_rect(x, y, x1, y1);
                        assert!(
                            delta.abs() <= tolerance,
                            "{w}x{h} bin{bin} ({x},{y}): {delta}"
                        );
                    }
                }
            }
        }
    }

    /// Extended-features test profile (all channels/scales active, default
    /// blur/scale config — the same knobs every shipped profile uses).
    fn test_zensim() -> Zensim {
        static PARAMS: OnceLock<ProfileParams> = OnceLock::new();
        Zensim::new(ZensimProfile::Custom {
            params: PARAMS.get_or_init(|| ProfileParams::builder().extended_features(true).build()),
            name: "attribution-test",
        })
    }

    /// Deterministic textured test pair (blocky JPEG-ish distortion).
    fn test_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // Mixed content: gradient + texture + edges.
                let base = ((x * 255) / w) as u8;
                let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
                let edge = if (y / 16) % 2 == 0 { 40 } else { 0 };
                let px = [
                    base.wrapping_add(tex),
                    base.wrapping_add(edge),
                    (255 - base).wrapping_add(tex / 2),
                ];
                src.push(px);
                // Distortion: quantize + local shift in one quadrant.
                let q = |v: u8| (v / 12) * 12;
                let mut d = [q(px[0]), q(px[1]), q(px[2])];
                if x < w / 2 && y < h / 2 {
                    d[0] = d[0].saturating_add(18);
                }
                dst.push(d);
            }
        }
        (src, dst)
    }

    /// SAT queries must equal naive density sums over random rectangles.
    #[test]
    fn sat_matches_naive_rect_sums() {
        let (w, h) = (97, 61); // deliberately non-aligned dims
        // Deterministic pseudo-random signed density.
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let density: Vec<f32> = (0..w * h).map(|_| next() as f32).collect();
        let attr = AttributionResult::from_density(density.clone(), w, h);
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut rnd = move |m: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 33) as usize % m
        };
        for _ in 0..200 {
            let x0 = rnd(w);
            let x1 = x0 + rnd(w - x0) + 1;
            let y0 = rnd(h);
            let y1 = y0 + rnd(h - y0) + 1;
            let mut naive = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    naive += density[y * w + x] as f64;
                }
            }
            let q = attr.query_rect(x0, y0, x1, y1);
            let tol = 1e-6 * naive.abs().max(1e-9) + 1e-9;
            assert!(
                (q - naive).abs() <= tol,
                "rect ({x0},{y0})..({x1},{y1}): sat {q} vs naive {naive}"
            );
        }
        // Degenerate / clamped rectangles.
        assert_eq!(attr.query_rect(5, 5, 5, 9), 0.0);
        assert_eq!(attr.query_rect(30, 20, 10, 25), 0.0);
        let full = attr.query_rect(0, 0, usize::MAX, usize::MAX);
        let total: f64 = density.iter().map(|&v| v as f64).sum();
        assert!((full - total).abs() <= 1e-6 * total.abs().max(1.0));
    }

    /// Deterministic signed f64 canvas for the binning tests.
    fn synth_canvas(w: usize, h: usize) -> Vec<f64> {
        let mut seed = 0xD1B54A32D192ED03u64;
        (0..w * h)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// `bin == 1` through the binned constructor is bit-identical to the
    /// per-pixel constructor (delegation, not reimplementation).
    #[test]
    fn binned_bin1_delegates_bit_identical() {
        let (w, h) = (97, 61);
        let canvas = synth_canvas(w, h);
        let a = AttributionResult::from_f64_canvas(canvas.clone(), w, h);
        let b = AttributionResult::from_f64_canvas_binned(canvas, w, h, 1);
        assert_eq!(a.bin(), b.bin());
        assert_eq!(a.density(), b.density());
        assert_eq!(a.sat, b.sat);
        for &(x0, y0, x1, y1) in &[
            (0usize, 0usize, 97usize, 61usize),
            (3, 5, 42, 37),
            (13, 1, 14, 60),
        ] {
            assert_eq!(a.query_rect(x0, y0, x1, y1), b.query_rect(x0, y0, x1, y1));
        }
    }

    /// Binned results answer bin-aligned and edge-clamped queries exactly
    /// (f64 reassociation only), including `block_sums` when bin | block —
    /// the queries codec partitions actually make.
    #[test]
    fn binned_aligned_and_edge_queries_exact() {
        let (w, h) = (97, 61); // non-multiples of bin: clipped edge bins
        let bin = 8;
        let canvas = synth_canvas(w, h);
        let full = AttributionResult::from_f64_canvas(canvas.clone(), w, h);
        let binned = AttributionResult::from_f64_canvas_binned(canvas, w, h, bin);
        assert_eq!(binned.bin(), bin);
        assert_eq!((binned.grid_width(), binned.grid_height()), (13, 8));
        assert_eq!(binned.density().len(), 13 * 8);
        assert_eq!(binned.sat.len(), 14 * 9);
        let close = |a: f64, b: f64, what: &str| {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1e-9),
                "{what}: binned {a} vs full {b}"
            );
        };
        // Full image (both edges clamp past the clipped bins).
        close(
            binned.query_rect(0, 0, usize::MAX, usize::MAX),
            full.query_rect(0, 0, usize::MAX, usize::MAX),
            "full image",
        );
        // Bin-aligned interior rects + rects clamped at the ragged edge.
        for &(x0, y0, x1, y1) in &[
            (0usize, 0usize, 32usize, 32usize),
            (8, 16, 96, 56),
            (16, 8, 24, 16),
            (80, 40, 97, 61),
            (0, 56, 97, 61),
        ] {
            close(
                binned.query_rect(x0, y0, x1, y1),
                full.query_rect(x0, y0, x1, y1),
                &format!("aligned rect ({x0},{y0})..({x1},{y1})"),
            );
        }
        // block_sums exact for bin | block.
        for block in [16usize, 32] {
            let got = binned.block_sums(block);
            let want = full.block_sums(block);
            assert_eq!(got.len(), want.len());
            for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
                close(*g, *wv, &format!("block_sums({block})[{i}]"));
            }
        }
    }

    /// Unaligned interior queries are area-weighted approximations whose
    /// error is bounded by the |mass| in partially-covered boundary bins.
    #[test]
    fn binned_unaligned_query_bounded() {
        let (w, h) = (96, 64);
        let bin = 8;
        let canvas = synth_canvas(w, h);
        let full = AttributionResult::from_f64_canvas(canvas.clone(), w, h);
        let binned = AttributionResult::from_f64_canvas_binned(canvas.clone(), w, h, bin);
        for &(x0, y0, x1, y1) in &[
            (3usize, 5usize, 42usize, 37usize),
            (10, 2, 61, 59),
            (17, 23, 30, 31),
        ] {
            let exact = full.query_rect(x0, y0, x1, y1);
            let approx = binned.query_rect(x0, y0, x1, y1);
            // Bound: total |mass| of bins the rect intersects but does not
            // fully contain.
            let mut bound = 0.0f64;
            for by in y0 / bin..h.div_ceil(bin).min(y1.div_ceil(bin)) {
                for bx in x0 / bin..w.div_ceil(bin).min(x1.div_ceil(bin)) {
                    let (px0, py0) = (bx * bin, by * bin);
                    let (px1, py1) = ((px0 + bin).min(w), (py0 + bin).min(h));
                    let contained = px0 >= x0 && px1 <= x1 && py0 >= y0 && py1 <= y1;
                    if !contained {
                        for y in py0..py1 {
                            for x in px0..px1 {
                                bound += canvas[y * w + x].abs();
                            }
                        }
                    }
                }
            }
            assert!(
                (approx - exact).abs() <= bound + 1e-9,
                "unaligned ({x0},{y0})..({x1},{y1}): |{approx} - {exact}| > ring bound {bound}"
            );
        }
    }

    /// The exported `f32` view is the bin MEAN over each bin's REAL pixel
    /// count (per-pixel units; edge bins clipped).
    #[test]
    fn binned_density_is_real_pixel_mean() {
        let (w, h) = (97, 61);
        let bin = 8;
        let canvas = synth_canvas(w, h);
        let binned = AttributionResult::from_f64_canvas_binned(canvas.clone(), w, h, bin);
        let gw = binned.grid_width();
        // Interior bin (2,3) and the bottom-right clipped bin (12,7): 1×5 px.
        for (bx, by) in [(2usize, 3usize), (12, 7)] {
            let (px0, py0) = (bx * bin, by * bin);
            let (px1, py1) = ((px0 + bin).min(w), (py0 + bin).min(h));
            let mut sum = 0.0f64;
            for y in py0..py1 {
                for x in px0..px1 {
                    sum += canvas[y * w + x];
                }
            }
            let mean = (sum / ((px1 - px0) * (py1 - py0)) as f64) as f32;
            let got = binned.density()[by * gw + bx];
            assert!(
                (got - mean).abs() <= 1e-6 * mean.abs().max(1e-9),
                "bin ({bx},{by}): density {got} vs real-pixel mean {mean}"
            );
        }
    }

    /// End-to-end: the binned entry points against the real pipeline —
    /// bit-identity at bin=1, aligned-query equality at bin=8.
    #[test]
    fn binned_entry_points_match_per_pixel() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let s = vec![-0.5f64; 156];
        let full = z.compute_attribution_density(&rs, &ds, &s).unwrap();
        let b1 = z
            .compute_attribution_density_binned(&rs, &ds, &s, 1)
            .unwrap();
        assert_eq!(full.density(), b1.density());
        assert_eq!(
            full.query_rect(3, 7, 51, 33),
            b1.query_rect(3, 7, 51, 33),
            "bin=1 entry point must be bit-identical"
        );
        let b8 = z
            .compute_attribution_density_binned(&rs, &ds, &s, 8)
            .unwrap();
        assert_eq!((b8.width(), b8.height()), (w, h));
        assert_eq!((b8.grid_width(), b8.grid_height()), (12, 10));
        let close = |a: f64, b: f64, what: &str| {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1e-9),
                "{what}: binned {a} vs full {b}"
            );
        };
        close(
            b8.query_rect(0, 0, w, h),
            full.query_rect(0, 0, w, h),
            "full image",
        );
        let (got, want) = (b8.block_sums(16), full.block_sums(16));
        assert_eq!(got.len(), want.len());
        for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
            close(*g, *wv, &format!("block_sums(16)[{i}]"));
        }
        #[cfg(feature = "feature-regime-v2")]
        {
            let f8 = z
                .compute_attribution_density_full_binned(&rs, &ds, &s, 8)
                .unwrap();
            let f1 = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
            close(
                f8.query_rect(0, 0, w, h),
                f1.query_rect(0, 0, w, h),
                "full-coverage full image",
            );
        }
    }

    /// Level-2 vs Level-1: the sink accumulation (no full-res canvas) must
    /// agree with folding the full-res canvas, on EVERY bin cell — the
    /// difference is float reassociation only.
    #[test]
    fn binned_l2_matches_l1_fold() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let s = vec![-0.5f64; 156];
        let bin = 8;
        // L1 reference: full canvas, then fold.
        let pre = z.precompute_reference(&rs).unwrap();
        let (canvas, cw, chh) = z.basic_canvas_trimmed(&pre, &ds, &s).unwrap();
        let l1 = AttributionResult::from_f64_canvas_binned(canvas, cw, chh, bin);
        // L2: sink accumulation through the public entry.
        let l2 = z
            .compute_attribution_density_binned(&rs, &ds, &s, bin)
            .unwrap();
        assert_eq!(
            (l2.grid_width(), l2.grid_height()),
            (l1.grid_width(), l1.grid_height())
        );
        for by in 0..l1.grid_height() {
            for bx in 0..l1.grid_width() {
                let (x0, y0) = (bx * bin, by * bin);
                let a = l2.query_rect(x0, y0, x0 + bin, y0 + bin);
                let b = l1.query_rect(x0, y0, x0 + bin, y0 + bin);
                assert!(
                    (a - b).abs() <= 1e-9 * b.abs().max(1e-9),
                    "bin ({bx},{by}): L2 {a} vs L1 {b}"
                );
            }
        }
        // The _full path: both blocks into one accumulator vs the L1 merge.
        #[cfg(feature = "feature-regime-v2")]
        {
            let s944 = vec![-0.05f64; 944];
            let l1f = {
                let f = z.compute_attribution_density_full(&rs, &ds, &s944).unwrap();
                // Fold the per-pixel result's density (f32 view) is lossy;
                // rebuild the L1 reference through the canvas path instead.
                let (mut canvas, cw, chh) = z.basic_canvas_trimmed(&pre, &ds, &s944).unwrap();
                let v2a = crate::feature_v2::compute_v2_append_attribution(
                    &rs,
                    &ds,
                    &s944[372..720],
                    Some(&s944[720..924]),
                    Some(&s944[924..944]),
                    None,
                    false,
                )
                .unwrap();
                for (c, v) in canvas.iter_mut().zip(v2a.density.iter()) {
                    *c += *v;
                }
                drop(f);
                // The L1 reference carries the same exact lattice-step boundary terms as the entry under
                // test (they are part of the map's rectangle semantics, not of its pixel density).
                AttributionResult::from_f64_canvas_binned(canvas, cw, chh, bin)
                    .with_lattice(v2a.lattice)
            };
            let l2f = z
                .compute_attribution_density_full_binned(&rs, &ds, &s944, bin)
                .unwrap();
            for by in 0..l1f.grid_height() {
                for bx in 0..l1f.grid_width() {
                    let (x0, y0) = (bx * bin, by * bin);
                    let a = l2f.query_rect(x0, y0, x0 + bin, y0 + bin);
                    let b = l1f.query_rect(x0, y0, x0 + bin, y0 + bin);
                    assert!(
                        (a - b).abs() <= 1e-9 * b.abs().max(1e-9),
                        "full bin ({bx},{by}): L2 {a} vs L1 {b}"
                    );
                }
            }
        }
    }

    /// Fused binned: score BIT-identical to the per-pixel fused entry; map
    /// agrees on every bin cell within the f32 cross-scale accumulation
    /// class (the binned path accumulates cross-scale in f64 — slightly
    /// more precise than the f32 canvas).
    #[test]
    fn fused_binned_score_bitwise_and_map_close() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let s = vec![-0.5f64; 156];
        let bin = 8;
        let pre = z.precompute_reference(&rs).unwrap();
        let (res_full, attr_full) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        let (res_bin, attr_bin) = z
            .compute_with_ref_score_and_attribution_binned(&pre, &ds, &s, bin)
            .unwrap();
        assert_eq!(
            res_full.score(),
            res_bin.score(),
            "score must be bit-identical"
        );
        assert_eq!(attr_bin.bin(), bin);
        for by in 0..attr_bin.grid_height() {
            for bx in 0..attr_bin.grid_width() {
                let (x0, y0) = (bx * bin, by * bin);
                let a = attr_bin.query_rect(x0, y0, x0 + bin, y0 + bin);
                let b = attr_full.query_rect(x0, y0, x0 + bin, y0 + bin);
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1e-6),
                    "fused bin ({bx},{by}): binned {a} vs full {b}"
                );
            }
        }
    }

    /// Stale binned: the priming call matches the fresh binned entry; the
    /// second same-pair call is exactly reproducible; the session's
    /// full-resolution canvas is never allocated on a binned-only session.
    #[test]
    fn stale_binned_reuse_and_no_canvas() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let s = vec![-0.5f64; 156];
        let bin = 8;
        let pre = z.precompute_reference(&rs).unwrap();
        let mut sess = AttributionSession::new();
        let (_r1, a1) = z
            .compute_with_ref_score_and_attribution_stale_binned(&pre, &ds, &s, &mut sess, bin)
            .unwrap();
        // Priming call == the fresh binned entry (same code path).
        let (_rf, af) = z
            .compute_with_ref_score_and_attribution_binned(&pre, &ds, &s, bin)
            .unwrap();
        for by in 0..a1.grid_height() {
            for bx in 0..a1.grid_width() {
                let (x0, y0) = (bx * bin, by * bin);
                assert_eq!(
                    a1.query_rect(x0, y0, x0 + bin, y0 + bin),
                    af.query_rect(x0, y0, x0 + bin, y0 + bin),
                    "prime bin ({bx},{by})"
                );
            }
        }
        // Second call (single-pass stale) — deterministic across repeats.
        let (_r2, a2) = z
            .compute_with_ref_score_and_attribution_stale_binned(&pre, &ds, &s, &mut sess, bin)
            .unwrap();
        let (_r3, a3) = z
            .compute_with_ref_score_and_attribution_stale_binned(&pre, &ds, &s, &mut sess, bin)
            .unwrap();
        for by in 0..a2.grid_height() {
            for bx in 0..a2.grid_width() {
                let (x0, y0) = (bx * bin, by * bin);
                assert_eq!(
                    a2.query_rect(x0, y0, x0 + bin, y0 + bin),
                    a3.query_rect(x0, y0, x0 + bin, y0 + bin),
                    "stale repeat bin ({bx},{by})"
                );
            }
        }
        assert!(
            sess.canvas.is_empty(),
            "binned-only session must never allocate the full-res canvas"
        );
    }

    /// Fused-944 binned: score + 944 features bit-identical to the
    /// per-pixel entry; map agrees on every bin cell (f32 class).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_binned_matches_per_pixel() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let s = vec![-0.05f64; 944];
        let bin = 8;
        let pre = z.precompute_reference(&rs).unwrap();
        let mut sess_a = Fused944Session::new();
        let mut sess_b = Fused944Session::new();
        let (r1, f1, a1) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess_a)
            .unwrap();
        let (r2, f2, a2) = z
            .compute_folded944_score_and_attribution_binned(&rs, &pre, &ds, &s, &mut sess_b, bin)
            .unwrap();
        assert_eq!(r1.score(), r2.score(), "944 score must be bit-identical");
        assert_eq!(
            f1.features(),
            f2.features(),
            "944 features must be bit-identical"
        );
        for by in 0..a2.grid_height() {
            for bx in 0..a2.grid_width() {
                let (x0, y0) = (bx * bin, by * bin);
                let a = a2.query_rect(x0, y0, x0 + bin, y0 + bin);
                let b = a1.query_rect(x0, y0, x0 + bin, y0 + bin);
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1e-6),
                    "944 bin ({bx},{by}): binned {a} vs full {b}"
                );
            }
        }
    }

    /// Manual END-TO-END perf/memory probe at 12 MP (run `--ignored` in
    /// release mode, under `/usr/bin/time -v` for peak RSS;
    /// `ZENSIM_L2_PROBE_BIN` selects the arm — 1 = per-pixel, 8 = Level-2).
    #[test]
    #[ignore = "manual perf/memory probe — run in release mode"]
    fn l2_end_to_end_probe() {
        let bin: usize = std::env::var("ZENSIM_L2_PROBE_BIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let (w, h) = (4000, 3000);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let s = vec![-0.5f64; 156];
        let t = crate::perf_clock::Instant::now();
        let r = z
            .compute_attribution_density_binned(&rs, &ds, &s, bin)
            .unwrap();
        let el = t.elapsed();
        eprintln!(
            "L2 e2e 12MP bin={bin}: {el:?}, retained {} KB",
            (r.density.len() * 4 + r.sat.len() * 8) / 1024
        );
    }

    /// Manual perf/memory probe (run `--ignored` in release mode):
    /// construction cost + retained bytes, per-pixel vs `bin = 8`, 12 MP.
    #[test]
    #[ignore = "manual perf probe — run in release mode"]
    fn binned_construction_perf_probe() {
        let (w, h) = (4000, 3000);
        let canvas = synth_canvas(w, h);
        let t = crate::perf_clock::Instant::now();
        let full = AttributionResult::from_f64_canvas(canvas.clone(), w, h);
        let t_full = t.elapsed();
        let t = crate::perf_clock::Instant::now();
        let binned = AttributionResult::from_f64_canvas_binned(canvas, w, h, 8);
        let t_bin = t.elapsed();
        let bytes = |r: &AttributionResult| r.density.len() * 4 + r.sat.len() * 8;
        eprintln!(
            "12MP construction: full {t_full:?} ({} MB retained) vs bin=8 {t_bin:?} ({} KB retained)",
            bytes(&full) / (1024 * 1024),
            bytes(&binned) / 1024
        );
    }

    /// block_sums must equal the harness-style naive block partition.
    #[test]
    fn block_sums_match_naive_partition() {
        let (w, h) = (70, 50);
        let density: Vec<f32> = (0..w * h).map(|i| ((i % 17) as f32) - 4.0).collect();
        let attr = AttributionResult::from_density(density.clone(), w, h);
        for block in [16usize, 32, 33] {
            let bx = w.div_ceil(block);
            let by = h.div_ceil(block);
            let mut naive = vec![0.0f64; bx * by];
            for y in 0..h {
                for x in 0..w {
                    naive[(y / block) * bx + (x / block)] += density[y * w + x] as f64;
                }
            }
            let got = attr.block_sums(block);
            assert_eq!(got.len(), naive.len());
            for (b, (g, n)) in got.iter().zip(naive.iter()).enumerate() {
                assert!(
                    (g - n).abs() <= 1e-6 * n.abs().max(1.0),
                    "block {b} ({block}px): {g} vs {n}"
                );
            }
        }
    }

    /// Sum preservation, mean-pooled slots: with `s` = −1 on a single mean
    /// slot, the full-image query equals that feature's value exactly
    /// (density = plane/N; the f64 SAT keeps the identity to summation-order
    /// precision). Covers scale 0 AND scale 1 (exercises the sum-preserving
    /// upsample) on both the ssim (stored-plane) and mse (recomputed) slots.
    /// 64×64: SIMD-padded width == width, so no padding-column loss.
    #[test]
    fn sum_preservation_mean_slots() {
        let (w, h) = (64, 64);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let feats = z.compute_extended_features(&rs, &ds).unwrap();
        let feats = feats.features();
        // (k, tolerance): ssim-mean slots compare against the identical
        // stored f32 plane (1e-9); mse recomputes (s−d)² in f64 vs the
        // kernel's f32 product (1e-6).
        let cases = [
            (13usize, 1e-9),     // scale 0, ch Y, slot 0 (ssim mean)
            (39 + 13, 1e-9),     // scale 1, ch Y, slot 0 — upsample path
            (2 * 39 + 13, 1e-9), // scale 2, ch Y, slot 0
            (9, 1e-6),           // scale 0, ch X, slot 9 (mse)
            (39 + 26 + 9, 1e-6), // scale 1, ch B, slot 9
        ];
        for &(k, tol) in &cases {
            let mut s = vec![0.0f64; 156];
            s[k] = -1.0;
            let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            let full = attr.query_rect(0, 0, w, h);
            let expect = feats[k];
            assert!(
                (full - expect).abs() <= tol * expect.abs().max(1e-12),
                "k={k}: full-image query {full} vs production feature {expect} (rel {})",
                ((full - expect) / expect.abs().max(1e-30)).abs()
            );
        }
    }

    /// p-pooled and hf slots: full-image sums land on the documented
    /// identities — f/p for p-pooled (removal-consistent 1/p integrand),
    /// exactly f for the active hf ratio slots.
    #[test]
    fn sum_preservation_ppool_and_hf_slots() {
        let (w, h) = (64, 64);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let feats = z.compute_extended_features(&rs, &ds).unwrap();
        let feats = feats.features();
        // (k, divisor): slot 1 = ssim p4 → f/4; slot 2 = ssim p2 → f/2;
        // slot 5 = art p2 → f/2 (scale 0, ch Y).
        for &(k, p) in &[(14usize, 4.0), (15, 2.0), (18, 2.0)] {
            let mut s = vec![0.0f64; 156];
            s[k] = -1.0;
            let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            let full = attr.query_rect(0, 0, w, h);
            let expect = feats[k] / p;
            assert!(
                (full - expect).abs() <= 1e-5 * expect.abs().max(1e-12),
                "k={k}: full query {full} vs f/p {expect}"
            );
        }
        // hf slots: whichever of loss/gain is active sums to its feature; the
        // clamped side must contribute an all-zero density. Quantization
        // mostly removes texture → expect energy-loss active on Y at scale 0.
        for &(k_loss, k_gain) in &[(23usize, 25usize)] {
            for &(k, other) in &[(k_loss, k_gain), (k_gain, k_loss)] {
                let mut s = vec![0.0f64; 156];
                s[k] = -1.0;
                let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
                let full = attr.query_rect(0, 0, w, h);
                if feats[k] > 0.0 {
                    assert!(
                        (full - feats[k]).abs() <= 1e-5 * feats[k].max(1e-12),
                        "hf k={k}: full query {full} vs feature {}",
                        feats[k]
                    );
                    assert!(feats[other] == 0.0, "hf clamp pair both active?");
                } else {
                    assert!(
                        full.abs() <= 1e-12,
                        "hf k={k} clamped (feature 0) but density sums to {full}"
                    );
                }
            }
        }
        // hf_mag_loss (slot 11, ch Y scale 0 → k=24).
        let k = 24;
        let mut s = vec![0.0f64; 156];
        s[k] = -1.0;
        let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
        let full = attr.query_rect(0, 0, w, h);
        if feats[k] > 0.0 {
            assert!(
                (full - feats[k]).abs() <= 1e-5 * feats[k].max(1e-12),
                "hf_mag k={k}: {full} vs {}",
                feats[k]
            );
        } else {
            assert!(full.abs() <= 1e-12);
        }
    }

    /// Identical images: every signal plane is ~0, so the density is ~0
    /// everywhere regardless of `s`. The fused blur path carries ~3e-5
    /// float-precision noise on identical pairs (same floor the diffmap
    /// identical-images test documents), which the hf ratio slots divide by
    /// a near-cancelling Σ — 1e-4 is the established noise bound, not a
    /// correctness tolerance.
    #[test]
    fn identical_images_zero_density() {
        let (w, h) = (64, 64);
        let (src, _) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let s = vec![-1.0f64; 156];
        let attr = z.compute_attribution_density(&rs, &rs, &s).unwrap();
        let max = attr.density().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max < 1e-4, "identical-pair density max {max}");
        assert_eq!(attr.width(), w);
        assert_eq!(attr.height(), h);
        assert_eq!(attr.density().len(), w * h);
    }

    /// `compute_attribution_density_full` wiring: with a gradient that is
    /// zero beyond the basic block it must equal the basic-only path
    /// exactly, and with v2/append weights present the result must equal
    /// basic + the feature_v2 density (same trimmed canvas addition).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn full_density_is_basic_plus_v2_append() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let mut s = vec![0.0f64; 924];
        s[13] = -0.7; // basic: scale 0, ch Y, ssim mean
        let full_basic_only = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
        let basic = z.compute_attribution_density(&rs, &ds, &s[..156]).unwrap();
        for (a, b) in full_basic_only.density().iter().zip(basic.density().iter()) {
            assert!((a - b).abs() <= 1e-12, "basic-only full != basic path");
        }
        s[372 + 29 + 3] = -0.4; // v2 ART, scale 0, ch Y
        s[720 + 17 + 9] = -0.3; // append TEXTURE_DISSIM, scale 0, ch Y
        let full = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
        let v2a = crate::feature_v2::compute_v2_append_attribution(
            &rs,
            &ds,
            &s[372..720],
            Some(&s[720..924]),
            None,
            None,
            false,
        )
        .unwrap();
        for i in 0..w * h {
            let expect = basic.density()[i] as f64 + v2a.density[i];
            let got = full.density()[i] as f64;
            assert!(
                (got - expect).abs() <= 1e-6 * expect.abs().max(1e-9) + 1e-9,
                "pixel {i}: full {got} vs basic+v2app {expect}"
            );
        }
    }

    /// `LatticeSet`: a disabled set (the sampling-geometry path) hands out nothing and finishes empty; a repeated
    /// `take` for one scale yields independent buffers whose corrections add (no reuse of recorded steps).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn lattice_set_disabled_and_repeated_take() {
        let mut set = LatticeSet::default();
        set.reset(64, 48, false);
        assert!(set.take(0, 64, 48).is_none());
        assert!(set.finish().is_empty());
        set.reset(64, 48, true);
        let (mut a, lw, lh) = set.take(0, 64, 48).unwrap();
        assert_eq!((lw, lh), (64, 48));
        a.add_vertical_step(8, 3, 0.5);
        set.put(a);
        let (mut b, _, _) = set.take(0, 64, 48).unwrap();
        assert!(
            b.is_empty(),
            "a second take must start empty, not reuse recorded steps"
        );
        b.add_vertical_step(8, 3, 0.25);
        set.put(b);
        let terms = set.finish();
        assert_eq!(terms.len(), 2);
        let sum: f64 = terms.iter().map(|l| l.correction(8, 0, 16, 8)).sum();
        // One-sided cut at x = 8 through row 3: -I/2 per buffer.
        assert!((sum - (-0.5 * (0.5 + 0.25))).abs() < 1e-15, "{sum}");
        // Scale 1 with a 70-pixel logical image: plane limit is 35 columns.
        set.reset(70, 50, true);
        let (_, lw, lh) = set.take(1, 64, 48).unwrap();
        assert_eq!((lw, lh), (35, 25));
    }

    /// BLOCKINESS maps are exact for rectangle repairs through every pipeline (the 2026-10-02 v2spatial fix).
    ///
    /// For plane-aligned rectangles on an image whose scales all fit (256x192: scale-3 plane 32x24), the
    /// predicted map mass of each BLOCKINESS slot (unit sensitivity on that one slot) must equal the slot's
    /// actual feature change under the finite repair "copy the reference into the rectangle", through the f64
    /// per-pixel entry, the f64 binned entry, the f32 fused per-pixel entry and the f32 fused binned entry.
    /// The symmetric 50/50 step split these entries used before misses one-sided repairs by a large fraction of
    /// the effect, so this test fails on that code.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn blockiness_map_matches_finite_repairs_through_every_pipeline() {
        // Exact-bit comparison across pipelines: hold the token lock so a permutation test in this binary
        // cannot switch the SIMD tier between the two sides.
        let _tokens = archmage::testing::lock_token_testing();
        use crate::feature_v2::idx;
        const LAYOUT: usize = 29; // FEATURES_PER_CHANNEL_V2_TOTAL
        let (w, h) = (256usize, 192usize);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        // REV4SERVE: at Rev4 the V2Bounded walk refuses (its v1 moments are a
        // different summation tree than the canonical fold — named refusal in
        // `feature_v2`); the folded append2 entry emits the same v2 block
        // canonically, offset by the v1 prefix.
        let rev4 =
            crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev4;
        let v2_off = if rev4 { BLOCK_END_V1_POOLS } else { 0 };
        let v2_of = |z: &Zensim, s: &RgbSlice, d: &RgbSlice| {
            if rev4 {
                z.compute_folded720_append2_features(s, d).unwrap()
            } else {
                z.compute_v2_features(s, d).unwrap()
            }
        };
        let base = v2_of(&z, &rs, &ds);
        let pre = z.precompute_reference(&rs).unwrap();
        let rects = [
            (32usize, 32usize, 96usize, 128usize),
            (0, 0, 64, 64),
            (128, 96, 256, 192),
            (64, 0, 192, 32),
        ];
        let mut checked = 0usize;
        let mut max_obs = 0.0f64;
        for (x0, y0, x1, y1) in rects {
            let mut repaired = dst.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    repaired[y * w + x] = src[y * w + x];
                }
            }
            let rrs = RgbSlice::new(&repaired, w, h);
            let after = v2_of(&z, &rs, &rrs);
            for scale in 0..4 {
                for ch in 0..3 {
                    let local = scale * 3 * LAYOUT + ch * LAYOUT + idx::BLOCKINESS;
                    let k = BLOCK_END_V1_POOLS + local;
                    let observed =
                        -(after.features()[v2_off + local] - base.features()[v2_off + local]);
                    let mut s = vec![0.0f64; BLOCK_END_APPEND2];
                    s[k] = -1.0;
                    let per_pixel = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
                    let binned = z
                        .compute_attribution_density_full_binned(&rs, &ds, &s, 8)
                        .unwrap();
                    let mut sess = Fused944Session::new();
                    let fused = z
                        .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
                        .unwrap()
                        .2;
                    let mut sess = Fused944Session::new();
                    let fused_binned = z
                        .compute_folded944_score_and_attribution_binned(
                            &rs, &pre, &ds, &s, &mut sess, 8,
                        )
                        .unwrap()
                        .2;
                    for (name, result, tol) in [
                        ("f64 per-pixel", &per_pixel, 1e-9),
                        ("f64 binned", &binned, 1e-9),
                        ("f32 fused per-pixel", &fused, 2e-6),
                        ("f32 fused binned", &fused_binned, 2e-6),
                    ] {
                        let predicted = result.query_rect(x0, y0, x1, y1);
                        assert!(
                            (predicted - observed).abs() <= tol * observed.abs().max(1.0),
                            "{name}: scale {scale} ch {ch} rect {:?}: predicted {predicted} vs finite repair {observed}",
                            (x0, y0, x1, y1)
                        );
                    }
                    checked += 1;
                    max_obs = max_obs.max(observed.abs());
                }
            }
        }
        assert_eq!(checked, 48);
        assert!(
            max_obs > 1e-4,
            "fixture has no BLOCKINESS effect to test (max {max_obs})"
        );
    }

    /// REV4SERVE gate: the finite-repair exactness contract must also hold
    /// in a Rev4 process, where the v2 features come from the canonical
    /// folded entry (the V2Bounded walk refuses at Rev4 by name) and every
    /// attribution pipeline entry serves the canonical arithmetic.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn blockiness_finite_repair_exactness_serves_at_rev4() {
        // Exact-bit comparison across pipelines: hold the token lock so a permutation test in this binary
        // cannot switch the SIMD tier between the two sides.
        let _tokens = archmage::testing::lock_token_testing();
        crate::ssim_form::rerun_tests_at_revision("4", "blockiness_map_", 2);
    }

    /// ANTI-RECURRENCE GUARD (campaign appendix E.2): the full-coverage
    /// density must cover exactly the intended slot set at every supported
    /// gradient width. The append2 block (`f924-943`) was silently dropped
    /// for the whole 944 era by a hard-coded `min(len, 924)` bound with no
    /// test standing between the regime bump and the loss — this is that
    /// test. Adding a block means adding a `BLOCK_END_*` constant AND a row
    /// here.
    ///
    /// Each probe sets ONE slot's gradient and asserts the density is
    /// non-zero (block reached, slot decomposable) or identically zero
    /// (block not reached at this width, or the slot is registered class N).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn attribution_covers_expected_slots_per_width() {
        use crate::feature_v2::{APPEND2_PER_SCALE, idx_append2};
        let (w, h) = (96usize, 80usize);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);

        // |density| mass for a unit gradient on exactly one slot.
        let mass_at = |width: usize, k: usize| -> f64 {
            let mut s = vec![0.0f64; width];
            s[k] = -1.0;
            let d = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
            d.density().iter().map(|v| v.abs() as f64).sum()
        };
        // Same probe through the FUSED folded-944 entry (G-N3, appendix N):
        // the fused map must cover exactly the same slot set.
        let pre = z.precompute_reference(&rs).unwrap();
        let mut sess = Fused944Session::new();
        let mut mass_at_fused = |width: usize, k: usize| -> f64 {
            let mut s = vec![0.0f64; width];
            s[k] = -1.0;
            let (_, _, d) = z
                .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
                .unwrap();
            d.density().iter().map(|v| v.abs() as f64).sum()
        };

        // Representative decomposable slot per block, and the registered
        // class-N probes. (scale 0, ch Y where the block has a channel axis.)
        let basic_y = 13; // scale 0, ch Y, ssim mean
        let v1_pool = BLOCK_END_BASIC + 3; // f156-371: never spatialized
        let v2_y = BLOCK_END_V1_POOLS + 29 + 3; // v2 ART, scale 0, ch Y
        let append_y = BLOCK_END_V2 + 17 + 9; // append TEXTURE_DISSIM, scale 0, Y
        // append2 is Y-only with no channel axis; BANDVIS is a COARSE-scale
        // detector, so probe the coarsest scale (scale 3) where it fires.
        let app2 =
            |scale: usize, local: usize| BLOCK_END_APPEND + scale * APPEND2_PER_SCALE + local;

        // (width, slot, must_be_nonzero, why)
        let cases: &[(usize, usize, bool, &str)] = &[
            // 372: basic reached; v1 pools never; nothing above exists.
            (372, basic_y, true, "372: basic block"),
            (372, v1_pool, false, "372: f156-371 is never spatialized"),
            // 720: v2 reached, append does not exist at this width.
            (720, basic_y, true, "720: basic block"),
            (720, v1_pool, false, "720: f156-371 is never spatialized"),
            (720, v2_y, true, "720: v2 block"),
            // 924: append reached, append2 does not exist at this width.
            (924, v2_y, true, "924: v2 block"),
            (924, append_y, true, "924: append block"),
            // 944: append2 reached — the block the old slice dropped.
            (944, append_y, true, "944: append block still reached"),
            (
                944,
                app2(3, idx_append2::BANDVIS_GAIN),
                true,
                "944: append2 BANDVIS_GAIN is class E and MUST be covered",
            ),
            (
                944,
                app2(3, idx_append2::BANDVIS_LOSS),
                true,
                "944: append2 BANDVIS_LOSS is class E and MUST be covered",
            ),
            (
                944,
                app2(3, idx_append2::LUMA_MEAN_REF),
                false,
                "944: LUMA_MEAN_REF is reference-only — class N by definition",
            ),
            (
                944,
                app2(3, idx_append2::HL_BIN1),
                false,
                "944: HL_BIN1 is HDR-gated — structural zero on the SDR route",
            ),
            (
                944,
                app2(3, idx_append2::HL_BIN2),
                false,
                "944: HL_BIN2 is HDR-gated — structural zero on the SDR route",
            ),
            // CLASS-C free slots (2026-09-04): the bounded-error tranche
            // adds NO new slot numbers, so there is no new block for this
            // test to reach — but the 24 positions it now FILLS from the
            // cheap walk have to be decomposable, or a steering loop that
            // reads them gets a silently-empty map. Probed explicitly here
            // rather than assumed from "they were always in the layout".
            (
                720,
                BLOCK_END_V1_POOLS + 29 + crate::feature_v2::idx::MSE,
                true,
                "720: v2 MSE (class-C free slot) must be spatialized",
            ),
            (
                944,
                BLOCK_END_V2 + 17 + crate::feature_v2::idx_append::LUM_DARK_ERR,
                true,
                "944: append LUM_DARK_ERR (class-C free slot) must be spatialized",
            ),
            (
                944,
                BLOCK_END_V2 + 17 + crate::feature_v2::idx_append::LUM_MID_ERR,
                true,
                "944: append LUM_MID_ERR (class-C free slot) must be spatialized",
            ),
            (
                944,
                BLOCK_END_V2 + 17 + crate::feature_v2::idx_append::LUM_BRIGHT_ERR,
                true,
                "944: append LUM_BRIGHT_ERR (class-C free slot) must be spatialized",
            ),
        ];
        for &(width, k, want_nonzero, why) in cases {
            let m = mass_at(width, k);
            let m_f = mass_at_fused(width, k);
            if want_nonzero {
                assert!(m > 0.0, "{why}: expected non-zero density, got |mass| {m}");
                assert!(
                    m_f > 0.0,
                    "{why} (FUSED): expected non-zero density, got |mass| {m_f}"
                );
            } else {
                assert_eq!(
                    m, 0.0,
                    "{why}: expected identically-zero density, got |mass| {m}"
                );
                assert_eq!(
                    m_f, 0.0,
                    "{why} (FUSED): expected identically-zero density, got |mass| {m_f}"
                );
            }
        }

        // Beyond append2 (the f944+ CSFW block) there is no integrand yet —
        // deliberately not sliced. Pinned so a CSFW regime bump has to come
        // here and decide, rather than silently inheriting zero.
        let m = mass_at(BLOCK_END_APPEND2 + 12, BLOCK_END_APPEND2 + 3);
        assert_eq!(
            m, 0.0,
            "f944+ (CSFW) has no attribution integrand yet — must be exactly 0, got {m}"
        );
        let m_f = mass_at_fused(BLOCK_END_APPEND2 + 12, BLOCK_END_APPEND2 + 3);
        assert_eq!(
            m_f, 0.0,
            "f944+ (CSFW) has no FUSED attribution integrand yet — must be exactly 0, got {m_f}"
        );
    }

    /// C3a golden gate 1: the fused compare's SCORE is bit-identical to the
    /// fold-diffmap path's score (same pipeline, same stats, same
    /// apply_mlp_scoring) on a bake profile.
    #[test]
    fn fused_score_bit_matches_diffmap_path() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::profile::historical_b_for_test());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let fold = z
            .compute_with_ref_and_diffmap(&pre, &ds, crate::DiffmapWeighting::Trained)
            .unwrap();
        let s = vec![-1.0f64; 156];
        let (res, attr) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        assert_eq!(
            res.score().to_bits(),
            fold.score().to_bits(),
            "fused score {} != fold-path score {}",
            res.score(),
            fold.score()
        );
        assert_eq!(attr.width(), w);
        assert_eq!(attr.height(), h);
        assert!(attr.density().iter().all(|v| v.is_finite()));
    }

    /// C3a golden gate 2: the fused attribution equals the standalone f64
    /// density to f32-combine precision (identical retained planes by the
    /// C1 banding-parity construction; the deltas are the f32 kernel, the
    /// f32 canvas, and stats-derived vs sum-derived coefficients).
    #[test]
    fn fused_matches_standalone_attribution() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::profile::historical_b_for_test());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 156];
        for (k, v) in s.iter_mut().enumerate() {
            // Mixed-sign, all-slot gradient exercising every integrand.
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let std_attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
        let (_, fused_attr) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        let max_abs = std_attr
            .density()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs > 0.0);
        for (i, (a, b)) in fused_attr
            .density()
            .iter()
            .zip(std_attr.density().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 3e-5 * max_abs + 1e-9,
                "pixel {i}: fused {a} vs standalone {b} (max_abs {max_abs})"
            );
        }
        // Block sums through the SAT agree at the same class.
        let bs_f = fused_attr.block_sums(16);
        let bs_s = std_attr.block_sums(16);
        let bmax = bs_s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in bs_f.iter().zip(bs_s.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * bmax.max(1e-12),
                "block {i}: fused {a} vs standalone {b}"
            );
        }
    }

    /// Appendix N G-N1: the fused folded-944 entry's features are BITWISE
    /// the canonical 944 extraction's (the retention hooks only copy —
    /// accumulation is untouched), so the standalone 944 score path
    /// (`score_features_with_profile` over the features) is bit-identical
    /// too — asserted through a shipped profile's forward.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_features_bitwise_and_score_match_standalone() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let std_v2 = z.compute_folded720_append2_features(&rs, &ds).unwrap();
        let mut s = vec![0.0f64; 944];
        for (k, v) in s.iter_mut().enumerate() {
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let mut sess = Fused944Session::new();
        let (_res, fused_v2, attr) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        assert_eq!(fused_v2.features().len(), 944);
        assert_eq!(std_v2.features().len(), 944);
        for (k, (a, b)) in fused_v2
            .features()
            .iter()
            .zip(std_v2.features().iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "feature {k}: fused {a} vs standalone {b}"
            );
        }
        // The forward over the features (the loop's score step) — the
        // caller-width prefix branch accepts the 944 vector.
        let sf = crate::score_features_with_profile(
            crate::profile::historical_b_for_test(),
            fused_v2.features(),
            w as u32,
            h as u32,
        )
        .unwrap();
        let ss = crate::score_features_with_profile(
            crate::profile::historical_b_for_test(),
            std_v2.features(),
            w as u32,
            h as u32,
        )
        .unwrap();
        assert_eq!(
            sf.to_bits(),
            ss.to_bits(),
            "forward score: fused {sf} vs standalone {ss}"
        );
        assert_eq!((attr.width(), attr.height()), (w, h));
        assert!(attr.density().iter().all(|v| v.is_finite()));
    }

    /// Appendix N G-N2: the fused folded-944 density equals the standalone
    /// `compute_attribution_density_full` within the C3a tolerance class —
    /// the deltas are the f32 basic combine (C3a-gated) and the coefficient
    /// inputs (exact walk accumulators vs the standalone's 1e-9-parity
    /// pass-A replication); the retained planes are bitwise-equal.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_density_matches_standalone_full() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 944];
        for (k, v) in s.iter_mut().enumerate() {
            // Mixed-sign, all-slot gradient exercising every integrand in
            // every block (the C3a gate's pattern, extended to 944).
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let std_attr = z.compute_attribution_density_full(&rs, &ds, &s).unwrap();
        let mut sess = Fused944Session::new();
        let (_res, _v2, fused_attr) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        let max_abs = std_attr
            .density()
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(max_abs > 0.0);
        for (i, (a, b)) in fused_attr
            .density()
            .iter()
            .zip(std_attr.density().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 3e-5 * max_abs + 1e-9,
                "pixel {i}: fused {a} vs standalone {b} (max_abs {max_abs})"
            );
        }
        let bs_f = fused_attr.block_sums(16);
        let bs_s = std_attr.block_sums(16);
        let bmax = bs_s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (a, b)) in bs_f.iter().zip(bs_s.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * bmax.max(1e-12),
                "block {i}: fused {a} vs standalone {b}"
            );
        }
    }

    /// Fused-944 session reuse is pure buffer reuse: a second call with the
    /// same inputs through the SAME session reproduces features and density
    /// bitwise (no cross-compare numeric state, unlike AttributionSession).
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn fused944_session_reuse_is_deterministic() {
        let (w, h) = (150, 96);
        let (src, dst) = test_pair(w, h);
        let z = test_zensim();
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 944];
        for (k, v) in s.iter_mut().enumerate() {
            *v = -0.5 - (k % 5) as f64 * 0.1;
        }
        let mut sess = Fused944Session::new();
        let (_r1, v1, a1) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        let (_r2, v2, a2) = z
            .compute_folded944_score_and_attribution(&rs, &pre, &ds, &s, &mut sess)
            .unwrap();
        for (a, b) in v1.features().iter().zip(v2.features().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "session reuse changed features");
        }
        for (a, b) in a1.density().iter().zip(a2.density().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "session reuse changed density");
        }
    }

    /// The density must be finite everywhere on adversarial flat/extreme
    /// content (guards the pooled-scalar divisions).
    #[test]
    fn no_nan_on_flat_and_extreme_pairs() {
        let z = test_zensim();
        let s = vec![-0.5f64; 156];
        #[allow(clippy::type_complexity)]
        let cases: Vec<(Vec<[u8; 3]>, Vec<[u8; 3]>)> = vec![
            (vec![[0, 0, 0]; 64 * 64], vec![[255, 255, 255]; 64 * 64]),
            (
                vec![[128, 128, 128]; 64 * 64],
                vec![[128, 128, 128]; 64 * 64],
            ),
            (
                vec![[255, 255, 255]; 64 * 64],
                vec![[254, 255, 255]; 64 * 64],
            ),
        ];
        for (src, dst) in &cases {
            let rs = RgbSlice::new(src, 64, 64);
            let ds = RgbSlice::new(dst, 64, 64);
            let attr = z.compute_attribution_density(&rs, &ds, &s).unwrap();
            assert!(
                attr.density().iter().all(|v| v.is_finite()),
                "non-finite density value"
            );
        }
    }

    /// #70 stale-scalar single-pass, exactness gate 1: with the SAME pair
    /// presented twice, the second (stale single-pass) call's map equals
    /// the fresh fused map of the previous input pair BITWISE — the
    /// in-strip fold with matching coefficient packs reproduces the
    /// retained-plane combine exactly (per-pixel elementwise kernel,
    /// identical spread/merge/upsample tail). Scores stay bit-identical on
    /// every call.
    #[test]
    fn stale_same_pair_second_call_equals_fresh_map_exactly() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::profile::historical_b_for_test());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let mut s = vec![0.0f64; 156];
        for (k, v) in s.iter_mut().enumerate() {
            *v = if k % 3 == 0 { -1.0 } else { -0.25 } * (1.0 + (k % 7) as f64 * 0.1);
        }
        let (res_fresh, attr_fresh) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s)
            .unwrap();
        let mut sess = AttributionSession::new();
        let (res1, attr1) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sess)
            .unwrap();
        // The priming call IS the fresh path.
        assert_eq!(res1.score().to_bits(), res_fresh.score().to_bits());
        for (a, b) in attr1.density().iter().zip(attr_fresh.density().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "priming map != fresh map");
        }
        // Second call: single-pass with the previous (same-pair) scalars —
        // must equal the fresh map of the previous input pair EXACTLY.
        let (res2, attr2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sess)
            .unwrap();
        assert_eq!(
            res2.score().to_bits(),
            res_fresh.score().to_bits(),
            "stale-path score must stay bit-identical"
        );
        for (i, (a, b)) in attr2
            .density()
            .iter()
            .zip(attr_fresh.density().iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "pixel {i}: stale map {a} != fresh-previous-pair map {b}"
            );
        }
    }

    /// #70 exactness gate 2 (the stale semantics on DISTINCT pairs): after
    /// priming on pair A, the call on pair B returns score(B) bit-identical
    /// and a map equal — bitwise — to the reference construction "pair B's
    /// planes combined with pair A's coefficient packs" built through the
    /// retained-plane walk. Also proves staleness engages (differs from
    /// fresh(B)'s map).
    #[test]
    fn stale_call_on_new_pair_combines_current_planes_with_previous_scalars() {
        let (w, h) = (150, 170);
        let (src, dst_a) = test_pair(w, h);
        // A second, visibly different distortion for pair B.
        let dst_b: Vec<[u8; 3]> = dst_a
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let d = ((i % 13) as i16) - 6;
                [
                    (p[0] as i16 + d).clamp(0, 255) as u8,
                    (p[1] as i16 - d).clamp(0, 255) as u8,
                    (p[2] as i16 + d / 2).clamp(0, 255) as u8,
                ]
            })
            .collect();
        let z = crate::Zensim::new(crate::profile::historical_b_for_test());
        let rs = RgbSlice::new(&src, w, h);
        let dsa = RgbSlice::new(&dst_a, w, h);
        let dsb = RgbSlice::new(&dst_b, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let s = vec![-1.0f64; 156];

        let mut sess = AttributionSession::new();
        let _ = z
            .compute_with_ref_score_and_attribution_stale(&pre, &dsa, &s, &mut sess)
            .unwrap();
        let coeffs_a = sess.coeffs.clone();
        let (res_b, attr_ab) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &dsb, &s, &mut sess)
            .unwrap();

        let (res_b_fresh, attr_b_fresh) = z
            .compute_with_ref_score_and_attribution(&pre, &dsb, &s)
            .unwrap();
        assert_eq!(res_b.score().to_bits(), res_b_fresh.score().to_bits());

        // Reference construction: planes of B × coefficient packs of A,
        // through the retained-plane walk + the identical tail.
        let params = z.profile().params();
        let config = config_from_params(params, z.parallel());
        let (comp_pw, comp_h) = (pre.scales[0].1, pre.scales[0].2);
        let mut canvas = vec![0.0f32; comp_pw * comp_h];
        let mut id_plane = vec![0.0f32; comp_pw * comp_h];
        let mut win_plane = vec![0.0f32; comp_pw * comp_h];
        let mut spread_tmp: Vec<f32> = Vec::new();
        let mut spread_out: Vec<f32> = Vec::new();
        let on_scale = |scale: usize,
                        _stats: &crate::metric::ScaleStats,
                        src_planes: [&[f32]; 3],
                        dst_planes: [&[f32]; 3],
                        ret: &crate::streaming::AttrScaleRetention,
                        sw: usize,
                        sh: usize| {
            let n = sw * sh;
            id_plane[..n].fill(0.0);
            win_plane[..n].fill(0.0);
            let co32 = coeffs_a[scale];
            for c in 0..3 {
                fused_combine_plane_f32(
                    &ret.sd[c][..n],
                    &src_planes[c][..n],
                    &dst_planes[c][..n],
                    &ret.mu1[c][..n],
                    &ret.mu2[c][..n],
                    co32[c],
                    &mut id_plane[..n],
                    &mut win_plane[..n],
                    false,
                );
            }
            crate::blur::box_spread_merge_f32(
                &mut win_plane[..n],
                &mut id_plane[..n],
                sw,
                sh,
                config.blur_radius,
                &mut spread_tmp,
                &mut spread_out,
                config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
            );
            upsample_add_sum_preserving_f32(
                &id_plane[..n],
                sw,
                sh,
                &mut canvas,
                comp_pw,
                comp_h,
                1usize << scale,
            );
        };
        let _ = crate::streaming::compute_zensim_streaming_with_ref_and_attr_planes(
            &pre,
            &dsb,
            &config,
            params.weights,
            on_scale,
        );
        let expected: Vec<f32> = if comp_pw == w && comp_h == h {
            canvas
        } else {
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h.min(comp_h) {
                out.extend_from_slice(&canvas[y * comp_pw..y * comp_pw + w.min(comp_pw)]);
            }
            out
        };
        for (i, (a, b)) in attr_ab.density().iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "pixel {i}: stale map {a} != planes(B)×coeffs(A) reference {b}"
            );
        }
        // Staleness engages: the returned map differs from fresh(B).
        assert!(
            attr_ab
                .density()
                .iter()
                .zip(attr_b_fresh.density().iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "stale map unexpectedly identical to fresh(B) — staleness did not engage"
        );
    }

    /// #70: a gradient change re-primes (fresh path for the new gradient)
    /// instead of silently mixing semantics.
    #[test]
    fn stale_reprimes_on_gradient_change() {
        let (w, h) = (96, 80);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::profile::historical_b_for_test());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let s1 = vec![-1.0f64; 156];
        let mut s2 = vec![-0.5f64; 156];
        s2[13] = -2.0;
        let mut sess = AttributionSession::new();
        let _ = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s1, &mut sess)
            .unwrap();
        let (_, attr_s2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s2, &mut sess)
            .unwrap();
        let (_, attr_s2_fresh) = z
            .compute_with_ref_score_and_attribution(&pre, &ds, &s2)
            .unwrap();
        for (a, b) in attr_s2.density().iter().zip(attr_s2_fresh.density().iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "re-primed map != fresh map");
        }
        assert_eq!(sess.s_saved, s2);
    }

    /// #70: recycling spent results into the session is a pure allocation
    /// optimization — density AND SAT (guard-row/column re-zero logic in
    /// `build_sat_into`) must be bitwise-identical to the never-recycled
    /// arm on every subsequent call.
    #[test]
    fn stale_recycled_buffers_produce_identical_maps() {
        let (w, h) = (150, 170);
        let (src, dst) = test_pair(w, h);
        let z = crate::Zensim::new(crate::profile::historical_b_for_test());
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&rs).unwrap();
        let s = vec![-1.0f64; 156];
        let rects = [
            (0usize, 0usize, w, h),
            (0, 0, 17, 23),
            (5, 7, 64, 64),
            (31, 100, 150, 170),
        ];
        // Arm A: never recycles.
        let mut sa = AttributionSession::new();
        let _ = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sa)
            .unwrap();
        let (_, a2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sa)
            .unwrap();
        let (_, a3) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sa)
            .unwrap();
        // Arm B: recycles the spent result before every call.
        let mut sb = AttributionSession::new();
        let (_, b1) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sb)
            .unwrap();
        sb.recycle(b1);
        let (_, b2) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sb)
            .unwrap();
        for (i, (x, y)) in b2.density().iter().zip(a2.density().iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "px {i}: recycled density != fresh-alloc"
            );
        }
        for &(x0, y0, x1, y1) in &rects {
            assert_eq!(
                b2.query_rect(x0, y0, x1, y1).to_bits(),
                a2.query_rect(x0, y0, x1, y1).to_bits(),
                "SAT rect ({x0},{y0})..({x1},{y1}) differs after recycle"
            );
        }
        sb.recycle(b2);
        let (_, b3) = z
            .compute_with_ref_score_and_attribution_stale(&pre, &ds, &s, &mut sb)
            .unwrap();
        for (i, (x, y)) in b3.density().iter().zip(a3.density().iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "px {i}: 2nd recycled density != fresh-alloc"
            );
        }
        for &(x0, y0, x1, y1) in &rects {
            assert_eq!(
                b3.query_rect(x0, y0, x1, y1).to_bits(),
                a3.query_rect(x0, y0, x1, y1).to_bits(),
                "SAT rect ({x0},{y0})..({x1},{y1}) differs after 2nd recycle"
            );
        }
    }

    /// NEIGHSTEER env gate end-to-end: `ZENSIM_NEIGHBOUR_EXACT` must
    /// capture the local-refinement snapshot ONLY on exactly `"1"` — a
    /// child process owns each env value, so there is no process-global
    /// `set_var` race — and with the gate on, `refinement_gain` must equal
    /// the remaining frozen density plus the SUMMED TRUE v2 deltas at
    /// scales 1–3 for a real reference-repair intervention.
    #[test]
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    fn neighbour_exact_env_gate_and_gain() {
        const PATH: &str = "attribution::tests::neighbour_exact_env_gate_and_gain";
        if std::env::var("ZENSIM_NEIGHSTEER_GATE_CHILD").as_deref() != Ok("1") {
            let exe = std::env::current_exe().expect("test binary path");
            for val in ["0", "", "yes", "1"] {
                let out = std::process::Command::new(&exe)
                    .args([PATH, "--exact", "--nocapture", "--test-threads=1"])
                    .env("ZENSIM_NEIGHBOUR_EXACT", val)
                    .env("ZENSIM_NEIGHSTEER_GATE_CHILD", "1")
                    .output()
                    .expect("re-exec the test binary");
                let stdout = String::from_utf8_lossy(&out.stdout);
                assert!(
                    out.status.success(),
                    "{PATH} failed for ZENSIM_NEIGHBOUR_EXACT={val:?}\n{stdout}\n{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let want = if val == "1" {
                    "NEIGHSTEER-GATE-ON"
                } else {
                    "NEIGHSTEER-GATE-OFF"
                };
                assert!(
                    stdout.contains(want),
                    "expected {want} for ZENSIM_NEIGHBOUR_EXACT={val:?}\n{stdout}"
                );
            }
            return;
        }
        let _token = archmage::testing::lock_token_testing();
        let on = std::env::var("ZENSIM_NEIGHBOUR_EXACT").as_deref() == Ok("1");
        let (w, h) = (137usize, 101usize);
        let (src, dst) = test_pair(w, h);
        // The v2basic-shaped linear bake the lane's panels use.
        let ids: Vec<usize> = (0..156).chain(372..720).collect();
        let weights: Vec<f64> = ids
            .iter()
            .map(|&id| -(0.002 + 0.0003 * ((id * 7) % 13) as f64))
            .collect();
        let recipe = serde_json::json!({
            "schema_hash":1,"scaler_mean":vec![0.0;ids.len()],"scaler_scale":vec![1.0;ids.len()],
            "metadata":[{"key":"zentrain.feature_ids","type":"utf8",
                "text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")}],
            "layers":[{"in_dim":ids.len(),"out_dim":1,"activation":"identity","dtype":"f32",
                "weights":weights,"biases":[100.0]}]
        });
        let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = crate::BakeScorer::new(&model).unwrap().with_parallel(false);
        let rs = RgbSlice::new(&src, w, h);
        let ds = RgbSlice::new(&dst, w, h);
        let mut worker = scorer.prepare_steering(&rs, 8).unwrap();
        let scored = worker.compute(&ds, None).unwrap();
        assert_eq!(
            scored.neighbour_exact.is_some(),
            on,
            "snapshot presence must track the exact-'1' gate"
        );
        println!(
            "{}",
            if on {
                "NEIGHSTEER-GATE-ON"
            } else {
                "NEIGHSTEER-GATE-OFF"
            }
        );
        let Some(snap) = scored.neighbour_exact.as_ref() else {
            return;
        };
        let s = scored.sensitivities();
        let base_f = scored.result().features();
        for &(x0, y0, x1, y1) in &[(8usize, 8usize, 16usize, 16usize), (3, 5, 26, 22)] {
            let mut intervened = dst.clone();
            for y in y0..y1 {
                for x in x0..x1 {
                    intervened[y * w + x] = src[y * w + x];
                }
            }
            let full = scorer
                .compute(&rs, &RgbSlice::new(&intervened, w, h), None)
                .unwrap();
            let deltas = snap
                .deltas((x0, y0, x1, y1), &crate::local_refine::Candidate::Reference)
                .unwrap();
            let sum_true: f64 = deltas
                .iter()
                .map(|(id, _)| s[*id] * (full.features()[*id] - base_f[*id]))
                .sum();
            let sum_engine: f64 = deltas.iter().map(|(id, d)| s[*id] * d).sum();
            assert!(
                (sum_engine - sum_true).abs() <= 1e-4 * (1.0 + sum_true.abs()),
                "rect ({x0},{y0},{x1},{y1}): Σs·Δ engine {sum_engine:e} vs true {sum_true:e}"
            );
            // refinement_gain = frozen density (the v2 coarse mass was
            // zeroed at capture) + the engine's exact deltas, i.e.
            // density + Σs·Δ_true.
            let mut density = scored.attribution().query_rect(x0, y0, x1, y1);
            for m in &scored.max_removals {
                density -= m.sensitivity * m.feature_drop(x0, y0, x1, y1);
            }
            for m in &scored.moment_removals {
                density += m.correction(x0, y0, x1, y1);
            }
            let gain = scored.refinement_gain(x0, y0, x1, y1);
            assert!(
                (gain - (density + sum_true)).abs() <= 1e-4 * (1.0 + sum_true.abs()),
                "rect ({x0},{y0},{x1},{y1}): refinement_gain {gain:e} vs \
                 density {density:e} + Σs·Δ_true {sum_true:e}"
            );
        }
    }
}

// ============================================================================
// C3a: the FUSED compare — scalar score + attribution map from ONE pipeline
// ============================================================================

/// Coefficient pack for the f32 fused combine, ordered
/// `[sd, sd4, sd2, art, art4, art2, det, det4, det2, mse, hfe, hfm]`.
fn coeffs_to_f32(co: &SlotCoeffs) -> [f32; 12] {
    [
        co.c_sd as f32,
        co.c_sd4 as f32,
        co.c_sd2 as f32,
        co.c_art as f32,
        co.c_art4 as f32,
        co.c_art2 as f32,
        co.c_det as f32,
        co.c_det4 as f32,
        co.c_det2 as f32,
        co.c_mse as f32,
        co.c_hfe as f32,
        co.c_hfm as f32,
    ]
}

/// f32 fused-combine kernel (C3a perf lever): the same integrand formulas as
/// [`basic_combine_channel`], computed in f32 and auto-vectorized per ISA.
/// Precision class: the density-sum identities move from the f64 path's
/// 1e-9/1e-6 to ~1e-5 relative (measured; the standalone f64 path and its
/// strict tests are unchanged — this kernel serves the fused entry only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_combine_plane_f32(
    sd: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: [f32; 12],
    id_plane: &mut [f32],
    win_plane: &mut [f32],
    stable7: bool,
) {
    combine_basic_and_l8::<false>(
        sd, src, dst, mu1, mu2, co, [0.0; 3], id_plane, win_plane, stable7,
    );
}

#[autoversion]
#[allow(clippy::too_many_arguments)]
fn combine_basic_and_l8<const L8: bool>(
    sd: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: [f32; 12],
    l8: [f64; 3],
    id_plane: &mut [f32],
    win_plane: &mut [f32],
    // Revision 7: the v1 edge ratio as `(x − y)/(1 + y)`.
    stable7: bool,
) {
    let n = id_plane.len();
    let sd = &sd[..n];
    let src = &src[..n];
    let dst = &dst[..n];
    let mu1 = &mu1[..n];
    let mu2 = &mu2[..n];
    let win_plane = &mut win_plane[..n];
    for i in 0..n {
        let sdv = sd[i];
        let sv = src[i];
        let dv = dst[i];
        let m1 = mu1[i];
        let m2 = mu2[i];
        let d1 = (sv - m1).abs();
        let d2 = (dv - m2).abs();
        let ed = if stable7 {
            (d2 - d1) / (1.0 + d1)
        } else {
            (1.0 + d2) / (1.0 + d1) - 1.0
        };
        let art = ed.max(0.0);
        let det = (-ed).max(0.0);
        let sd2 = sdv * sdv;
        let a2 = art * art;
        let dt2 = det * det;
        let pd = sv - dv;
        let e_hf = d1 * d1 - d2 * d2;
        let win_term = co[0] * sdv + co[1] * (sd2 * sd2) + co[2] * sd2;
        let res_term = co[3] * art
            + co[4] * (a2 * a2)
            + co[5] * a2
            + co[6] * det
            + co[7] * (dt2 * dt2)
            + co[8] * dt2
            + co[10] * e_hf
            + co[11] * (d1 - d2);
        let px_term = co[9] * (pd * pd);
        id_plane[i] += px_term + res_term;
        win_plane[i] += win_term;
        if L8 {
            // Preserve both old accumulation/rounding points. Only the loads,
            // division and powers are shared; L8 coefficients remain f64.
            let sd4 = sd2 * sd2;
            let a4 = a2 * a2;
            let d4 = dt2 * dt2;
            win_plane[i] += (l8[0] * f64::from(sd4 * sd4)) as f32;
            id_plane[i] += (l8[1] * f64::from(a4 * a4) + l8[2] * f64::from(d4 * d4)) as f32;
        }
    }
}

/// Coefficients for the three L8 slots in each six-slot peak cell. Despite
/// their historical `p95` names, these stats are eighth-root means, not
/// quantiles. Retain f64 coefficients: a representable f32 eighth moment can
/// require a coefficient above f32::MAX, while its map contribution is tiny.
fn l8_coefficients(
    s: &[f64],
    scale: usize,
    channel: usize,
    stats: &crate::metric::ScaleStats,
    n: f64,
    split: crate::map_paint::SplitMode,
) -> [f64; 3] {
    // Removal `−s·x⁸/(8·N·L8⁷)`; the exact share is eight times it. The
    // amplitude split carries these slots in the linear terms instead
    // ([`l8_amplitude_fold`]).
    let share = match split {
        crate::map_paint::SplitMode::Exact => 1.0,
        crate::map_paint::SplitMode::Repair => 1.0 / 8.0,
        crate::map_paint::SplitMode::Amplitude => return [0.0; 3],
    };
    let roots = [
        stats.ssim_p95[channel],
        stats.art_p95[channel],
        stats.det_p95[channel],
    ];
    core::array::from_fn(|slot| {
        let sensitivity = s
            .get((scale * 3 + channel) * 6 + 3 + slot)
            .copied()
            .unwrap_or(0.0);
        if sensitivity == 0.0 || roots[slot] <= 0.0 {
            0.0
        } else {
            -sensitivity * share / (n * roots[slot].powi(7))
        }
    })
}

/// `SplitMode::Amplitude`: the three L8 slots of a peak cell as mean pools of
/// their members (`sd`, `art`, `det`), `−s/N` each, to add to the linear
/// coefficients of the basic combine (`[sd, art, det]`).
fn l8_amplitude_fold(s: &[f64], scale: usize, channel: usize, n: f64) -> [f64; 3] {
    core::array::from_fn(|slot| {
        -s.get((scale * 3 + channel) * 6 + 3 + slot)
            .copied()
            .unwrap_or(0.0)
            / n
    })
}

/// Logical source-coordinate bounds owned by each sample along a pyramid
/// axis. This is a frozen-signal ownership model, not the blur dependency
/// graph. Reflection precedes zero padding; padding belongs to the nearest
/// extended-image edge. Halving drops odd trailing samples, as the producer
/// does. Explicit bounds also handle noncontiguous reflected footprints.
fn max_axis_footprints(logical: usize, scale_len: usize, scale: usize) -> Vec<(usize, usize)> {
    let extended = logical.max(MIN_PYRAMID_DIM);
    let factor = 1usize << scale;
    (0..scale_len)
        .map(|i| {
            let mut lo = logical;
            let mut hi = 0;
            for j in i * factor..(i + 1) * factor {
                let source = crate::metric::reflect_index(j.min(extended - 1), logical);
                lo = lo.min(source);
                hi = hi.max(source);
            }
            (lo, hi)
        })
        .collect()
}

/// Non-additive max removal, retained in logical source coordinates. A signal
/// survives a rectangle if ANY owned source coordinate is outside it. The
/// four one-dimensional projections answer that union exactly, including
/// ties; no full signal plane is retained.
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
pub(crate) struct MaxRemoval {
    feature_id: usize,
    sensitivity: f64,
    left: Vec<f32>,
    right: Vec<f32>,
    top: Vec<f32>,
    bottom: Vec<f32>,
}

/// Opt-in finite-root correction. The integral contains only base-image mass.
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
pub(crate) struct MomentRemoval {
    sensitivity: f64,
    root: f64,
    power: u32,
    total: f64,
    mass: AttributionResult,
}

#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
fn finite_root_extra(fraction: f64, power: u32) -> f64 {
    let r = fraction.clamp(0.0, 1.0);
    // Factor 1-u^p into (1-u)(1+u)... for p=2,4,8. This avoids
    // cancellation in 1-(1-r)^(1/p) and needs only square roots.
    let mut u = 1.0 - r;
    let mut denominator = 1.0;
    for _ in 0..power.trailing_zeros() {
        u = u.sqrt();
        denominator *= 1.0 + u;
    }
    r / denominator - r / f64::from(power)
}

#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
impl MomentRemoval {
    fn correction(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let fraction = self.mass.query_rect(x0, y0, x1, y1) / self.total;
        -self.sensitivity * self.root * finite_root_extra(fraction, self.power)
    }
}

#[autoversion]
fn moment_signal_plane(
    sd: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    signal: usize,
    power: u32,
    output: &mut [f32],
) {
    for i in 0..output.len() {
        let value = if signal == 0 {
            sd[i]
        } else {
            let ed = (1.0 + (dst[i] - mu2[i]).abs()) / (1.0 + (src[i] - mu1[i]).abs()) - 1.0;
            if signal == 1 {
                ed.max(0.0)
            } else {
                (-ed).max(0.0)
            }
        };
        let p2 = value * value;
        output[i] = if power == 2 {
            p2
        } else {
            let p4 = p2 * p2;
            if power == 4 { p4 } else { p4 * p4 }
        };
    }
}

fn retain_moment_removals(
    output: &mut Vec<MomentRemoval>,
    s: &[f64],
    peaks: &[f64],
    scale: usize,
    stats: &crate::metric::ScaleStats,
    width: usize,
    height: usize,
    sw: usize,
    sh: usize,
    src: [&[f32]; 3],
    dst: [&[f32]; 3],
    ret: &BasicPlaneView<'_>,
    sampling: Option<&crate::sampling::Geometry>,
    bin: usize,
    radius: usize,
) {
    let n = sw * sh;
    let mut plane = vec![0.0; n];
    let mut spread = vec![0.0; n];
    let mut tmp = Vec::new();
    let mut scratch = Vec::new();
    for c in 0..3 {
        let roots = [
            [stats.ssim_2nd[c], stats.ssim[c * 2 + 1], stats.ssim_p95[c]],
            [
                stats.edge_2nd[c * 2],
                stats.edge[c * 4 + 1],
                stats.art_p95[c],
            ],
            [
                stats.edge_2nd[c * 2 + 1],
                stats.edge[c * 4 + 3],
                stats.det_p95[c],
            ],
        ];
        for (signal, roots) in roots.into_iter().enumerate() {
            for (j, (power, root)) in [2_u32, 4, 8].into_iter().zip(roots).enumerate() {
                let sensitivity = if power == 8 {
                    peaks.get((scale * 3 + c) * 6 + 3 + signal)
                } else {
                    s.get((scale * 3 + c) * 13 + signal * 3 + if j == 0 { 2 } else { 1 })
                }
                .copied()
                .unwrap_or(0.0);
                if sensitivity == 0.0 || root == 0.0 {
                    continue;
                }
                moment_signal_plane(
                    &ret.sd[c][..n],
                    &src[c][..n],
                    &dst[c][..n],
                    &ret.mu1[c][..n],
                    &ret.mu2[c][..n],
                    signal,
                    power,
                    &mut plane,
                );
                let values = if signal == 0 {
                    spread.fill(0.0);
                    crate::blur::box_spread_merge_f32(
                        &mut plane,
                        &mut spread,
                        sw,
                        sh,
                        radius,
                        &mut tmp,
                        &mut scratch,
                        false,
                    );
                    &spread
                } else {
                    &plane
                };
                let mut accum = BinAccum::new(width, height, bin);
                if let Some(geometry) = sampling {
                    let projected = geometry.project(values, scale);
                    accum.add_scale_plane_f32(&projected, width, height, 1);
                } else {
                    accum.add_scale_plane_f32(values, sw, sh, 1 << scale);
                }
                output.push(MomentRemoval {
                    sensitivity,
                    root,
                    power,
                    total: n as f64 * root.powi(power as i32),
                    mass: accum.into_result(),
                });
            }
        }
    }
}

#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
impl MaxRemoval {
    fn new(feature_id: usize, sensitivity: f64, width: usize, height: usize) -> Self {
        Self {
            feature_id,
            sensitivity,
            left: vec![0.0; width + 1],
            right: vec![0.0; width + 1],
            top: vec![0.0; height + 1],
            bottom: vec![0.0; height + 1],
        }
    }

    // Bounds are inclusive source-coordinate extrema, not half-open ranges.
    // A single physical sample owns (x, x), (y, y).
    #[cfg(test)]
    fn add(&mut self, value: f32, x: (usize, usize), y: (usize, usize)) {
        self.left[x.0 + 1] = self.left[x.0 + 1].max(value);
        self.right[x.1] = self.right[x.1].max(value);
        self.top[y.0 + 1] = self.top[y.0 + 1].max(value);
        self.bottom[y.1] = self.bottom[y.1].max(value);
    }

    fn finish(&mut self) {
        for prefix in [&mut self.left, &mut self.top] {
            for i in 1..prefix.len() {
                prefix[i] = prefix[i].max(prefix[i - 1]);
            }
        }
        for suffix in [&mut self.right, &mut self.bottom] {
            for i in (0..suffix.len() - 1).rev() {
                suffix[i] = suffix[i].max(suffix[i + 1]);
            }
        }
    }

    fn global(&self) -> f64 {
        f64::from(self.left[self.left.len() - 1])
    }

    fn feature_drop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let x1 = x1.min(self.left.len() - 1);
        let y1 = y1.min(self.top.len() - 1);
        if x0 >= x1 || y0 >= y1 {
            return 0.0;
        }
        let outside = self.left[x0]
            .max(self.right[x1])
            .max(self.top[y0])
            .max(self.bottom[y1]);
        self.global() - f64::from(outside)
    }
}

/// Row/column maxima of the unchanged canonical peak signals. Contiguous
/// columns permit SIMD updates; only the final projections scatter through
/// possibly reflected or fractional source-coordinate footprints.
#[autoversion]
#[allow(clippy::too_many_arguments)]
fn project_max_row(
    sd: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    active: [bool; 3],
    cs: &mut [f32],
    ca: &mut [f32],
    cd: &mut [f32],
) -> [f32; 3] {
    let mut row = [0.0f32; 3];
    if active[0] {
        for i in 0..sd.len() {
            cs[i] = cs[i].max(sd[i]);
            row[0] = row[0].max(sd[i]);
        }
    }
    if active[1] || active[2] {
        for i in 0..sd.len() {
            let ed = (1.0 + (dst[i] - mu2[i]).abs()) / (1.0 + (src[i] - mu1[i]).abs()) - 1.0;
            if active[1] {
                let value = ed.max(0.0);
                ca[i] = ca[i].max(value);
                row[1] = row[1].max(value);
            }
            if active[2] {
                let value = (-ed).max(0.0);
                cd[i] = cd[i].max(value);
                row[2] = row[2].max(value);
            }
        }
    }
    row
}

fn retain_max_removals(
    output: &mut Vec<MaxRemoval>,
    s_peaks: &[f64],
    scale: usize,
    logical_width: usize,
    logical_height: usize,
    sw: usize,
    sh: usize,
    src: [&[f32]; 3],
    dst: [&[f32]; 3],
    ret: &BasicPlaneView<'_>,
    sampling: Option<&crate::sampling::Geometry>,
) {
    #[cfg(test)]
    if tests::MAX_REFERENCE.with(std::cell::Cell::get) {
        tests::MAX_REFERENCE_CALLS.with(|count| count.set(count.get() + 1));
        retain_max_removals_reference(
            output,
            s_peaks,
            scale,
            logical_width,
            logical_height,
            sw,
            sh,
            src,
            dst,
            ret,
            sampling,
        );
        return;
    }
    #[cfg(test)]
    let first_map = output.len();
    if !(0..3).any(|c| {
        (0..3).any(|slot| {
            s_peaks
                .get((scale * 3 + c) * 6 + slot)
                .is_some_and(|s| *s != 0.0)
        })
    }) {
        return;
    }
    let xs = sampling.map_or_else(
        || max_axis_footprints(logical_width, sw, scale),
        |s| s.footprints(0, scale),
    );
    let ys = sampling.map_or_else(
        || max_axis_footprints(logical_height, sh, scale),
        |s| s.footprints(1, scale),
    );
    for c in 0..3 {
        let base = (scale * 3 + c) * 6;
        let mut maps: [Option<MaxRemoval>; 3] = core::array::from_fn(|slot| {
            let s = s_peaks.get(base + slot).copied().unwrap_or(0.0);
            (s != 0.0).then(|| MaxRemoval::new(156 + base + slot, s, logical_width, logical_height))
        });
        if maps.iter().all(Option::is_none) {
            continue;
        }
        // A column owns the same x footprint at every row, and a row
        // owns the same y footprint at every column. Max is separable:
        // reduce first, scatter O(sw + sh) times instead of O(sw * sh).
        let mut columns: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; sw]);
        let active = maps.each_ref().map(Option::is_some);
        for (y, &yf) in ys.iter().enumerate() {
            let range = y * sw..(y + 1) * sw;
            let [cs, ca, cd] = &mut columns;
            let row_max = project_max_row(
                &ret.sd[c][range.clone()],
                &src[c][range.clone()],
                &dst[c][range.clone()],
                &ret.mu1[c][range.clone()],
                &ret.mu2[c][range],
                active,
                cs,
                ca,
                cd,
            );
            for (map, value) in maps.iter_mut().zip(row_max) {
                if let Some(map) = map {
                    map.top[yf.0 + 1] = map.top[yf.0 + 1].max(value);
                    map.bottom[yf.1] = map.bottom[yf.1].max(value);
                }
            }
        }
        for (map, column) in maps.iter_mut().zip(&columns) {
            if let Some(map) = map {
                for (&xf, &value) in xs.iter().zip(column) {
                    map.left[xf.0 + 1] = map.left[xf.0 + 1].max(value);
                    map.right[xf.1] = map.right[xf.1].max(value);
                }
            }
        }
        for mut map in maps.into_iter().flatten() {
            map.finish();
            output.push(map);
        }
    }
    // Exercise the historical per-pixel implementation on the actual retained
    // planes in every serving test too, not only hand-made signal fixtures.
    #[cfg(test)]
    {
        let mut reference = Vec::new();
        retain_max_removals_reference(
            &mut reference,
            s_peaks,
            scale,
            logical_width,
            logical_height,
            sw,
            sh,
            src,
            dst,
            ret,
            sampling,
        );
        assert_eq!(output.len() - first_map, reference.len());
        for (actual, expected) in output[first_map..].iter().zip(&reference) {
            assert_eq!(actual.feature_id, expected.feature_id);
            assert_eq!(actual.sensitivity.to_bits(), expected.sensitivity.to_bits());
            for (a, b) in [
                (&actual.left, &expected.left),
                (&actual.right, &expected.right),
                (&actual.top, &expected.top),
                (&actual.bottom, &expected.bottom),
            ] {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "peak {} scale {scale}",
                        actual.feature_id
                    );
                }
            }
        }
    }
}

#[cfg(test)]
fn retain_max_removals_reference(
    output: &mut Vec<MaxRemoval>,
    s_peaks: &[f64],
    scale: usize,
    logical_width: usize,
    logical_height: usize,
    sw: usize,
    sh: usize,
    src: [&[f32]; 3],
    dst: [&[f32]; 3],
    ret: &BasicPlaneView<'_>,
    sampling: Option<&crate::sampling::Geometry>,
) {
    if !(0..3).any(|c| {
        (0..3).any(|slot| {
            s_peaks
                .get((scale * 3 + c) * 6 + slot)
                .is_some_and(|s| *s != 0.0)
        })
    }) {
        return;
    }
    let xs = sampling.map_or_else(
        || max_axis_footprints(logical_width, sw, scale),
        |s| s.footprints(0, scale),
    );
    let ys = sampling.map_or_else(
        || max_axis_footprints(logical_height, sh, scale),
        |s| s.footprints(1, scale),
    );
    for c in 0..3 {
        let base = (scale * 3 + c) * 6;
        let mut maps: [Option<MaxRemoval>; 3] = core::array::from_fn(|slot| {
            let s = s_peaks.get(base + slot).copied().unwrap_or(0.0);
            (s != 0.0).then(|| MaxRemoval::new(156 + base + slot, s, logical_width, logical_height))
        });
        if maps.iter().all(Option::is_none) {
            continue;
        }
        for (y, &yf) in ys.iter().enumerate() {
            for (x, &xf) in xs.iter().enumerate() {
                let i = y * sw + x;
                // Same f32 expression as the canonical fused accumulator.
                let ed = (1.0 + (dst[c][i] - ret.mu2[c][i]).abs())
                    / (1.0 + (src[c][i] - ret.mu1[c][i]).abs())
                    - 1.0;
                let signals = [ret.sd[c][i], ed.max(0.0), (-ed).max(0.0)];
                for (map, value) in maps.iter_mut().zip(signals) {
                    if let Some(map) = map {
                        map.add(value, xf, yf);
                    }
                }
            }
        }
        for mut map in maps.into_iter().flatten() {
            map.finish();
            output.push(map);
        }
    }
}

#[cfg(feature = "feature-regime-v2")]
pub(crate) fn bind_max_removals(
    maps: &mut Vec<MaxRemoval>,
    features: &[f64],
    density_missing: &[usize],
) -> Vec<usize> {
    // An extraction variant or numerical mismatch cannot silently gain
    // coverage. The old density remains usable with its old coverage report.
    maps.retain(|m| {
        features
            .get(m.feature_id)
            .is_some_and(|f| f.to_bits() == m.global().to_bits())
    });
    density_missing
        .iter()
        .copied()
        .filter(|id| !maps.iter().any(|m| m.feature_id == *id))
        .collect()
}

/// Independent pre-fusion L8 combine retained for numerical regression tests.
/// Evaluate the signal and
/// eighth powers in the SAME f32 order as the canonical fused accumulator;
/// multiply in f64 before writing finite f32 mass. Keeping this separate
/// preserves every legacy basic/stale combine operation and rounding point.
#[cfg(test)]
#[autoversion]
fn fused_combine_l8_f32(
    sd: &[f32],
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    co: [f64; 3],
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    for i in 0..id_plane.len() {
        let ed = (1.0 + (dst[i] - mu2[i]).abs()) / (1.0 + (src[i] - mu1[i]).abs()) - 1.0;
        let art = ed.max(0.0);
        let det = (-ed).max(0.0);
        let sd2 = sd[i] * sd[i];
        let a2 = art * art;
        let d2 = det * det;
        let sd4 = sd2 * sd2;
        let a4 = a2 * a2;
        let d4 = d2 * d2;
        win_plane[i] += (co[0] * f64::from(sd4 * sd4)) as f32;
        id_plane[i] += (co[1] * f64::from(a4 * a4) + co[2] * f64::from(d4 * d4)) as f32;
    }
}

/// `Σ (src−mu1)²` and `Σ |src−mu1|` over a plane — the two reference-side
/// pooled sums the hf coefficients need that `ScaleStats` does not carry.
#[autoversion]
fn hf_src_sums(src: &[f32], mu1: &[f32]) -> (f64, f64) {
    let mut sq = 0.0f64;
    let mut ab = 0.0f64;
    for i in 0..src.len().min(mu1.len()) {
        let r = src[i] - mu1[i];
        sq += (r * r) as f64;
        ab += r.abs() as f64;
    }
    (sq, ab)
}

/// f32 twin of [`upsample_add_sum_preserving`] for the fused canvas.
/// `pub(crate)`: also the fused folded-944 pass-B upsample (appendix P
/// lever 1, `feature_v2::attr_pass_b_for_scale_f32`).
pub(crate) fn upsample_add_sum_preserving_f32(
    scale_plane: &[f32],
    sw: usize,
    sh: usize,
    canvas: &mut [f32],
    cw: usize,
    ch: usize,
    factor: usize,
) {
    if factor == 1 {
        debug_assert_eq!(sw, cw);
        for (c, &v) in canvas.iter_mut().zip(scale_plane.iter()) {
            *c += v;
        }
        return;
    }
    // Destination-row-major replicate-add via the diffmap fusion's SIMD
    // kernel (#70: the former source-major footprint loop revisited each
    // canvas row `factor` times with scattered block writes — same op
    // count, far worse locality). Bitwise-identical: each canvas element
    // still receives exactly one add of the identically-formed product
    // `src · inv_area`.
    let inv_area = 1.0 / ((factor * factor) as f32);
    let copy_w = (sw * factor).min(cw);
    let dh = (sh * factor).min(ch);
    for dy in 0..dh {
        let sy = dy / factor;
        let src_row = &scale_plane[sy * sw..(sy + 1) * sw];
        let dst_row = &mut canvas[dy * cw..dy * cw + copy_w];
        crate::streaming::upsample_row_powx_add(src_row, dst_row, factor, inv_area);
    }
}

impl SlotCoeffs {
    /// Derive the combine coefficients from a finalized [`ScaleStats`]
    /// (the fused path — the pooled roots are inverted back to raw-moment
    /// means: `M4 = f4⁴`, `M2 = f2²`) plus the two reference-side hf sums
    /// the stats do not carry. Matches [`SlotCoeffs::derive`]'s math; the
    /// clamp-side gating for the hf ratio slots reads the stats' CLAMPED
    /// feature values directly (`f10 > 0` ⇔ loss side active, etc.).
    ///
    /// `split` ([`crate::map_paint::SplitMode`]): `Repair` keeps the
    /// removal-consistent `1/p` integrand of the p-pooled slots; `Exact` gives
    /// each pixel its Euler share `xᵖ·M^((1−p)/p)/N` (p times the removal
    /// form), so those slots sum to their feature, and splits a nonlinear HF
    /// gain by the secant, `f·e_i/(Σdst² − Σsrc²)`. Mean and HF-loss slots are
    /// linear and identical in both.
    #[allow(clippy::too_many_arguments)]
    fn from_scale_stats(
        s: &[f64],
        base_k: usize,
        stats: &crate::metric::ScaleStats,
        c: usize,
        n_f: f64,
        hf_sq_src_sum: f64,
        hf_abs_src_sum: f64,
        gain_form: crate::hf_gain_form::HfGainForm,
        split: crate::map_paint::SplitMode,
    ) -> Self {
        let exact = split == crate::map_paint::SplitMode::Exact;
        // `SplitMode::Amplitude`: every p-pool distributed as a mean pool of
        // its member (folded into the linear coefficients below).
        let amplitude = split == crate::map_paint::SplitMode::Amplitude;
        let g = |slot: usize| s.get(base_k + slot).copied().unwrap_or(0.0);
        let inv_n = 1.0 / n_f;
        let p_coeff = |sk: f64, m: f64, p: f64| -> f64 {
            if amplitude {
                0.0
            } else if m > 0.0 {
                let share = if exact { 1.0 } else { 1.0 / p };
                -sk * inv_n * share * m.powf((1.0 - p) / p)
            } else {
                0.0
            }
        };
        let m4_ssim = stats.ssim[c * 2 + 1].powi(4);
        let m2_ssim = stats.ssim_2nd[c].powi(2);
        let m4_art = stats.edge[c * 4 + 1].powi(4);
        let m2_art = stats.edge_2nd[c * 2].powi(2);
        let m4_det = stats.edge[c * 4 + 3].powi(4);
        let m2_det = stats.edge_2nd[c * 2 + 1].powi(2);
        let c_hfe = if hf_sq_src_sum > 1e-10 * n_f {
            if stats.hf_energy_loss[c] > 0.0 {
                -g(10) / hf_sq_src_sum
            } else if stats.hf_energy_gain[c] > 0.0 {
                if gain_form == crate::hf_gain_form::HfGainForm::RatioExcess {
                    // Preserve the established revision-1 spelling bit for bit
                    // (this gain is linear in Σdst², so both splits agree).
                    g(12) / hf_sq_src_sum
                } else if exact || amplitude {
                    g(12) * stats.hf_energy_gain[c] / (stats.hf_sq_dst_sum[c] - hf_sq_src_sum)
                } else {
                    g(12)
                        * crate::hf_gain_form::hf_energy_gain_d_sum_dst_sq(
                            gain_form,
                            hf_sq_src_sum,
                            stats.hf_sq_dst_sum[c],
                            inv_n,
                        )
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        let c_hfm = if hf_abs_src_sum > 1e-10 * n_f && stats.hf_mag_loss[c] > 0.0 {
            -g(11) / hf_abs_src_sum
        } else {
            0.0
        };
        let fold = |a: usize, b: usize| if amplitude { g(a) + g(b) } else { 0.0 };
        Self {
            c_sd: -(g(0) + fold(1, 2)) * inv_n,
            c_sd4: p_coeff(g(1), m4_ssim, 4.0),
            c_sd2: p_coeff(g(2), m2_ssim, 2.0),
            c_art: -(g(3) + fold(4, 5)) * inv_n,
            c_art4: p_coeff(g(4), m4_art, 4.0),
            c_art2: p_coeff(g(5), m2_art, 2.0),
            c_det: -(g(6) + fold(7, 8)) * inv_n,
            c_det4: p_coeff(g(7), m4_det, 4.0),
            c_det2: p_coeff(g(8), m2_det, 2.0),
            c_mse: -g(9) * inv_n,
            c_hfe,
            c_hfm,
        }
    }
}

/// Cross-compare state for the **stale-scalar single-pass** fused compare
/// (task #70, C3a ranked lever 3):
/// [`Zensim::compute_with_ref_score_and_attribution_stale`](crate::Zensim::compute_with_ref_score_and_attribution_stale).
///
/// Holds, from the previous compare against the same reference: the
/// per-(scale, channel) fused-combine coefficient packs (derived from that
/// compare's pooled scalars) and the reference-side hf sums (distortion-
/// independent, cached at the priming call). One session serves ONE
/// (reference, image-size, gradient) triple — a codec loop creates one per
/// encode. Reuse across a *different* reference of the same size is a
/// caller error the session cannot detect (the map would mix references);
/// call [`reset`](Self::reset) when the reference changes. Size or
/// gradient changes are detected and re-prime automatically.
#[derive(Default)]
pub struct AttributionSession {
    /// Per visited scale: per-channel combine coefficient packs from the
    /// previous compare's pooled scalars.
    coeffs: Vec<[[f32; 12]; 3]>,
    /// Per visited scale: per-channel reference-side hf sums
    /// `(Σ(src−μ1)², Σ|src−μ1|)` — constant for a fixed reference.
    hf_src: Vec<[(f64, f64); 3]>,
    /// The gradient + dims the session was primed with (identity guard).
    s_saved: Vec<f64>,
    width: usize,
    height: usize,
    /// Reusable per-call scratch (comp-plane sized) — kept across calls so
    /// the single-pass path pays no allocation / page-fault cost per
    /// iteration.
    canvas: Vec<f32>,
    id_plane: Vec<f32>,
    win_plane: Vec<f32>,
    spread_tmp: Vec<f32>,
    spread_out: Vec<f32>,
    /// Recycled result buffers (see [`recycle`](Self::recycle)); empty
    /// unless the caller opts in.
    density_scratch: Vec<f32>,
    sat_scratch: Vec<f64>,
}

impl AttributionSession {
    /// Empty (unprimed) session — the first compare through it runs the
    /// fresh fused path and primes the cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all cached state; the next compare re-primes (fresh path).
    /// Call when switching to a different reference image.
    pub fn reset(&mut self) {
        self.coeffs.clear();
        self.hf_src.clear();
        self.s_saved.clear();
        self.width = 0;
        self.height = 0;
    }

    /// Return a spent [`AttributionResult`] to the session so the next
    /// stale compare reuses its buffers (density plane + f64 SAT) instead
    /// of allocating and page-faulting fresh multi-MB ones every
    /// iteration — the codec-loop shape drops one result per compare.
    /// Purely an allocation optimization: opting out (never calling this)
    /// changes nothing but the per-call allocations, and the produced
    /// maps are bitwise-identical either way.
    pub fn recycle(&mut self, spent: AttributionResult) {
        self.density_scratch = spent.density;
        self.sat_scratch = spent.sat;
    }

    fn primed_for(&self, width: usize, height: usize, s: &[f64]) -> bool {
        !self.coeffs.is_empty() && self.width == width && self.height == height && self.s_saved == s
    }
}

/// The fused v1 walk core's output (appendix N refactor): the compare
/// result + the UNTRIMMED f32 basic-block attribution canvas at compute
/// dims, with the walk timings the ATTRPERF diagnostic reports.
struct FusedBasicCanvas {
    result: crate::metric::ZensimResult,
    canvas: Vec<f32>,
    comp_pw: usize,
    comp_h: usize,
    t_pipe_ms: f64,
    combine_ms: f64,
}

impl FusedBasicCanvas {
    /// Trim the compute-dims canvas to the logical image (top-left).
    fn trim_to(&self, width: usize, height: usize) -> Vec<f32> {
        if self.comp_pw == width && self.comp_h == height {
            self.canvas.clone()
        } else {
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height.min(self.comp_h) {
                out.extend_from_slice(
                    &self.canvas[y * self.comp_pw..y * self.comp_pw + width.min(self.comp_pw)],
                );
            }
            out
        }
    }
}

// Borrowed planes let Rev5 map consumers read the shared folded retention
// without allocating or copying a second set of basic moments.
struct BasicPlaneView<'a> {
    sd: [&'a [f32]; 3],
    mu1: [&'a [f32]; 3],
    mu2: [&'a [f32]; 3],
}
impl<'a> From<&'a crate::streaming::AttrScaleRetention> for BasicPlaneView<'a> {
    fn from(p: &'a crate::streaming::AttrScaleRetention) -> Self {
        Self {
            sd: p.sd.each_ref().map(Vec::as_slice),
            mu1: p.mu1.each_ref().map(Vec::as_slice),
            mu2: p.mu2.each_ref().map(Vec::as_slice),
        }
    }
}
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
enum BasicMapRetention<'a> {
    Basic(&'a BasicRetention),
    #[cfg(feature = "feature-regime-v2")]
    Fold(&'a crate::feature_v2::FoldRetention),
}
#[derive(Default)]
struct BasicMapScratch {
    id: Vec<f32>,
    win: Vec<f32>,
    spread_tmp: Vec<f32>,
    spread_out: Vec<f32>,
}

// Basic/peak signals retained by the scoring walk until model sensitivities
// are available. Reference planes remain in the caller's existing cache.
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
#[derive(Default)]
struct BasicRetention {
    scales: Vec<BasicRetainedScale>,
    result: Option<crate::ZensimResult>,
}
#[cfg_attr(not(feature = "feature-regime-v2"), allow(dead_code))]
struct BasicRetainedScale {
    stats: crate::metric::ScaleStats,
    distorted: [Vec<f32>; 3],
    planes: crate::streaming::AttrScaleRetention,
    width: usize,
    height: usize,
}

/// Reusable state for the fused folded-944 compare
/// ([`crate::Zensim::compute_folded944_score_and_attribution`]): the
/// streaming extraction's scratch + the walk retention (planes, pyramid
/// copies, exact accumulators). One session per encode loop; reuse across
/// compares avoids re-allocating ~42 MB (at 576²) of retained planes per
/// call. Unlike [`AttributionSession`] there is no cross-compare numeric
/// state — every retained value is rewritten by each call — so reuse
/// cannot change any output (gated by
/// `fused944_session_reuse_is_deterministic`).
#[cfg(feature = "feature-regime-v2")]
#[derive(Default)]
pub struct Fused944Session {
    basic: BasicRetention,
    basic_map: BasicMapScratch,
    // The temporary f64 accumulation canvas is not part of the returned map.
    // Keep its storage in the session; density and SAT remain independently owned.
    bin_scratch: Vec<f64>,
    scratch: crate::feature_v2::V2Scratch,
    retention: crate::feature_v2::FoldRetention,
    /// f32 pass-B scratch (appendix P lever 1) — plane-sized buffers
    /// reused across compares so per-iteration callers pay no
    /// allocation / page-fault cost.
    pass_b: crate::feature_v2::PassBScratchF32,
}

/// Test-only switch back to the pre-COSTSET3 route (folded walk with its own
/// v1 fold, then a second v1 walk for the basic map): the reference the
/// bit-identity gate compares against.
#[cfg(all(test, feature = "feature-regime-v2"))]
pub(crate) static LEGACY_V2_ROUTE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);
/// Test-only count of prepared computes served by the reuse route.
#[cfg(all(test, feature = "feature-regime-v2"))]
pub(crate) static REUSE_ROUTE_HITS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "feature-regime-v2")]
fn legacy_v2_route() -> bool {
    #[cfg(test)]
    {
        LEGACY_V2_ROUTE.load(core::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        false
    }
}

#[cfg(feature = "feature-regime-v2")]
impl Fused944Session {
    /// Empty session — buffers grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// The walk's retention — populated by the v2-aware compare paths; the
    /// `ZENSIM_NEIGHBOUR_EXACT` local-refinement snapshot reads it.
    pub(crate) fn retention(&self) -> &crate::feature_v2::FoldRetention {
        &self.retention
    }

    /// Mutable retention, for a steering snapshot that moves planes out at
    /// the end of a call (`LocalRefineSnapshot::capture_taking`).
    pub(crate) fn retention_mut(&mut self) -> &mut crate::feature_v2::FoldRetention {
        &mut self.retention
    }

    /// The v1 attr-planes streaming walk: fills `self.basic` (retained
    /// planes, `result`) and returns the scored result. Shared by the
    /// basic-only plans and the v2-bearing plans (whose v1 block it serves).
    fn run_basic_walk(
        &mut self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        plan: &crate::feature_plan::Plan,
        parallel: bool,
        encoding: Option<crate::feature_v2::HdrEncoding>,
    ) -> crate::ZensimResult {
        let mut config = config_from_params(crate::ZensimProfile::BHdr.params(), parallel);
        config.formula_revision = Some(plan.compute.formula_revision);
        config.compute_all_features = true;
        config.extended_features = false;
        config.compute_iw_features = false;
        config.local_only = plan.compute.local_only;
        config.omit_edges = plan.compute.omit_edges;
        config.attribution_channels = Some(core::array::from_fn(|scale| {
            core::array::from_fn(|ch| plan.compute.channel_active(scale, ch))
        }));
        let supplied_xyb = encoding.map(|encoding| {
            let (_, w, h) = precomputed.scale(0);
            let mut planes = core::array::from_fn(|_| vec![0.0; w * h]);
            let revision = plan.compute.formula_revision;
            if distorted.width() < 64 || distorted.height() < 64 {
                let padded = crate::metric::reflect_pad_to_min(distorted);
                crate::feature_v2_stream::hdr_source_to_xyb(
                    &padded,
                    encoding,
                    &mut planes,
                    revision,
                );
            } else {
                crate::feature_v2_stream::hdr_source_to_xyb(
                    distorted,
                    encoding,
                    &mut planes,
                    revision,
                );
            }
            planes
        });
        crate::streaming::compute_zensim_streaming_with_ref_and_attr_planes_input(
            precomputed,
            distorted,
            &config,
            crate::ZensimProfile::BHdr.params().weights,
            supplied_xyb,
            |scale, stats, _src, dst, planes, w, h| {
                let n = w * h;
                if self.basic.scales.len() <= scale {
                    self.basic.scales.push(BasicRetainedScale {
                        stats: stats.clone(),
                        distorted: core::array::from_fn(|_| Vec::new()),
                        planes: crate::streaming::AttrScaleRetention::new(0),
                        width: w,
                        height: h,
                    });
                }
                let out = &mut self.basic.scales[scale];
                out.stats.clone_from(stats);
                out.width = w;
                out.height = h;
                for (ch, distorted) in dst.iter().enumerate() {
                    if !plan.compute.channel_active(scale, ch) {
                        out.distorted[ch].clear();
                        out.planes.sd[ch].clear();
                        out.planes.mu1[ch].clear();
                        out.planes.mu2[ch].clear();
                        continue;
                    }
                    for (target, source) in [
                        (&mut out.distorted[ch], *distorted),
                        (&mut out.planes.sd[ch], &planes.sd[ch][..n]),
                        (&mut out.planes.mu1[ch], &planes.mu1[ch][..n]),
                        (&mut out.planes.mu2[ch], &planes.mu2[ch][..n]),
                    ] {
                        target.resize(n, 0.0);
                        target.copy_from_slice(source);
                    }
                }
            },
        )
    }

    pub(crate) fn planned_features(
        &mut self,
        source: &impl ImageSource,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        plan: &crate::feature_plan::Plan,
        parallel: bool,
        encoding: Option<crate::feature_v2::HdrEncoding>,
    ) -> Result<(Vec<f64>, [f64; 3]), ZensimError> {
        self.basic.result = None;
        if plan.toggles().v1_only
            && plan.compute.free_extras == crate::feature_v2::V1FreeExtras::Off
            && plan.compute.sampling.is_none()
            && matches!(
                plan.compute.v1_pools,
                crate::feature_v2::V1PoolsMode::Off | crate::feature_v2::V1PoolsMode::Peaks
            )
        {
            let result = self.run_basic_walk(precomputed, distorted, plan, parallel, encoding);
            let mut features = result.features().to_vec();
            // REV4SERVE review F1: the emit claim must already be
            // materialized — a promised slot the walk skipped must error,
            // not zero-fill a read.
            debug_assert!(
                plan.emit_covered(features.len()),
                "plan emit claims ids past the emitted vector (bound {} > {})",
                plan.emit_bound(),
                features.len()
            );
            plan.check_emit_covered(features.len())
                .map_err(|_| ZensimError::ModelLoadFailed {
                    reason: "bake plan emits features the streaming walk did not materialize",
                })?;
            features.resize(plan.walk_width().max(372), 0.0);
            let mean_offset = result.mean_offset();
            self.basic.result = Some(result);
            return Ok((features, mean_offset));
        }
        if let Some(encoding) = encoding {
            let mut mean = crate::feature_v2::MeanOffsetRows::new(
                precomputed.scale(0).1,
                precomputed.scale(0).2,
            );
            let result = crate::feature_v2::compute_folded720_hdr_streaming_extras(
                source,
                distorted,
                encoding,
                Some(120_000_000),
                parallel,
                plan.toggles(),
                &mut self.scratch,
                crate::feature_v2::FoldWalkExtras {
                    stop: None,
                    compute: Some(plan.compute),
                    retention: Some(&mut self.retention),
                    mean_offset: Some(&mut mean),
                    ref_planes: Some(&precomputed.scales),
                },
            )?;
            return Ok((result.into_features(), mean.finish()));
        }
        if plan.toggles().v1_only
            && plan.compute.free_extras == crate::feature_v2::V1FreeExtras::Off
            && !source.is_hdr()
            && let Some(result) = crate::feature_v2::compute_folded_v1_372_with_ref_impl(
                precomputed,
                distorted,
                parallel,
                &mut self.scratch,
                Some(plan.compute.v1_pools),
                Some(plan),
            )
        {
            return Ok(result);
        }
        // v2-bearing plans: the v1 walk below owns every basic/peak slot and
        // retains the planes the basic map reads; the v2 walk then runs with
        // its v1 block switched off, so no v1 accumulator runs twice. The v2
        // walk's own accumulation order is unchanged (the v1 fold writes only
        // `acc.v1` and the v1 slots), and `mean_offset` is the v2 walk's, as
        // before.
        if !plan.toggles().v1_only
            && plan.compute.formula_revision < crate::feature_defs::FormulaRevision::Rev5
            && !legacy_v2_route()
            && plan.compute.v1_basic
            && plan.compute.free_extras == crate::feature_v2::V1FreeExtras::Off
            && plan.compute.sampling.is_none()
            && !source.is_hdr()
            && !distorted.is_hdr()
            && matches!(
                plan.compute.v1_pools,
                crate::feature_v2::V1PoolsMode::Off | crate::feature_v2::V1PoolsMode::Peaks
            )
        {
            #[cfg(test)]
            REUSE_ROUTE_HITS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            let n_scales = crate::NUM_SCALES;
            let basic_result = self.run_basic_walk(precomputed, distorted, plan, parallel, None);
            let mut v2_plan = plan.clone();
            v2_plan.compute.v1_basic = false;
            let (v2_row, mean_offset) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
                source,
                distorted,
                Some(120_000_000),
                parallel,
                &mut self.scratch,
                Some(&v2_plan),
                Some(&mut self.retention),
            )?;
            let v1_total = n_scales * 3 * 31;
            let peaks_on = plan.compute.v1_pools == crate::feature_v2::V1PoolsMode::Peaks;
            let basic_row = basic_result.features();
            let mut features = Vec::with_capacity(v1_total + v2_row.len());
            features.resize(v1_total, 0.0);
            for scale in 0..n_scales {
                for ch in 0..3 {
                    if !plan.compute.channel_active(scale, ch) {
                        continue;
                    }
                    let b = scale * 39 + ch * 13;
                    features[b..b + 13].copy_from_slice(&basic_row[b..b + 13]);
                    if peaks_on {
                        let p = n_scales * 39 + (scale * 3 + ch) * 6;
                        features[p..p + 6].copy_from_slice(&basic_row[p..p + 6]);
                    }
                }
            }
            features.extend_from_slice(&v2_row);
            self.basic.result = Some(basic_result);
            return Ok((features, mean_offset));
        }
        // Basic-only maps use the cached v1 owner. Cheap free extras do not
        // populate the v2 retention cells; they are reported as unsupported
        // below rather than read from stale/default accumulators.
        let retention = (!plan.toggles().v1_only).then_some(&mut self.retention);
        // A prepared reference already holds the source's converted pyramid;
        // feed it instead of converting and downscaling the source again.
        #[cfg(feature = "custom-profiles")]
        if crate::feature_v2::retained_ref_feed_admits(precomputed, distorted, plan) {
            return Ok(crate::feature_v2::compute_folded_with_ref_retained(
                source,
                precomputed,
                distorted,
                parallel,
                &mut self.scratch,
                plan,
                retention,
            ));
        }
        crate::feature_v2::compute_folded_v1_372_streaming_impl(
            source,
            distorted,
            Some(120_000_000),
            parallel,
            &mut self.scratch,
            Some(plan),
            retention,
        )
    }
}

/// A complete candidate comparison and its local spatial attribution.
///
/// The candidate's declared feature plan supplies the complete serving score.
/// The map uses local sensitivities, with prepared lower-floor recovery as
/// described in [`Self::sensitivities`]. Attribution remains a local approximation
/// of finite pixel changes. [`Self::attribution`] supplies an additive density;
/// [`Self::refinement_gain`] additionally includes finite max-signal removal.
/// Inspect the corresponding coverage report and [`Self::has_corruption_gate`].
#[cfg(feature = "feature-regime-v2")]
#[non_exhaustive]
pub struct ScoredAttribution {
    /// Research-only complete native reference-paste gains, eagerly scored.
    pub(crate) exact_refinement: Option<Vec<([usize; 4], f64)>>,
    pub(crate) result: crate::ZensimResult,
    pub(crate) attribution: AttributionResult,
    pub(crate) sensitivities: Vec<f64>,
    pub(crate) unsupported_feature_ids: Vec<usize>,
    pub(crate) has_corruption_gate: bool,
    pub(crate) max_removals: Vec<MaxRemoval>,
    pub(crate) moment_removals: Vec<MomentRemoval>,
    pub(crate) unsupported_refinement_feature_ids: Vec<usize>,
    /// Rev5 defaults to replay; an explicit `ZENSIM_NEIGHBOUR_EXACT`
    /// value enables it only when exactly `"1"`. The coarse-retention snapshot the
    /// local refinement engine queries in [`Self::refinement_gain`].
    /// `None` when replay is disabled or the snapshot refused
    /// (sampling plan, v2 off, reflect-padded input, foreign dims —
    /// never silently degraded). Boxed so `ScoredAttribution` does not
    /// grow inline by the snapshot's array fields when off.
    /// Gated with the module's only builder (`compute_attribution_input`).
    #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
    pub(crate) neighbour_exact: Option<Box<crate::local_refine::LocalRefineSnapshot>>,
}

#[cfg(feature = "feature-regime-v2")]
impl ScoredAttribution {
    /// The complete served score and its identity-indexed feature row.
    pub fn result(&self) -> &crate::ZensimResult {
        &self.result
    }

    /// Signed score-gain density for supported integrands.
    /// With neighbour replay, coarse v2 contributions are supplied by
    /// [`Self::refinement_gain`] and omitted here to avoid counting them twice.
    pub fn attribution(&self) -> &AttributionResult {
        &self.attribution
    }

    /// Complete finite sensitivities, including unsupported spatial terms.
    /// Prepared steering below a single bake's constant lower spline floor
    /// can return pre-calibration sensitivities in network-score units.
    /// The [`Self::result`] score remains the complete serving score.
    pub fn sensitivities(&self) -> &[f64] {
        &self.sensitivities
    }

    /// Locally nonzero sensitivities whose integrands the additive density cannot serve.
    ///
    /// Reference-only and SDR structural-zero terms are deliberately excluded.
    /// A missing ID is not proof that a finite intervention has no effect:
    /// a clamp or gate may have zero local sensitivity and still be crossed.
    pub fn unsupported_feature_ids(&self) -> &[usize] {
        &self.unsupported_feature_ids
    }

    /// Estimated score gain for refining a half-open source-pixel rectangle.
    ///
    /// Adds finite max-signal removal to the additive density integral. This
    /// result is **not additive**: two blocks may each leave a tied maximum
    /// unchanged while their union removes it. Do not sum separate queries
    /// to obtain the union's estimate. Coordinates are clamped to the image;
    /// empty or inverted rectangles return zero.
    ///
    /// A max sample is removed only when its entire downsampled source
    /// footprint is contained. Reflected duplicates are included; SIMD padding
    /// belongs to the nearest extended-image edge. Partial coarse footprints
    /// survive. This freezes the extracted signals: actual pixel edits also
    /// change blurred neighborhoods and can create new maxima. Root curvature,
    /// model nonlinearity and corruption/clamp crossing remain approximations.
    /// Binning affects only the additive term, as in [`AttributionResult::query_rect`].
    /// Rev5 defaults to exact neighbour replay for coarse v2 features at
    /// scales 1–3; other frozen-signal and model-curvature limits remain.
    /// Prepared steering beneath a constant lower calibration floor can
    /// express gains in pre-calibration network-score units, while the
    /// [`Self::result`] score keeps the serving floor.
    /// With [`BakeScorer::with_finite_moment_refinement`](crate::BakeScorer::with_finite_moment_refinement),
    /// this also adds finite L2/L4/L8 removal corrections from base-image
    /// binned moment integrals. Those corrections are non-additive and use
    /// the same bin interpolation. Frozen-signal and head approximations remain.
    ///
    /// Research mode `ZENSIM_STEER_EXACT=all-blocks-v1` replaces this estimate
    /// with an eagerly scored native reference-paste gain on its fixed grids.
    /// A rectangle outside that exact inventory returns NaN (unsupported),
    /// never an approximate value. The ordinary score and density are unchanged.
    ///
    /// Inspect [`Self::unsupported_refinement_feature_ids`] and
    /// [`Self::has_corruption_gate`]. Complete local coverage does not establish
    /// accurate finite pixel or codec steering.
    pub fn refinement_gain(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        if let Some(table) = &self.exact_refinement {
            return table
                .iter()
                .find(|(r, _)| *r == [x0, y0, x1, y1])
                .map_or(f64::NAN, |(_, gain)| *gain);
        }
        if x0 >= x1.min(self.attribution.width()) || y0 >= y1.min(self.attribution.height()) {
            return 0.0;
        }
        let mut gain = self.attribution.query_rect(x0, y0, x1, y1);
        for map in &self.max_removals {
            gain -= map.sensitivity * map.feature_drop(x0, y0, x1, y1);
        }
        for map in &self.moment_removals {
            gain += map.correction(x0, y0, x1, y1);
        }
        // `ZENSIM_NEIGHBOUR_EXACT`: the exact finite v2 deltas at scales
        // 1–3, whose frozen-density mass was zeroed at build time. A
        // captured snapshot always produces a value — every refusal case
        // is decided at capture — so `None` can only come from an
        // out-of-range scale request, which the fixed 1..=3 range never
        // is; it adds nothing rather than fabricating a value.
        #[cfg(all(feature = "custom-profiles", feature = "feature-regime-v2"))]
        if let Some(snap) = &self.neighbour_exact
            && let Some(g) = snap.weighted_gain(
                &self.sensitivities,
                (x0, y0, x1, y1),
                &crate::local_refine::Candidate::Reference,
            )
        {
            gain += g;
        }
        gain
    }

    /// Locally active feature IDs not represented by [`Self::refinement_gain`].
    ///
    /// Unlike density-only [`Self::unsupported_feature_ids`], this excludes
    /// max terms whose retained signal maximum exactly matches the complete
    /// served feature. A zero local derivative is not proof that a finite
    /// edit is inert. Reflected/coarse ownership and blur approximations still
    /// apply even when this list is empty.
    pub fn unsupported_refinement_feature_ids(&self) -> &[usize] {
        &self.unsupported_refinement_feature_ids
    }

    /// Whether the complete score includes a discontinuous corruption gate.
    /// Central probes include that gate, but cannot guarantee that a finite
    /// block edit stays on the same side of it.
    pub fn has_corruption_gate(&self) -> bool {
        self.has_corruption_gate
    }
}

#[cfg(feature = "feature-regime-v2")]
pub(crate) fn candidate_map_sensitivities(
    plan: &crate::feature_plan::Plan,
    sensitivities: &[f64],
) -> (Vec<f64>, Vec<usize>) {
    use crate::feature_defs::{Form, def_at};
    use crate::feature_v2::{APPEND2_PER_SCALE, idx_append2};
    let toggles = plan.toggles();
    let mut spatial = sensitivities.to_vec();
    let mut unsupported = Vec::new();
    for (id, value) in spatial.iter_mut().enumerate() {
        if *value == 0.0 {
            continue;
        }
        let reference_only =
            def_at(id, crate::NUM_SCALES).is_some_and(|def| def.signal.form == Form::ReferenceOnly);
        let append2_slot = (BLOCK_END_APPEND..BLOCK_END_APPEND2)
            .contains(&id)
            .then(|| (id - BLOCK_END_APPEND) % APPEND2_PER_SCALE);
        let sdr_zero = matches!(
            append2_slot,
            Some(idx_append2::HL_BIN1 | idx_append2::HL_BIN2)
        );
        if reference_only || sdr_zero {
            *value = 0.0;
            continue;
        }
        let v1_pool_missing = (BLOCK_END_BASIC..BLOCK_END_V1_POOLS).contains(&id)
            && !(id < BLOCK_END_V1_PEAKS && (id - BLOCK_END_BASIC) % 6 >= 3);
        // The globaff family has its own map term (`crate::globaff::map_planes`).
        let globaff = toggles.globaff
            && (crate::globaff::FIRST_ID..crate::globaff::FIRST_ID + crate::globaff::SLOTS)
                .contains(&id);
        let missing = !globaff
            && (v1_pool_missing
                || id >= BLOCK_END_APPEND2
                || (toggles.v1_only && id >= BLOCK_END_V1_POOLS)
                || (toggles.append2_dst_activity && append2_slot.is_some()));
        if missing {
            unsupported.push(id);
            *value = 0.0;
        }
    }
    // Avoid pass B when no supported v2 integrand is active, including a
    // basic-only plan after reuse of a session that previously held v2 data.
    if spatial.iter().skip(BLOCK_END_V1_POOLS).all(|s| *s == 0.0) {
        let end = if spatial.iter().skip(BLOCK_END_BASIC).any(|s| *s != 0.0) {
            BLOCK_END_V1_PEAKS
        } else {
            BLOCK_END_BASIC
        };
        spatial.truncate(end);
    }
    (spatial, unsupported)
}

#[cfg(feature = "feature-regime-v2")]
pub(crate) fn zero_attribution(width: usize, height: usize, bin: usize) -> AttributionResult {
    BinAccum::new(width, height, bin).into_result()
}

impl crate::metric::Zensim {
    /// **The fused compare (task #67 C3a)**: scalar score + attribution
    /// steering map from ONE plane pipeline — the codec-loop call shape.
    ///
    /// Runs the SAME per-scale walk as
    /// [`compute_with_ref_and_diffmap`](Self::compute_with_ref_and_diffmap)
    /// (the score is bit-identical to that path's), retaining each scale's
    /// SSIM-error and mu planes and deriving the BASIC-block (f0-155)
    /// attribution density from them with no additional blur work. The
    /// marginal cost over a score-only compare is the f32 combine + spread
    /// + SAT.
    ///
    /// `s` is the caller-supplied model gradient (`∂score/∂f_k`, basic
    /// layout; entries past 155 ignored). The density uses the C2b-final
    /// allocation (window-only spread) and the f32 fused-combine kernel —
    /// vs the standalone f64 [`compute_attribution_density`]
    /// (Self::compute_attribution_density) the density agrees to f32
    /// recompute precision (~2e-5 relative, gated by
    /// `fused_matches_standalone_attribution`), while the standalone path
    /// and its strict f64 identities are unchanged.
    ///
    /// # Errors
    ///
    /// Besides the usual pair validations: the profile must compute the
    /// full basic feature set on every channel (an MLP/bake profile —
    /// `compute_all_features` — or `extended_features`), and
    /// `blur_passes == 1`; otherwise
    /// [`ZensimError::ModelForwardFailed`] with a static reason.
    pub fn compute_with_ref_score_and_attribution(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        self.fused_score_attr_fresh(precomputed, distorted, s, None)
    }

    /// [`compute_with_ref_score_and_attribution`](Self::compute_with_ref_score_and_attribution)
    /// with a Level-2 binned map: the per-scale mass folds straight into a
    /// `bin × bin` grid — no full-resolution canvas, trim copy, density
    /// plane, or full-resolution SAT exist at any point. The **score is
    /// bit-identical to every other compare path** (the sink only receives
    /// map mass; the stats pipeline is untouched). `bin == 1` delegates to
    /// the per-pixel entry point bit-identically.
    ///
    /// # Panics
    ///
    /// Panics if `bin == 0`.
    ///
    /// # Errors
    ///
    /// Same contract as the per-pixel fused entry.
    pub fn compute_with_ref_score_and_attribution_binned(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        bin: usize,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        validate_ref_match(precomputed, distorted)?;
        assert!(bin > 0, "bin must be non-zero");
        if bin == 1 {
            return self.compute_with_ref_score_and_attribution(precomputed, distorted, s);
        }
        let mut accum = BinAccum::new(distorted.width(), distorted.height(), bin);
        let (result, _, _) = self.fused_basic_into(
            precomputed,
            distorted,
            s,
            &[],
            None,
            None,
            &mut AttrSinkF32::Bins(&mut accum),
        )?;
        Ok((result, accum.into_result()))
    }

    /// The fresh fused pipeline (the pre-#70 body of
    /// [`compute_with_ref_score_and_attribution`](Self::compute_with_ref_score_and_attribution),
    /// unchanged numerically). When `prime` is `Some`, additionally record
    /// each scale's coefficient packs + reference-side hf sums into the
    /// session — the stale entry's priming path.
    fn fused_score_attr_fresh(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        prime: Option<&mut AttributionSession>,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        let fb = self.fused_basic_canvas(precomputed, distorted, s, prime)?;
        let width = distorted.width();
        let height = distorted.height();
        let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
        let t_sat0 = crate::perf_clock::Instant::now();
        let trimmed = fb.trim_to(width, height);
        let attr = AttributionResult::from_density(trimmed, width, height);
        if perf_log {
            eprintln!(
                "ATTRPERF fused: pipeline {:.1} ms (combine/spread/upsample {:.1} ms → retention+stats {:.1} ms) | trim+SAT {:.1} ms",
                fb.t_pipe_ms,
                fb.combine_ms,
                fb.t_pipe_ms - fb.combine_ms,
                t_sat0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok((fb.result, attr))
    }

    /// The fused v1 walk core (appendix N refactor of the pre-existing
    /// fused pipeline, numerically unchanged): one score walk with
    /// attribution-plane retention → the compare's [`crate::metric::ZensimResult`]
    /// plus the UNTRIMMED f32 basic-block canvas at compute dims. The C3a
    /// entry trims + SATs it directly; the fused folded-944 entry sums the
    /// v2/append/append2 retention density into it first.
    fn fused_basic_canvas(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        prime: Option<&mut AttributionSession>,
    ) -> Result<FusedBasicCanvas, ZensimError> {
        validate_ref_match(precomputed, distorted)?;
        let (_, comp_pw, comp_h) = precomputed.scale(0);
        let mut canvas = vec![0.0f32; comp_pw * comp_h];
        let (result, t_pipe_ms, combine_ms) = self.fused_basic_into(
            precomputed,
            distorted,
            s,
            &[],
            None,
            prime,
            &mut AttrSinkF32::Canvas(&mut canvas),
        )?;
        Ok(FusedBasicCanvas {
            result,
            canvas,
            comp_pw,
            comp_h,
            t_pipe_ms,
            combine_ms,
        })
    }

    /// [`fused_basic_canvas`](Self::fused_basic_canvas) generalized over
    /// the mass sink (Level 2): the Canvas arm is the byte-identical
    /// pre-Level-2 fused path (incl. the scale-0 direct spread-into-canvas
    /// fusion); the Bins arm folds every scale's plane straight into the
    /// accumulator — no full-resolution canvas exists at all (the scale-0
    /// spread merges into the already-allocated id plane instead, then
    /// folds). Returns `(result, t_pipe_ms, combine_ms)`.
    fn fused_basic_into(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        s_peaks: &[f64],
        max_removals: Option<&mut Vec<MaxRemoval>>,
        prime: Option<&mut AttributionSession>,
        sink: &mut AttrSinkF32<'_>,
    ) -> Result<(crate::metric::ZensimResult, f64, f64), ZensimError> {
        self.fused_basic_into_at_revision(
            precomputed,
            distorted,
            s,
            s_peaks,
            max_removals,
            prime,
            sink,
            None,
            None,
            1,
            None,
            None,
            crate::map_paint::MapPaint::default(),
        )
    }

    fn fused_basic_into_at_revision(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        s_peaks: &[f64],
        mut max_removals: Option<&mut Vec<MaxRemoval>>,
        mut prime: Option<&mut AttributionSession>,
        sink: &mut AttrSinkF32<'_>,
        revision: Option<crate::feature_defs::FormulaRevision>,
        mut moment_removals: Option<&mut Vec<MomentRemoval>>,
        moment_bin: usize,
        retained: Option<BasicMapRetention<'_>>,
        map_scratch: Option<&mut BasicMapScratch>,
        paint: crate::map_paint::MapPaint,
    ) -> Result<(crate::metric::ZensimResult, f64, f64), ZensimError> {
        const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        crate::metric::validate_ref_dimensions(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels())?;
        let mut config = config_from_params(params, self.parallel());
        config.formula_revision = revision;
        // Only the candidate map owner may skip channels: its score/features
        // were already computed. Legacy callers also consume the returned row.
        if revision.is_some() {
            // This stage consumes only basic/peak signals. The canonical
            // planned extraction above already handled all scored families;
            // B's inherited masked/IW switches would run an unused chain here.
            config.compute_all_features = true;
            config.extended_features = false;
            config.compute_iw_features = false;
            config.attribution_channels = Some(core::array::from_fn(|scale| {
                core::array::from_fn(|c| {
                    let basic = (scale * 3 + c) * FPC;
                    let peaks = (scale * 3 + c) * 6;
                    (0..FPC).any(|k| s.get(basic + k).is_some_and(|v| *v != 0.0))
                        || (0..6).any(|k| s_peaks.get(peaks + k).is_some_and(|v| *v != 0.0))
                })
            }));
        }
        config.local_only = revision.is_some()
            && s.iter()
                .enumerate()
                .all(|(id, v)| *v == 0.0 || id % FPC < 10)
            && s_peaks.iter().all(|v| *v == 0.0);
        config.omit_edges = config.local_only
            && s.iter()
                .enumerate()
                .all(|(id, v)| *v == 0.0 || matches!(id % FPC, 0..=2 | 9));
        crate::metric::check_profile_route(params, &config)?;
        crate::metric::validate_cached_revision(precomputed, config.revision())?;
        if config.blur_passes != 1 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires blur_passes == 1 (all shipped profiles)",
            });
        }
        if !(config.compute_all_features || config.extended_features) {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires a profile computing all basic features \
                         (an MLP/bake profile or extended_features)",
            });
        }

        let width = distorted.width();
        let height = distorted.height();
        let (_, comp_pw, comp_h) = precomputed.scale(0);

        let t_all = crate::perf_clock::Instant::now();
        let combine_ms = std::cell::Cell::new(0.0f64);
        let mut local_scratch = BasicMapScratch::default();
        let work = map_scratch.unwrap_or(&mut local_scratch);
        let BasicMapScratch {
            id: id_plane,
            win: win_plane,
            spread_tmp,
            spread_out,
        } = work;
        if id_plane.len() < comp_pw * comp_h {
            id_plane.resize(comp_pw * comp_h, 0.0);
        }
        if win_plane.len() < comp_pw * comp_h {
            win_plane.resize(comp_pw * comp_h, 0.0);
        }

        // Source-resolution difference guide for research guided painting,
        // filled at scale 0 (`crate::map_paint`).
        let mut guide0_cell: Option<Vec<f64>> = None;
        let guide0 = &mut guide0_cell;
        let mut on_scale = |scale: usize,
                            stats: &crate::metric::ScaleStats,
                            src_planes: [&[f32]; 3],
                            dst_planes: [&[f32]; 3],
                            ret: &BasicPlaneView<'_>,
                            sw: usize,
                            sh: usize,
                            retained_hf: Option<[(f64, f64); 3]>| {
            let t_c = crate::perf_clock::Instant::now();
            let n = sw * sh;
            let n_f = n as f64;
            if let Some(output) = max_removals.as_deref_mut() {
                retain_max_removals(
                    output,
                    s_peaks,
                    scale,
                    width,
                    height,
                    sw,
                    sh,
                    src_planes,
                    dst_planes,
                    ret,
                    precomputed.sampling_geometry.as_ref(),
                );
            }
            id_plane[..n].fill(0.0);
            win_plane[..n].fill(0.0);
            let hf: [(f64, f64); 3] = retained_hf.unwrap_or_else(|| {
                core::array::from_fn(|c| {
                    let base = (scale * 3 + c) * FPC;
                    if prime.is_some()
                        || (10..13).any(|k| s.get(base + k).is_some_and(|v| *v != 0.0))
                    {
                        hf_src_sums(&src_planes[c][..n], &ret.mu1[c][..n])
                    } else {
                        (0.0, 0.0)
                    }
                })
            });
            let co32: [[f32; 12]; 3] = core::array::from_fn(|c| {
                let base_k = scale * FPC * 3 + c * FPC;
                let mut co = SlotCoeffs::from_scale_stats(
                    s,
                    base_k,
                    stats,
                    c,
                    n_f,
                    hf[c].0,
                    hf[c].1,
                    crate::hf_gain_form::HfGainForm::at_revision(config.formula_revision),
                    paint.split,
                );
                if paint.split == crate::map_paint::SplitMode::Amplitude {
                    let [sd, art, det] = l8_amplitude_fold(s_peaks, scale, c, n_f);
                    co.c_sd += sd;
                    co.c_art += art;
                    co.c_det += det;
                }
                coeffs_to_f32(&co)
            });
            let l8_co: [[f64; 3]; 3] = core::array::from_fn(|c| {
                l8_coefficients(s_peaks, scale, c, stats, n_f, paint.split)
            });
            // task #70: prime the stale session — THIS compare's coefficient
            // packs become the NEXT stale call's fold input; the hf sums are
            // reference-side constants cached once here.
            if let Some(sess) = prime.as_deref_mut() {
                sess.coeffs.push(co32);
                sess.hf_src.push(hf);
            }
            // Row-banded parallel combine (disjoint id/win rows per band).
            let stable7 = config.revision().stable_forms();
            let run_band = |band: usize, idc: &mut [f32], winc: &mut [f32]| {
                let off = band * 64 * sw;
                let len = idc.len();
                for c in 0..3 {
                    if co32[c].iter().all(|v| *v == 0.0) && l8_co[c].iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    let combine = if l8_co[c].iter().any(|v| *v != 0.0) {
                        combine_basic_and_l8::<true>
                    } else {
                        combine_basic_and_l8::<false>
                    };
                    combine(
                        &ret.sd[c][off..off + len],
                        &src_planes[c][off..off + len],
                        &dst_planes[c][off..off + len],
                        &ret.mu1[c][off..off + len],
                        &ret.mu2[c][off..off + len],
                        co32[c],
                        l8_co[c],
                        idc,
                        winc,
                        stable7,
                    );
                }
            };
            #[cfg(feature = "threads")]
            let banded = if config.allow_multithreading && sh > 64 {
                use rayon::prelude::*;
                id_plane[..n]
                    .par_chunks_mut(64 * sw)
                    .zip(win_plane[..n].par_chunks_mut(64 * sw))
                    .enumerate()
                    .for_each(|(band, (idc, winc))| run_band(band, idc, winc));
                true
            } else {
                false
            };
            #[cfg(not(feature = "threads"))]
            let banded = false;
            if !banded {
                run_band(0, &mut id_plane[..n], &mut win_plane[..n]);
            }
            // Window-class spread (C2b-final allocation) fused with the
            // window→identity merge, f32 fast path (#70 lever 1:
            // parallel, bitwise-invariant to banding). Canvas arm at
            // scale 0: the upsample is elementwise, so the spread merges
            // directly into the canvas — skipping a full id-plane
            // store+reload; this is value-exact: `(0+a)+b` and `0+(a+b)`
            // round identically (the first add is exact, signed zeros
            // included). Bins arm: no canvas exists — the spread merges
            // into the (already allocated) id plane at every scale, then
            // the id plane folds into the accumulator (reassociation-class
            // difference, gated by the L2-vs-L1 tests).
            match &mut *sink {
                AttrSinkF32::Canvas(canvas) => {
                    if scale == 0 {
                        upsample_add_sum_preserving_f32(
                            &id_plane[..n],
                            sw,
                            sh,
                            canvas,
                            comp_pw,
                            comp_h,
                            1,
                        );
                        crate::blur::box_spread_merge_f32(
                            &mut win_plane[..n],
                            &mut canvas[..n],
                            sw,
                            sh,
                            config.blur_radius,
                            spread_tmp,
                            spread_out,
                            config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                        );
                    } else {
                        crate::blur::box_spread_merge_f32(
                            &mut win_plane[..n],
                            &mut id_plane[..n],
                            sw,
                            sh,
                            config.blur_radius,
                            spread_tmp,
                            spread_out,
                            config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                        );
                        upsample_add_sum_preserving_f32(
                            &id_plane[..n],
                            sw,
                            sh,
                            canvas,
                            comp_pw,
                            comp_h,
                            1usize << scale,
                        );
                    }
                }
                AttrSinkF32::Bins(accum) => {
                    // The research window/paint modes (`crate::map_paint`);
                    // the defaults are the uniform spread and fold above.
                    let (window_mode, paint_mode) = (paint.window, paint.paint);
                    crate::map_paint::merge_window(
                        window_mode,
                        &mut win_plane[..n],
                        &mut id_plane[..n],
                        sw,
                        sh,
                        config.blur_radius,
                        || {
                            crate::map_paint::difference_guide(
                                src_planes,
                                dst_planes,
                                n,
                                paint.guide,
                            )
                        },
                        spread_tmp,
                        spread_out,
                        config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                    );
                    if scale == 0 && paint_mode == crate::map_paint::PaintMode::Guided {
                        *guide0 = Some(crate::map_paint::difference_guide(
                            src_planes,
                            dst_planes,
                            n,
                            paint.guide,
                        ));
                    }
                    if let Some(geometry) = &precomputed.sampling_geometry {
                        let projected = geometry.project(&id_plane[..n], scale);
                        accum.add_scale_plane_f32(&projected, width, height, 1);
                    } else {
                        accum.add_scale_plane_painted_f32(
                            &id_plane[..n],
                            sw,
                            sh,
                            scale,
                            paint_mode,
                            guide0.as_deref(),
                        );
                    }
                }
            }
            combine_ms.set(combine_ms.get() + t_c.elapsed().as_secs_f64() * 1e3);
            if let Some(output) = moment_removals.as_deref_mut() {
                retain_moment_removals(
                    output,
                    s,
                    s_peaks,
                    scale,
                    stats,
                    width,
                    height,
                    sw,
                    sh,
                    src_planes,
                    dst_planes,
                    ret,
                    precomputed.sampling_geometry.as_ref(),
                    moment_bin,
                    config.blur_radius,
                );
            }
        };

        let result = if let Some(BasicMapRetention::Basic(retained)) = retained {
            for (scale, cell) in retained.scales.iter().enumerate() {
                on_scale(
                    scale,
                    &cell.stats,
                    precomputed.scale(scale).0,
                    cell.distorted.each_ref().map(Vec::as_slice),
                    &BasicPlaneView::from(&cell.planes),
                    cell.width,
                    cell.height,
                    None,
                );
            }
            retained
                .result
                .as_ref()
                .expect("successful retained extraction")
                .clone()
        } else {
            #[cfg(feature = "feature-regime-v2")]
            if let Some(BasicMapRetention::Fold(retained)) = retained {
                for (scale, &(w, h)) in retained.dims.iter().enumerate() {
                    let view = BasicPlaneView {
                        sd: retained.basic_sd[scale].each_ref().map(Vec::as_slice),
                        mu1: core::array::from_fn(|c| retained.planes[scale][c].mu1.as_slice()),
                        mu2: core::array::from_fn(|c| retained.planes[scale][c].mu2.as_slice()),
                    };
                    on_scale(
                        scale,
                        &retained.basic_stats[scale],
                        retained.pyr_src[scale].each_ref().map(Vec::as_slice),
                        retained.pyr_dst[scale].each_ref().map(Vec::as_slice),
                        &view,
                        w,
                        h,
                        Some(retained.basic_hf[scale]),
                    );
                }
                // The complete candidate result is supplied by the caller.
                return Ok((
                    crate::ZensimResult::new(0.0, 0.0, Vec::new(), self.profile(), [0.0; 3]),
                    t_all.elapsed().as_secs_f64() * 1e3,
                    combine_ms.get(),
                ));
            }
            crate::streaming::compute_zensim_streaming_with_ref_and_attr_planes(
                precomputed,
                distorted,
                &config,
                params.weights,
                |scale, stats, src, dst, ret, w, h| {
                    on_scale(
                        scale,
                        stats,
                        src,
                        dst,
                        &BasicPlaneView::from(ret),
                        w,
                        h,
                        None,
                    )
                },
            )
        };
        let mut result = result.with_profile(self.profile());
        // Same real-scoring step as `compute_with_ref_and_diffmap` — the
        // scalar golden gate (`fused_score_bit_matches_diffmap_path`) holds
        // this path bit-identical to the fold-diffmap call's score.
        // Candidate callers already scored their own model. This pass only
        // supplies its map signals, so do not run the legacy profile's head
        // against a deliberately narrower feature row.
        if revision.is_none() && precomputed.sampling.is_none() {
            crate::metric::apply_mlp_scoring_with_codec(
                &mut result,
                params,
                width as u32,
                height as u32,
                None,
            )?;
        }

        let t_pipe = t_all.elapsed().as_secs_f64() * 1e3;
        Ok((result, t_pipe, combine_ms.get()))
    }

    /// **Stale-scalar single-pass fused compare (task #70; C3a ranked
    /// lever 3)**: scalar score + attribution steering map from ONE
    /// pipeline pass with NO second sweep — the attribution combine runs
    /// in-strip on cache-hot planes using the coefficient packs derived
    /// from the PREVIOUS compare through the same `session`.
    ///
    /// **Semantics — the map lags one iterate.** The returned map combines
    /// THIS compare's per-pixel signal planes with the PREVIOUS compare's
    /// pooled scalars (`SlotCoeffs` — the stats-derived coefficient
    /// packs). One-iterate staleness of the steering signal is the
    /// proven-free semantics in the codec loop (C3b `attr-stale`, #69 G4,
    /// and the metric-matrix study all measured it free); this variant is
    /// strictly fresher than those arms' fully-stale map (planes are
    /// current, only the pooled scalars lag). The **score is bit-identical
    /// to every other compare path** (the fold never touches the stats
    /// pipeline) — only the map lags.
    ///
    /// The FIRST call through an unprimed `session` (and any call after
    /// [`AttributionSession::reset`], or when the image size or `s`
    /// changes) runs the fresh fused path
    /// ([`compute_with_ref_score_and_attribution`](Self::compute_with_ref_score_and_attribution))
    /// and primes the session — that call returns the FRESH map at fresh
    /// cost; subsequent calls are single-pass. The session is valid for
    /// ONE precomputed reference; switching references without `reset()`
    /// is a caller error the session cannot detect.
    ///
    /// # Errors
    ///
    /// Same contract as the fresh fused entry (all-basic-features profile,
    /// `blur_passes == 1`, matching pair dims).
    pub fn compute_with_ref_score_and_attribution_stale(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut AttributionSession,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        self.stale_core(precomputed, distorted, s, session, None)
    }

    /// [`compute_with_ref_score_and_attribution_stale`](Self::compute_with_ref_score_and_attribution_stale)
    /// with a Level-2 binned map (same one-iterate-stale semantics, same
    /// bit-identical score): the in-strip fold's per-scale planes fold
    /// straight into the `bin × bin` grid — the session's full-resolution
    /// canvas is never allocated, and the per-iteration trim + SAT build
    /// shrink by `bin²`. `bin == 1` delegates to the per-pixel stale entry
    /// bit-identically. The priming call (first through the session, or
    /// after `reset`/size/gradient change) also returns a binned map.
    ///
    /// # Panics
    ///
    /// Panics if `bin == 0`.
    ///
    /// # Errors
    ///
    /// Same contract as the per-pixel stale entry.
    pub fn compute_with_ref_score_and_attribution_stale_binned(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut AttributionSession,
        bin: usize,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        assert!(bin > 0, "bin must be non-zero");
        if bin == 1 {
            return self.stale_core(precomputed, distorted, s, session, None);
        }
        let accum = BinAccum::new(distorted.width(), distorted.height(), bin);
        self.stale_core(precomputed, distorted, s, session, Some(accum))
    }

    /// Shared core of the two stale entries: `bins == None` is the
    /// byte-identical pre-Level-2 canvas path; `Some(accum)` folds into the
    /// bins (no session canvas, no trim, tiny SAT).
    fn stale_core(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut AttributionSession,
        mut bins: Option<BinAccum>,
    ) -> Result<(crate::metric::ZensimResult, AttributionResult), ZensimError> {
        const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        validate_ref_match(precomputed, distorted)?;
        check_within_max_pixels(distorted.width(), distorted.height(), self.max_pixels())?;
        let config = config_from_params(params, self.parallel());
        crate::metric::check_profile_route(params, &config)?;
        crate::metric::validate_cached_revision(precomputed, config.revision())?;
        if config.blur_passes != 1 {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires blur_passes == 1 (all shipped profiles)",
            });
        }
        if !(config.compute_all_features || config.extended_features) {
            return Err(ZensimError::ModelForwardFailed {
                reason: "fused attribution requires a profile computing all basic features \
                         (an MLP/bake profile or extended_features)",
            });
        }

        let width = distorted.width();
        let height = distorted.height();

        // Unprimed, or the size/gradient changed: fresh fused path, prime.
        if !session.primed_for(width, height, s) {
            session.reset();
            session.width = width;
            session.height = height;
            session.s_saved = s.to_vec();
            return match bins {
                None => self.fused_score_attr_fresh(precomputed, distorted, s, Some(session)),
                Some(mut accum) => {
                    let (result, _, _) = self.fused_basic_into(
                        precomputed,
                        distorted,
                        s,
                        &[],
                        None,
                        Some(session),
                        &mut AttrSinkF32::Bins(&mut accum),
                    )?;
                    Ok((result, accum.into_result()))
                }
            };
        }

        let (_, comp_pw, comp_h) = precomputed.scale(0);
        let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
        let t_all = crate::perf_clock::Instant::now();
        let tail_ms = std::cell::Cell::new(0.0f64);
        // Session-owned scratch: no per-iteration allocation/page-fault
        // cost on the single-pass path. The canvas accumulates across
        // scales, so it is re-zeroed here — Canvas arm only; the bins arm
        // never allocates it.
        let comp_n = comp_pw * comp_h;
        if bins.is_none() {
            session.canvas.resize(comp_n, 0.0);
            session.canvas.fill(0.0);
        }
        session.id_plane.resize(comp_n, 0.0);
        session.win_plane.resize(comp_n, 0.0);
        let mut canvas = std::mem::take(&mut session.canvas);
        let mut id_plane = std::mem::take(&mut session.id_plane);
        let mut win_plane = std::mem::take(&mut session.win_plane);
        let mut spread_tmp = std::mem::take(&mut session.spread_tmp);
        let mut spread_out = std::mem::take(&mut session.spread_out);
        // The previous compare's packs drive THIS pass's in-strip fold;
        // this pass's stats produce the packs for the NEXT call. (On an
        // error return below the session is left unprimed — the next call
        // re-primes through the fresh path.)
        let coeffs_prev = std::mem::take(&mut session.coeffs);
        let mut next_coeffs: Vec<[[f32; 12]; 3]> = Vec::with_capacity(coeffs_prev.len());
        let hf_cached = &session.hf_src;

        let on_scale = |scale: usize,
                        stats: &crate::metric::ScaleStats,
                        idp: &mut [f32],
                        winp: &mut [f32],
                        sw: usize,
                        sh: usize| {
            let t_c = crate::perf_clock::Instant::now();
            let n = sw * sh;
            let n_f = n as f64;
            // Post-scale tail — Canvas arm IDENTICAL to the fresh path
            // (same fused spread+merge, same scale-0 canvas-target fusion,
            // same upsample ⇒ bitwise-equal maps when the coefficient
            // packs coincide). Bins arm mirrors the fresh binned path:
            // spread merges into the walk's id plane, then folds.
            match bins.as_mut() {
                None => {
                    if scale == 0 {
                        upsample_add_sum_preserving_f32(
                            &idp[..n],
                            sw,
                            sh,
                            &mut canvas,
                            comp_pw,
                            comp_h,
                            1,
                        );
                        crate::blur::box_spread_merge_f32(
                            &mut winp[..n],
                            &mut canvas[..n],
                            sw,
                            sh,
                            config.blur_radius,
                            &mut spread_tmp,
                            &mut spread_out,
                            config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                        );
                    } else {
                        crate::blur::box_spread_merge_f32(
                            &mut winp[..n],
                            &mut idp[..n],
                            sw,
                            sh,
                            config.blur_radius,
                            &mut spread_tmp,
                            &mut spread_out,
                            config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                        );
                        upsample_add_sum_preserving_f32(
                            &idp[..n],
                            sw,
                            sh,
                            &mut canvas,
                            comp_pw,
                            comp_h,
                            1usize << scale,
                        );
                    }
                }
                Some(accum) => {
                    crate::blur::box_spread_merge_f32(
                        &mut winp[..n],
                        &mut idp[..n],
                        sw,
                        sh,
                        config.blur_radius,
                        &mut spread_tmp,
                        &mut spread_out,
                        config.allow_multithreading && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
                    );
                    accum.add_scale_plane_f32(&idp[..n], sw, sh, 1usize << scale);
                }
            }
            // Derive the NEXT call's packs from THIS compare's pooled
            // scalars + the cached reference-side hf sums.
            let hf = hf_cached.get(scale).copied().unwrap_or([(0.0, 0.0); 3]);
            next_coeffs.push(core::array::from_fn(|c| {
                let base_k = scale * FPC * 3 + c * FPC;
                coeffs_to_f32(&SlotCoeffs::from_scale_stats(
                    s,
                    base_k,
                    stats,
                    c,
                    n_f,
                    hf[c].0,
                    hf[c].1,
                    crate::hf_gain_form::HfGainForm::at_revision(config.formula_revision),
                    // The stale route has no paint modes.
                    crate::map_paint::SplitMode::Repair,
                ))
            }));
            tail_ms.set(tail_ms.get() + t_c.elapsed().as_secs_f64() * 1e3);
        };

        let result = crate::streaming::compute_zensim_streaming_with_ref_and_attr_fold(
            precomputed,
            distorted,
            &config,
            params.weights,
            &coeffs_prev,
            &mut id_plane,
            &mut win_plane,
            on_scale,
        );
        let mut result = result.with_profile(self.profile());
        // Same real-scoring step as every compare path — bit-identical
        // score (the in-strip fold reads planes, never writes stats).
        crate::metric::apply_mlp_scoring_with_codec(
            &mut result,
            params,
            width as u32,
            height as u32,
            None,
        )?;
        session.coeffs = next_coeffs;

        let t_pipe = t_all.elapsed().as_secs_f64() * 1e3;
        let t_sat0 = crate::perf_clock::Instant::now();
        let attr = match bins {
            None => {
                // Trim into the recycled density buffer (fresh alloc when
                // the caller never recycles — identical behavior, just
                // re-allocating).
                let mut trimmed = std::mem::take(&mut session.density_scratch);
                trimmed.clear();
                if comp_pw == width && comp_h == height {
                    trimmed.extend_from_slice(&canvas);
                } else {
                    trimmed.reserve(width * height);
                    for y in 0..height.min(comp_h) {
                        trimmed.extend_from_slice(
                            &canvas[y * comp_pw..y * comp_pw + width.min(comp_pw)],
                        );
                    }
                }
                let sat_scratch = std::mem::take(&mut session.sat_scratch);
                AttributionResult::from_density_with_sat_scratch(
                    trimmed,
                    width,
                    height,
                    sat_scratch,
                )
            }
            Some(accum) => accum.into_result(),
        };
        // Return the scratch to the session for the next iteration.
        session.canvas = canvas;
        session.id_plane = id_plane;
        session.win_plane = win_plane;
        session.spread_tmp = spread_tmp;
        session.spread_out = spread_out;
        if perf_log {
            eprintln!(
                "ATTRPERF stale: pipeline {:.1} ms (spread/upsample/derive tail {:.1} ms; in-strip fold inside walk) | trim+SAT {:.1} ms",
                t_pipe,
                tail_ms.get(),
                t_sat0.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok((result, attr))
    }

    /// **The fused folded-944 compare (campaign appendix N)**: canonical
    /// streaming folded-944 extraction + FULL-coverage attribution
    /// steering map from ONE extraction pass plus one v1 walk — the
    /// 944-class codec-loop call shape.
    ///
    /// Returns, in order:
    /// 1. the v1 walk's [`crate::metric::ZensimResult`] under `self`'s
    ///    profile (the loop reads `approx_butteraugli` from it — its score
    ///    is NOT the 944 model's score);
    /// 2. the [`crate::feature_v2::ZensimV2Result`] carrying the 944
    ///    features, BITWISE identical to
    ///    [`compute_folded720_append2_features`](Self::compute_folded720_append2_features)
    ///    (G-N1 — the retention hooks only copy; accumulation is
    ///    untouched). The caller forwards `features()` through
    ///    [`crate::score_features_with_profile`] exactly as the unfused
    ///    score path does — a PRUNED bake (caller 944 / internal 667)
    ///    consumes the full caller-width vector;
    /// 3. the [`AttributionResult`]: basic block (f0-155) from the C3a
    ///    fused v1 machinery + v2/append/append2 (f372-943) from pass-B
    ///    over the walk retention, coefficients derived from the EXACT
    ///    extraction accumulators. Matches the standalone
    ///    [`compute_attribution_density_full`](Self::compute_attribution_density_full)
    ///    within the C3a tolerance class (G-N2); per-width slot coverage
    ///    is pinned by the shared coverage gate (G-N3).
    ///
    /// `source` must be the SAME image `precomputed` was built from (the
    /// usual `*_with_ref` caller contract; dims are validated, content
    /// cannot be). `s` is the raw caller-layout gradient (944-wide for the
    /// 944 class; shorter widths slice per the named `BLOCK_END_*`
    /// bounds). SDR route only: HDR-declared inputs get
    /// [`ZensimError::HdrInputRequiresPuPath`].
    ///
    /// A byte-identical pair returns exactly what
    /// [`crate::Zensim::compute`] returns — 100 — with a zero map; the 944
    /// feature row is the real extraction either way. Only this entry and the
    /// [`crate::BakeScorer`] ones can make that call: the `*_with_ref*`
    /// attribution entries hold an XYB pyramid, not the source pixels, so a
    /// perfect copy is scored through the model there.
    ///
    /// # Errors
    ///
    /// The C3a fused contract (all-basic-features profile,
    /// `blur_passes == 1`, matching pair dims) plus the folded
    /// extraction's own validations.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded944_score_and_attribution(
        &self,
        source: &impl ImageSource,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut Fused944Session,
    ) -> Result<
        (
            crate::metric::ZensimResult,
            crate::feature_v2::ZensimV2Result,
            AttributionResult,
        ),
        ZensimError,
    > {
        // REV4SERVE: this entry follows the process revision — at Rev4 the
        // retention walk and `attr_pass_b_*` are all per-pixel canonical.
        session.basic.result = None;
        validate_pair(source, distorted)?;
        // ZENSIM_ATTR_PERF=1: coarse section timing (perf lever triage).
        let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
        let t0 = crate::perf_clock::Instant::now();
        // 1) Folded-944 extraction with retention — exact features.
        let v2res = crate::feature_v2::compute_folded944_streaming_with_retention(
            source,
            distorted,
            self.max_pixels(),
            self.parallel(),
            &mut session.scratch,
            &mut session.retention,
        )?;
        let t_extract = t0.elapsed();
        // Identity: the fused v1 walk has no short-circuit of its own, so a
        // byte-identical pair scored 96.2384 here while `compute` returned
        // exactly 100 on the same pixels. This entry is the one member of the
        // attribution family that HOLDS the source, so it can reach the single
        // identity owner (`metric::images_byte_identical` -> `compute`); the
        // `*_with_ref*` entries cannot and still score a perfect copy through
        // the model. Same disposition `BakeScorer::compute_attribution_input`
        // already uses: identity result plus a zero map. The 944 feature row is
        // the real extraction either way — its bitwise contract with
        // `compute_folded720_append2_features` (G-N1) is unconditional.
        if crate::metric::images_byte_identical(source, distorted) {
            validate_ref_match(precomputed, distorted)?;
            return Ok((
                self.compute(source, distorted)?,
                v2res,
                zero_attribution(distorted.width(), distorted.height(), 1),
            ));
        }
        // 2) Fused v1 walk — basic-block canvas + the map-profile result.
        let t1 = crate::perf_clock::Instant::now();
        let fb = self.fused_basic_canvas(precomputed, distorted, s, None)?;
        let t_v1 = t1.elapsed();
        let width = distorted.width();
        let height = distorted.height();
        // 3) v2/append/append2 density from retention (same block slicing
        //    as `compute_attribution_density_full`).
        let block = |start: usize, end: usize| -> Option<&[f64]> {
            (s.len() > start).then(|| &s[start..s.len().min(end)])
        };
        let s_v2: &[f64] = block(BLOCK_END_V1_POOLS, BLOCK_END_V2).unwrap_or(&[]);
        let s_append: Option<&[f64]> = block(BLOCK_END_V2, BLOCK_END_APPEND);
        let s_append2: Option<&[f64]> = block(BLOCK_END_APPEND, BLOCK_END_APPEND2);
        // Appendix P lever 1: fully-f32 canvas assembly — the f32 basic
        // canvas plus the f32 pass-B retention density, one f64 conversion
        // at the SAT build (`from_density`, the C3a entry's own shape).
        let t2 = crate::perf_clock::Instant::now();
        let mut canvas: Vec<f32> = fb.trim_to(width, height);
        let mut lattice = Vec::new();
        if !s_v2.is_empty() || s_append.is_some() || s_append2.is_some() {
            let (v2a, v2_lattice) = crate::feature_v2::compute_v2_append_attribution_from_retention(
                &session.retention,
                s_v2,
                s_append,
                s_append2,
                self.parallel(),
                width,
                height,
                &mut session.pass_b,
            );
            debug_assert_eq!(v2a.len(), width * height);
            for (c, v) in canvas.iter_mut().zip(v2a.iter()) {
                *c += *v;
            }
            lattice = v2_lattice;
        }
        let t_pass_b = t2.elapsed();
        let t3 = crate::perf_clock::Instant::now();
        let out = AttributionResult::from_density(canvas, width, height).with_lattice(lattice);
        if perf_log {
            eprintln!(
                "ATTRPERF fused944: extraction+retention {:.1} ms | v1 walk+basic {:.1} ms | pass-B f32 {:.1} ms | trim+SAT {:.1} ms",
                t_extract.as_secs_f64() * 1e3,
                t_v1.as_secs_f64() * 1e3,
                t_pass_b.as_secs_f64() * 1e3,
                t3.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok((fb.result, v2res, out))
    }

    /// [`compute_folded944_score_and_attribution`](Self::compute_folded944_score_and_attribution)
    /// with a Level-2 binned map: both the fused v1 basic mass and the
    /// retention pass-B v2/append/append2 mass fold straight into ONE
    /// `bin × bin` grid — no full-resolution basic canvas, session pass-B
    /// canvas, trimmed clones, or full-resolution SAT. Score and 944
    /// features are bit-identical to the per-pixel entry (the sinks only
    /// receive map mass). `bin == 1` delegates to the per-pixel entry
    /// bit-identically.
    ///
    /// # Panics
    ///
    /// Panics if `bin == 0`.
    ///
    /// # Errors
    ///
    /// Same contract as the per-pixel folded-944 entry.
    #[cfg(feature = "feature-regime-v2")]
    pub fn compute_folded944_score_and_attribution_binned(
        &self,
        source: &impl ImageSource,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        session: &mut Fused944Session,
        bin: usize,
    ) -> Result<
        (
            crate::metric::ZensimResult,
            crate::feature_v2::ZensimV2Result,
            AttributionResult,
        ),
        ZensimError,
    > {
        validate_ref_match(precomputed, distorted)?;
        assert!(bin > 0, "bin must be non-zero");
        if bin == 1 {
            return self.compute_folded944_score_and_attribution(
                source,
                precomputed,
                distorted,
                s,
                session,
            );
        }
        session.basic.result = None;
        validate_pair(source, distorted)?;
        let v2res = crate::feature_v2::compute_folded944_streaming_with_retention(
            source,
            distorted,
            self.max_pixels(),
            self.parallel(),
            &mut session.scratch,
            &mut session.retention,
        )?;
        // Same identity disposition as the per-pixel entry above, at this
        // entry's map density.
        if crate::metric::images_byte_identical(source, distorted) {
            return Ok((
                self.compute(source, distorted)?,
                v2res,
                zero_attribution(distorted.width(), distorted.height(), bin),
            ));
        }
        let (result, attribution) = self.attribution_from_retention_binned(
            precomputed,
            distorted,
            s,
            &[],
            None,
            None,
            session,
            bin,
            None,
            crate::map_paint::MapPaint::default(),
        )?;
        Ok((result, v2res, attribution))
    }

    // Shared binned assembly for legacy folded-944 and complete candidates.
    // Extraction has already populated this session's retained cells.
    #[cfg(feature = "feature-regime-v2")]
    pub(crate) fn attribution_from_retention_binned(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        s: &[f64],
        s_peaks: &[f64],
        max_removals: Option<&mut Vec<MaxRemoval>>,
        moment_removals: Option<&mut Vec<MomentRemoval>>,
        session: &mut Fused944Session,
        bin: usize,
        revision: Option<crate::feature_defs::FormulaRevision>,
        paint: crate::map_paint::MapPaint,
    ) -> Result<(crate::ZensimResult, AttributionResult), ZensimError> {
        let width = distorted.width();
        let height = distorted.height();
        let mut accum = BinAccum::new_reusing(
            width,
            height,
            bin,
            &mut session.bin_scratch,
            self.parallel(),
        );
        let (result, _, _) = self.fused_basic_into_at_revision(
            precomputed,
            distorted,
            s,
            s_peaks,
            max_removals,
            None,
            &mut AttrSinkF32::Bins(&mut accum),
            revision,
            moment_removals,
            bin,
            if revision.is_some_and(|r| r >= crate::feature_defs::FormulaRevision::Rev5)
                && !session.retention.basic_stats.is_empty()
            {
                Some(BasicMapRetention::Fold(&session.retention))
            } else {
                session
                    .basic
                    .result
                    .as_ref()
                    .map(|_| BasicMapRetention::Basic(&session.basic))
            },
            Some(&mut session.basic_map),
            paint,
        )?;
        let block = |start: usize, end: usize| -> Option<&[f64]> {
            (s.len() > start).then(|| &s[start..s.len().min(end)])
        };
        let s_v2: &[f64] = block(BLOCK_END_V1_POOLS, BLOCK_END_V2).unwrap_or(&[]);
        let s_append: Option<&[f64]> = block(BLOCK_END_V2, BLOCK_END_APPEND);
        let s_append2: Option<&[f64]> = block(BLOCK_END_APPEND, BLOCK_END_APPEND2);
        if !s_v2.is_empty() || s_append.is_some() || s_append2.is_some() {
            crate::feature_v2::compute_v2_append_attribution_from_retention_into_bins(
                &session.retention,
                s_v2,
                s_append,
                s_append2,
                self.parallel(),
                &mut session.pass_b,
                &mut accum,
                precomputed.sampling_geometry.as_ref(),
                paint,
            );
        }
        // The globaff family (revision 6), over the pyramid planes its walk read.
        let first = crate::globaff::FIRST_ID;
        let s_globaff =
            block(first, first + crate::globaff::SLOTS).filter(|g| g.iter().any(|v| *v != 0.0));
        if let Some(s_globaff) = s_globaff {
            let rev = revision.unwrap_or_else(crate::ssim_form::active_revision);
            let dst = PrecomputedReference::new(
                distorted,
                precomputed.scales.len(),
                self.parallel(),
                rev,
            );
            let planes = crate::globaff::map_planes(precomputed, &dst, s_globaff, paint.split);
            let geometry = precomputed.sampling_geometry.as_ref();
            let guide0 = (paint.paint == crate::map_paint::PaintMode::Guided && geometry.is_none())
                .then(|| {
                    let (sp, w, h) = precomputed.scale(0);
                    let (dp, _, _) = dst.scale(0);
                    crate::map_paint::difference_guide(sp, dp, w * h, paint.guide)
                });
            for (scale, plane, w, h) in planes {
                if let Some(geometry) = geometry {
                    let projected = geometry.project(&plane, scale);
                    let (lw, lh) = geometry.logical_dimensions();
                    accum.add_scale_plane_f32(&projected, lw, lh, 1);
                } else {
                    accum.add_scale_plane_painted_f32(
                        &plane,
                        w,
                        h,
                        scale,
                        paint.paint,
                        guide0.as_deref(),
                    );
                }
            }
        }
        if paint.smooth_milli > 0 && precomputed.sampling_geometry.is_none() {
            accum.smooth_per_pixel(f64::from(paint.smooth_milli) / 1000.0);
        }
        Ok((result, accum.into_result_reusing(&mut session.bin_scratch)))
    }
}
