//! Streaming strip-plane producer for the folded-720+append walk.
//!
//! C1 of `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md` (plan:
//! `STREAMING_FOLDAPP_PLAN_2026-07-26.md`): produce, per pair, the exact
//! per-kernel-strip plane windows the v2 walk consumes — WITHOUT ever
//! materializing a full-image f32 plane on either side.
//!
//! # Architecture (design note §3)
//!
//! - **Rolling planes:** 2 sides × 3 channels × [`crate::NUM_SCALES`]
//!   scales of row-windowed buffers ([`RollingPlane`]: linear rows,
//!   `[lo, hi)` window, memmove compaction — kernels get ordinary
//!   contiguous `&[f32]` row slices, never ring indexing).
//! - **Scale-0 fill** converts source rows in chunks through
//!   [`crate::streaming::convert_source_to_xyb_into_slices`] with
//!   `abs_row_offset` so even translucent-alpha noise-background phase
//!   matches the whole-image conversion bit-for-bit.
//! - **Pyramid fill:** scale s+1 rows come from
//!   [`crate::blur::downscale_2x_into`] over even row pairs of scale s —
//!   the same arithmetic `build_v2_ref_scales` documents as bit-identical
//!   to the materialized walk's `downscale_2x_inplace`.
//! - **Per-scale kernel-strip cursors:** scale s emits its next
//!   [`STRIP_ROWS`]-row kernel strip (the 128-row tiling of the FULL
//!   plane, rooted at row 0 — identical to
//!   `compute_channel_scale_v2_with_fold`'s loop) as soon as the rolling
//!   window covers the strip's `HALO_P`-padded wide window. Strips of a
//!   scale always emit in ascending order; scales interleave (shallowest
//!   ready first) — irrelevant to parity because every consumer
//!   accumulates per-(scale, channel).
//!
//! # Parity contract (C1 unit gate)
//!
//! For every emitted strip, [`StripPlaneProducer::fill_wide`] writes ONE
//! channel-side's wide window with bytes EQUAL to
//! [`gather_strip_halo`]`(full_plane, …)` run against the materialized
//! pyramid of the same image — including `reflect_101` rows at the true
//! top/bottom. [`StripPlaneProducer::wide_window`] additionally returns
//! the window as a zero-copy slice when no reflection is needed
//! (interior strips — the fast path the C2 walk reads in place).

#[cfg(all(test, feature = "feature-regime-v2"))]
use crate::feature_v2::gather_strip_halo;
use crate::feature_v2::{HALO_P, HdrEncoding, STRIP_ROWS, reflect_101};
use crate::source::{ImageSource, PixelFormat, SubsetView};

/// Research toggle (`ZENSIM_PYRAMID=linear`): scales 1–3 from 2×2 averages of
/// linear RGB, each level converted to positive XYB (HaarXYB's pyramid),
/// instead of 2×2 averages of XYB. Scale 0 and every default path are
/// unchanged; only the streaming and block producers take it, and the
/// materialized pyramid builders refuse it. Unset or `xyb` keeps the default;
/// any other value panics. Exploration record:
/// `benchmarks/byconstruction_2026-10-22.md`, "Coarse scales in linear light".
pub(crate) fn linear_pyramid() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("ZENSIM_PYRAMID").as_deref() {
        Err(_) | Ok("xyb") => false,
        Ok("linear") => true,
        Ok(other) => panic!("ZENSIM_PYRAMID={other:?}: expected \"xyb\" or \"linear\""),
    })
}

/// Rows `y0 .. y0 + n` of an sRGB8 source as planar linear RGB (`out`: three
/// planes of `n · width`), through the sRGB8 table the conversion uses.
pub(crate) fn srgb8_rows_to_linear(
    src: &impl ImageSource,
    y0: usize,
    n: usize,
    width: usize,
    out: [&mut [f32]; 3],
) {
    assert!(
        src.pixel_format() == PixelFormat::Srgb8Rgb
            && src.color_primaries() == crate::source::ColorPrimaries::Srgb,
        "ZENSIM_PYRAMID=linear takes sRGB8 RGB sources"
    );
    let [r, g, b] = out;
    for k in 0..n {
        let row: &[[u8; 3]] = bytemuck::cast_slice(&src.row_bytes(y0 + k)[..width * 3]);
        for (x, p) in row.iter().enumerate() {
            let o = k * width + x;
            r[o] = crate::color::srgb_u8_to_linear(p[0]);
            g[o] = crate::color::srgb_u8_to_linear(p[1]);
            b[o] = crate::color::srgb_u8_to_linear(p[2]);
        }
    }
}

/// Positive XYB of `n` planar linear RGB rows (`lin`) into three XYB slices.
pub(crate) fn linear_rows_to_xyb(
    lin: [&[f32]; 3],
    n: usize,
    width: usize,
    out: [&mut [f32]; 3],
    revision: crate::feature_defs::FormulaRevision,
    buf: &mut Vec<[f32; 3]>,
) {
    let [r, g, b] = lin;
    let [x, y, bb] = out;
    for k in 0..n {
        let row = k * width..(k + 1) * width;
        buf.clear();
        buf.extend(row.clone().map(|i| [r[i], g[i], b[i]]));
        crate::color::linear_to_positive_xyb_planar_into(
            buf,
            &mut x[row.clone()],
            &mut y[row.clone()],
            &mut bb[row],
            revision,
        );
    }
}

/// Which front-end fills the scale-0 rolling planes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FrontEnd {
    /// The SDR path: `convert_source_to_xyb_into_slices` (sRGB → linear →
    /// opsin → cube-root XYB) — byte-identical to the pre-HDR producer.
    Sdr,
    /// The HDR path (HDR_PLAN chunk 2): pixel values → absolute cd/m² per
    /// the declared [`HdrEncoding`] → the UPIQ-validated PU-XYB front-end
    /// (`color::linear_to_pu_xyb_planar_into`: opsin mix → PU21
    /// banding_glare per channel ÷ PU21(100) → opponent axes, X×4).
    Hdr(HdrEncoding),
}

/// Rows appended to scale 0 per production step. A multiple of
/// `2^(NUM_SCALES-1)` keeps every deeper scale's fill cadence in whole
/// even row pairs (16 rows/step at scale 3), so downscale block parity
/// never depends on the step size.
/// Scale-0 rows converted per [`StripPlaneProducer::produce`] call.
///
/// **MUST stay a multiple of 64** — see [`CONVERT_CHUNK_ROWS`]: the producer's
/// conversion is byte-identical to the whole-image one only while every chunk
/// boundary lands on the same 64-row lattice, and `hi0` advances by exactly
/// this constant.
///
/// Raised 128 → 256 by the fold-MT lane. The conversion fans out at
/// [`CONVERT_CHUNK_ROWS`] rows per task, so this constant IS the producer
/// front end's parallel degree: at 128 a side was two chunks (four with both
/// sides concurrent) against a 16-thread pool, and the conversion was 23 % of
/// the fold-backed score's 16-thread wall at 2304². 256 doubles the chunk
/// count without moving a boundary. It costs rolling-plane capacity —
/// `scale_capacity_rows` budgets `advance >> scale` extra rows per scale —
/// which is measured in `benchmarks/fold_mt_scaling_2026-08-31.md`.
///
/// **Since the fold-footprint lane this is the CEILING, not the value.** A
/// producer takes [`advance_rows_for`]'s answer, which is 256 only on a pool
/// that can spend the degree; the capacity it costs is priced there.
const ADVANCE_ROWS: usize = 256;
const _: () = assert!(ADVANCE_ROWS.is_multiple_of(crate::streaming::DEFAULT_CONVERT_CHUNK_ROWS));

/// The advance actually used by one producer — [`ADVANCE_ROWS`] is its CEILING,
/// not its value (fold-footprint lane, `benchmarks/fold_footprint_2026-08-31.md`).
///
/// The constant exists to keep the conversion fan-out fed, and nothing else:
/// the two sides run concurrently and each fans out at [`CONVERT_CHUNK_ROWS`],
/// so one `produce()` call offers `2 * advance / CONVERT_CHUNK_ROWS` tasks and
/// `32 * threads` rows is the smallest advance that offers one per thread. On a
/// pool that cannot use them the extra rows buy nothing and are pure capacity:
/// `scale_capacity_rows` budgets `advance >> scale` extra rows at every scale,
/// which the footprint model prices at `24 * 1.328 * (advance - 64) * width`
/// bytes — **6.0 KiB per unit of width** between 64 and 256, independent of
/// height, i.e. 6.9 MB at 1152² and 13.7 MB at 2304² for a degree no
/// single-threaded walk can spend.
///
/// **Byte-neutral, and the existing suite is the gate.** Chunk HEIGHT is
/// semantics (`crate::streaming::tests::convert_chunk_rows_is_semantics_not_a_knob`
/// measures one ULP at 97×51); chunk COUNT is not, because every boundary still
/// lands on a global multiple of [`CONVERT_CHUNK_ROWS`] — which is why this
/// rounds UP to that lattice and asserts it. Within a scale, strips are still
/// emitted in strictly ascending order, and each scale owns its own
/// accumulator, so the cross-scale interleave the advance changes cannot reach
/// a sum. `fold_engine_parity` already sweeps rayon pools 1/2/3/8/16, so with
/// this function that sweep IS an advance sweep (64/64/128/256/256);
/// `producer_windows_are_advance_invariant` pins the producer directly.
fn advance_rows_for(parallel: bool, h0: usize) -> usize {
    #[cfg(feature = "threads")]
    let threads = if parallel {
        rayon::current_num_threads().max(1)
    } else {
        1
    };
    #[cfg(not(feature = "threads"))]
    let threads = {
        let _ = parallel;
        1usize
    };
    let want = threads
        .saturating_mul(CONVERT_CHUNK_ROWS / 2)
        .clamp(CONVERT_CHUNK_ROWS, ADVANCE_ROWS);
    // UP to the 64-row lattice — never below it, and never past the image.
    let a = want.div_ceil(CONVERT_CHUNK_ROWS) * CONVERT_CHUNK_ROWS;
    let h_lattice = h0.max(1).div_ceil(CONVERT_CHUNK_ROWS) * CONVERT_CHUNK_ROWS;
    let a = a.min(h_lattice);
    debug_assert_eq!(a % CONVERT_CHUNK_ROWS, 0);
    a
}

/// Rows per parallel chunk inside the producer's scale-0 XYB conversion.
///
/// **PINNED to `streaming`'s default, and it is not free to move.** Chunk
/// height is SEMANTICS: which elements land in a per-pixel kernel's SIMD body
/// versus its scalar tail depends on the buffer length, and the two disagree
/// in the last bit
/// (`crate::streaming::tests::convert_chunk_rows_is_semantics_not_a_knob`
/// measures one ULP at 97×51). The streaming and materialised conversions are
/// byte-identical only because every producer chunk boundary lands on a global
/// multiple of 64 rows, which is where the whole-image call cuts.
///
/// The fold-MT lane wanted a smaller height here — the producer hands this
/// function one advance's worth of rows, so at 64 a side is two chunks and the
/// front end caps at degree 2 (4 with both sides concurrent) — and took the
/// ADVANCE up instead, which raises the chunk COUNT while keeping every
/// boundary on the 64-row lattice. [`advance_rows_for`] then made that a
/// per-pool decision rather than a constant.
const CONVERT_CHUNK_ROWS: usize = crate::streaming::DEFAULT_CONVERT_CHUNK_ROWS;

/// One side (source or distorted) of the pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Source,
    Distorted,
}

/// A row-windowed plane buffer: rows `[lo, hi)` of a conceptual
/// `width × height` plane, stored contiguously starting at buffer row
/// `base` (compacted to 0 whenever appending would overflow capacity).
struct RollingPlane {
    buf: Vec<f32>,
    width: usize,
    /// First plane row currently retained.
    lo: usize,
    /// One past the last plane row produced.
    hi: usize,
    /// Buffer row index where plane row `lo` lives.
    base: usize,
}

impl RollingPlane {
    /// Build from a recycled buffer (see [`StripPlaneProducer::recycle`])
    /// — resized up if the pooled buffer is too small; contents are
    /// irrelevant (every row is fully written before it is read).
    fn from_pooled(mut buf: Vec<f32>, width: usize, cap_rows: usize) -> Self {
        let need = width * cap_rows;
        if buf.len() < need {
            // FRESH (empty pool) **or TOO SMALL** — one branch, because the
            // right move is the same: `vec![0.0; n]` lowers to `calloc`,
            // which hands back demand-zeroed pages the allocator can satisfy
            // lazily.
            //
            // `Vec::resize` cannot: it writes the fill value with a real
            // memset, and on a too-small NON-empty pooled buffer it also
            // reallocs (copying forward contents this type's own contract
            // says are irrelevant — `lo == hi == 0`, and every row is fully
            // written by `append_rows` before any reader can name it).
            //
            // MEASURED (callgrind, 576², `156` arm, 2 walks, v3 tier), the
            // grow path's cost split — the FILL dominates, the copy does not:
            //   __memset_avx2_unaligned_erms   4,112,820 Ir (12 calls)
            //   RawVecInner::…do_reserve…        733,763 Ir (12 calls)
            // i.e. 1.17 % of the whole program spent zeroing bytes that are
            // about to be overwritten, plus 0.21 % moving them first. The
            // pool is LIFO and holds planes of every scale, so a scale-3
            // buffer being popped for a scale-0 plane is the COMMON case,
            // not a corner. After this change `__memset_avx2_unaligned_erms`
            // across the whole run falls 4,695,365 → 582,545 (−87.6 %).
            //
            // Strictly more deterministic, too: this always yields zeros,
            // where `resize` left a prefix of the previous walk's pixels.
            crate::fold_timing::work(crate::fold_timing::Work::ScratchZeroElements, need);
            buf = vec![0.0; need];
        }
        Self {
            buf,
            width,
            lo: 0,
            hi: 0,
            base: 0,
        }
    }

    #[inline]
    fn cap_rows(&self) -> usize {
        self.buf.len() / self.width
    }

    /// Contiguous rows `[r0, r1)`. Panics (debug) if outside the window.
    #[inline]
    fn rows(&self, r0: usize, r1: usize) -> &[f32] {
        debug_assert!(
            self.lo <= r0 && r0 <= r1 && r1 <= self.hi,
            "rows [{r0}, {r1}) outside window [{}, {}) (base {})",
            self.lo,
            self.hi,
            self.base
        );
        let start = (self.base + (r0 - self.lo)) * self.width;
        &self.buf[start..start + (r1 - r0) * self.width]
    }

    /// Make room for `n_rows` new rows at `hi` and return the writable
    /// region for them (compacting or growing first as needed). Caller
    /// fills every element of the returned slice.
    fn append_rows(&mut self, n_rows: usize) -> &mut [f32] {
        let held = self.hi - self.lo;
        if self.base + held + n_rows > self.cap_rows() {
            // Compact the live window to the buffer front.
            self.buf
                .copy_within(self.base * self.width..(self.base + held) * self.width, 0);
            self.base = 0;
            if held + n_rows > self.cap_rows() {
                // Capacity fallback — correctness-safe, should not happen
                // with the sizing in `scale_capacity_rows` (asserted by
                // the C1 tests via `max_held_rows`).
                self.buf.resize((held + n_rows) * self.width, 0.0);
            }
        }
        let start = (self.base + held) * self.width;
        self.hi += n_rows;
        &mut self.buf[start..start + n_rows * self.width]
    }

    /// Raise `lo` (retire rows below `new_lo`). No-op if already ≥.
    fn retire_below(&mut self, new_lo: usize) {
        let new_lo = new_lo.min(self.hi);
        if new_lo > self.lo {
            self.base += new_lo - self.lo;
            self.lo = new_lo;
        }
    }
}

/// Identity of one ready kernel strip.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StripInfo {
    pub scale: usize,
    /// Strip's first plane row (`ks_idx * STRIP_ROWS`).
    pub y0: usize,
    /// Rows in the strip (`STRIP_ROWS`, truncated at the plane bottom).
    pub strip_h: usize,
    /// Plane dims at this scale.
    pub plane_w: usize,
    pub plane_h: usize,
}

impl StripInfo {
    /// Buffer height of the strip's `HALO_P`-padded wide window.
    #[inline]
    pub fn wide_h(&self) -> usize {
        self.strip_h + 2 * HALO_P
    }

    /// True when the wide window needs no boundary reflection — i.e.
    /// `[y0 − HALO_P, y0 + strip_h + HALO_P)` lies fully inside the
    /// plane, so it exists as contiguous real rows.
    #[inline]
    pub fn interior(&self) -> bool {
        self.y0 >= HALO_P && self.y0 + self.strip_h + HALO_P <= self.plane_h
    }
}

/// Per-scale production/emission cursors.
struct ScaleState {
    plane_w: usize,
    plane_h: usize,
    /// Next kernel strip to emit (index into the 128-row tiling).
    next_ks: usize,
}

/// The streaming producer: rolling XYB pyramids for both sides of a pair
/// plus the per-scale kernel-strip emission cursors.
pub(crate) struct StripPlaneProducer<'a, S: ImageSource, D: ImageSource> {
    source: &'a S,
    distorted: &'a D,
    sampled: Option<[Vec<crate::streaming::XybPyramidLevel>; 2]>,
    omit_scale0_xb: bool,
    /// `[side][channel][scale]` rolling planes; side 0 = source.
    planes: [[Vec<RollingPlane>; 3]; 2],
    scales: Vec<ScaleState>,
    parallel: bool,
    front_end: FrontEnd,
    /// Row scratch for the HDR front-end (decode + PU conversion input).
    hdr_row: Vec<[f32; 3]>,
    /// Peak `hi − lo` observed per scale (capacity-budget telemetry for
    /// the C1 tests; also what G-RAM's O(width) claim rests on).
    max_held_rows: Vec<usize>,
    /// REF-CACHED FEED (fold-engine lane, stage 3). When `Some`, the SOURCE
    /// side's rolling planes are filled by COPYING from an already-built XYB
    /// pyramid instead of decoding + converting scale 0 and downscaling the
    /// cascade — the streaming counterpart of `Zensim::precompute_reference`'s
    /// amortisation across N distorted candidates.
    ///
    /// Bit-exactness is BY CONSTRUCTION and is gated, not argued: the cache is
    /// built by `convert_source_to_xyb` + `downscale_2x_into`, the same two
    /// functions this producer uses (`producer_windows_byte_equal_materialized`
    /// pins the producer against exactly that materialisation), so the bytes
    /// copied in are the bytes the conversion would have produced. The
    /// distorted side is untouched.
    ///
    /// `[XybPyramidLevel; NUM_SCALES]`, borrowed from the caller's
    /// `PrecomputedReference`. The caller guarantees `len() == NUM_SCALES` and
    /// that every level's `(width, height)` matches this producer's — see
    /// `feature_v2::cached_ref_feed_usable`.
    ref_planes: Option<&'a [crate::streaming::XybPyramidLevel]>,
    /// Scale-0 rows converted per [`Self::produce`] call, and hence this
    /// producer's rolling-window capacity — see [`advance_rows_for`].
    advance_rows: usize,
    /// The walk's formula revision — what the scale-0 opsin conversion runs
    /// (canonical at Rev4, featcanon D1).
    revision: crate::feature_defs::FormulaRevision,
    /// [`linear_pyramid`]: `[side][R, G, B][scale]` linear rolling planes,
    /// the parents of every coarse level, and a row of interleaved pixels.
    lin: Option<Box<[[Vec<RollingPlane>; 3]; 2]>>,
    lin_buf: Vec<[f32; 3]>,
}

/// Convert `n_new` scale-0 rows of one side into that side's three rolling
/// planes — the body of the producer's per-side conversion, lifted to a free
/// function so the two sides can run as a `rayon::join` (fold-MT lane). SDR
/// route only; the HDR route keeps the in-method serial loop because it shares
/// one `hdr_row` scratch.
///
/// Byte-neutral by construction: it is the same
/// [`crate::streaming::convert_source_to_xyb_into_slices`] call the serial arm
/// makes, on the same [`SubsetView`], with the same `abs_row_offset` — only
/// which thread runs it changes.
#[cfg(feature = "threads")]
fn convert_side_scale0(
    src: &impl ImageSource,
    planes: &mut [Vec<RollingPlane>; 3],
    hi0: usize,
    n_new: usize,
    width: usize,
    parallel: bool,
    revision: crate::feature_defs::FormulaRevision,
) {
    let [c0, c1, c2] = planes;
    let (p0, p1, p2) = (
        c0[0].append_rows(n_new),
        c1[0].append_rows(n_new),
        c2[0].append_rows(n_new),
    );
    let sub = SubsetView::new(src, hi0, n_new);
    crate::streaming::convert_source_to_xyb_into_slices_chunked(
        &sub,
        p0,
        p1,
        p2,
        width,
        parallel,
        hi0,
        CONVERT_CHUNK_ROWS,
        revision,
    );
}

/// Rev5 unread X/B scale zero: retain two rows only, immediately downscale.
/// Canonical color conversion is pointwise, including its partial vector tail.
fn convert_side_y_and_downscale(
    source: &impl ImageSource,
    planes: &mut [Vec<RollingPlane>; 3],
    hi0: usize,
    n_new: usize,
    width: usize,
    revision: crate::feature_defs::FormulaRevision,
    #[allow(unused_variables)] parallel: bool,
) {
    #[cfg(feature = "threads")]
    if parallel
        && convert_side_y_and_downscale_parallel(source, planes, hi0, n_new, width, revision)
    {
        return;
    }
    let [x, y, b] = planes;
    for row in (hi0..hi0 + n_new).step_by(2) {
        let rows = (hi0 + n_new - row).min(2);
        let n = width * rows;
        let sub = SubsetView::new(source, row, rows);
        crate::streaming::convert_source_to_xyb_into_slices(
            &sub,
            &mut x[0].buf[..n],
            y[0].append_rows(rows),
            &mut b[0].buf[..n],
            width,
            false,
            row,
            revision,
        );
        if rows == 2 {
            for ch in [&mut *x, &mut *b] {
                let (head, tail) = ch.split_at_mut(1);
                crate::blur::downscale_2x_into(
                    &head[0].buf[..n],
                    width,
                    tail[0].append_rows(1),
                    width / 2,
                    1,
                );
            }
        }
    }
    // The cursor participates in readiness/retirement; its pixels are never read.
    x[0].hi = hi0 + n_new;
    b[0].hi = hi0 + n_new;
}

/// Restore the producer's existing row-chunk concurrency without retaining
/// full-resolution chroma. Each task runs the same two-row calls and owns
/// disjoint Y/coarse-X/coarse-B output rows and two-row chroma scratch.
#[cfg(feature = "threads")]
fn convert_side_y_and_downscale_parallel(
    source: &impl ImageSource,
    planes: &mut [Vec<RollingPlane>; 3],
    hi0: usize,
    n_new: usize,
    width: usize,
    revision: crate::feature_defs::FormulaRevision,
) -> bool {
    use rayon::prelude::*;
    let chunks = n_new / CONVERT_CHUNK_ROWS;
    if chunks < 2 {
        return false;
    }
    let Some(scratch_len) = width.checked_mul(2) else {
        return false;
    };
    let mut scratch = Vec::new();
    let mut work = Vec::new();
    // All fallible allocation precedes cursor/output mutation. Under memory
    // pressure the unchanged serial path can use its existing two-row buffers.
    if scratch.try_reserve_exact(chunks).is_err() || work.try_reserve_exact(chunks).is_err() {
        return false;
    }
    for _ in 0..chunks {
        let (mut x, mut b) = (Vec::new(), Vec::new());
        if x.try_reserve_exact(scratch_len).is_err() || b.try_reserve_exact(scratch_len).is_err() {
            return false;
        }
        x.resize(scratch_len, 0.0);
        b.resize(scratch_len, 0.0);
        scratch.push((x, b));
    }
    let [x, y, b] = planes;
    let mut ys = y[0].append_rows(n_new);
    let mut xs = x[1].append_rows(n_new / 2);
    let mut bs = b[1].append_rows(n_new / 2);
    let coarse_width = width / 2;
    for (chunk, (x_scratch, b_scratch)) in scratch.into_iter().enumerate() {
        // Keep a possible odd final row in the last task, including when
        // only one row remains after a lattice boundary. No empty coarse
        // slice is dropped by an iterator zip.
        let offset = chunk * CONVERT_CHUNK_ROWS;
        let rows = if chunk + 1 == chunks {
            n_new - offset
        } else {
            CONVERT_CHUNK_ROWS
        };
        let (yc, y_tail) = ys.split_at_mut(rows * width);
        let (xc, x_tail) = xs.split_at_mut((rows / 2) * coarse_width);
        let (bc, b_tail) = bs.split_at_mut((rows / 2) * coarse_width);
        work.push((hi0 + offset, yc, xc, bc, x_scratch, b_scratch));
        (ys, xs, bs) = (y_tail, x_tail, b_tail);
    }
    work.into_par_iter()
        .for_each(|(row0, ys, xs, bs, mut x, mut b)| {
            let rows = ys.len() / width;
            for dy in (0..rows).step_by(2) {
                let count = (rows - dy).min(2);
                let n = width * count;
                let sub = SubsetView::new(source, row0 + dy, count);
                crate::streaming::convert_source_to_xyb_into_slices(
                    &sub,
                    &mut x[..n],
                    &mut ys[dy * width..dy * width + n],
                    &mut b[..n],
                    width,
                    false,
                    row0 + dy,
                    revision,
                );
                if count == 2 {
                    let out = (dy / 2) * coarse_width;
                    crate::blur::downscale_2x_into(
                        &x[..n],
                        width,
                        &mut xs[out..out + coarse_width],
                        coarse_width,
                        1,
                    );
                    crate::blur::downscale_2x_into(
                        &b[..n],
                        width,
                        &mut bs[out..out + coarse_width],
                        coarse_width,
                        1,
                    );
                }
            }
        });
    x[0].hi = hi0 + n_new;
    b[0].hi = hi0 + n_new;
    true
}

/// Advance the source window over the caller's immutable cached XYB rows.
/// The row accessor borrows that storage directly.
#[cfg(feature = "threads")]
fn advance_cached_scale0(
    rp: &[crate::streaming::XybPyramidLevel],
    planes: &mut [Vec<RollingPlane>; 3],
    hi0: usize,
    n_new: usize,
    width: usize,
) {
    let (cached, cw, _) = &rp[0];
    debug_assert_eq!(*cw, width, "cached ref stride must equal the plane width");
    for (ch, plane) in planes.iter_mut().enumerate() {
        debug_assert!(cached[ch].len() >= (hi0 + n_new) * width);
        plane[0].hi = hi0 + n_new;
    }
}

/// Capacity (in rows) budgeted for one scale's rolling window: one
/// kernel strip + both halos + one production step + slack for the
/// downscale-consumption floor. Exceeding it merely grows the buffer
/// (correctness-safe); the C1 tests assert it is never exceeded.
fn scale_capacity_rows(scale: usize, plane_h: usize, advance_rows: usize) -> usize {
    let step = (advance_rows >> scale).max(2);
    (STRIP_ROWS + 2 * HALO_P + step + 32).min(plane_h.max(1) + 2 * HALO_P)
}

impl<'a, S: ImageSource, D: ImageSource> StripPlaneProducer<'a, S, D> {
    /// Build a producer for a dimension-matched pair. Caller has already
    /// applied entry-point validation (dims match, ≥ MIN_PYRAMID_DIM —
    /// sub-64 inputs are reflect-padded BEFORE constructing, exactly like
    /// the materialized entries).
    /// `pool` recycles rolling-plane buffers across pairs (the caller's
    /// [`crate::feature_v2::V2Scratch`] owns it): construction drains it,
    /// [`Self::recycle`] refills it — steady-state batch extraction
    /// performs zero producer allocation (measured ~1.7k page
    /// faults/pair from fresh per-pair Vecs before this).
    #[cfg(feature = "feature-regime-v2")]
    #[cfg_attr(not(test), allow(dead_code))] // research/test constructor
    pub(crate) fn new(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
    ) -> Self {
        Self::new_with_front_end(source, distorted, parallel, pool, FrontEnd::Sdr)
    }

    /// Test/instrument constructor: the walk runs at the process revision.
    #[cfg(feature = "feature-regime-v2")]
    pub(crate) fn new_with_front_end(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
        front_end: FrontEnd,
    ) -> Self {
        Self::new_with_ref_feed(
            source,
            distorted,
            parallel,
            pool,
            front_end,
            None,
            None,
            crate::ssim_form::active_revision(),
            false,
        )
    }

    /// [`Self::new_with_front_end`] with an optional pre-built source-side XYB
    /// pyramid (see the `ref_planes` field). `None` is byte-for-byte the
    /// non-cached constructor. `revision` is the walk's formula revision.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_ref_feed(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
        front_end: FrontEnd,
        ref_planes: Option<&'a [crate::streaming::XybPyramidLevel]>,
        sampling: Option<crate::sampling::Sampling>,
        revision: crate::feature_defs::FormulaRevision,
        omit_scale0_xb: bool,
    ) -> Self {
        if let Some(sampling) = sampling {
            assert!(matches!(front_end, FrontEnd::Sdr));
            assert!(
                !linear_pyramid(),
                "ZENSIM_PYRAMID=linear does not cover sampled walks"
            );
            let make_pair = || {
                [
                    sampling.pyramid(source, parallel, revision),
                    sampling.pyramid(distorted, parallel, revision),
                ]
            };
            #[cfg(feature = "threads")]
            let sampled = if parallel
                && source.width() * source.height() >= 65_536
                && rayon::current_num_threads() > 1
            {
                let (a, b) = rayon::join(
                    || sampling.pyramid(source, parallel, revision),
                    || sampling.pyramid(distorted, parallel, revision),
                );
                [a, b]
            } else {
                make_pair()
            };
            #[cfg(not(feature = "threads"))]
            let sampled = make_pair();
            let scales = sampled[0]
                .iter()
                .map(|p| ScaleState {
                    plane_w: p.1,
                    plane_h: p.2,
                    next_ks: 0,
                })
                .collect();
            return Self {
                source,
                distorted,
                sampled: Some(sampled),
                omit_scale0_xb: false,
                planes: std::array::from_fn(|_| std::array::from_fn(|_| Vec::new())),
                scales,
                parallel,
                front_end,
                hdr_row: Vec::new(),
                max_held_rows: vec![0; crate::NUM_SCALES],
                ref_planes: None,
                advance_rows: 0,
                revision,
                lin: None,
                lin_buf: Vec::new(),
            };
        }
        Self::new_inner(
            source,
            distorted,
            parallel,
            pool,
            front_end,
            ref_planes,
            None,
            revision,
            omit_scale0_xb,
        )
    }

    /// [`Self::new`] with the production advance replaced — the injection point
    /// for `producer_windows_are_advance_invariant`, which is the direct gate
    /// on [`advance_rows_for`] being a schedule knob and not semantics.
    #[cfg(all(test, feature = "feature-regime-v2"))]
    pub(crate) fn new_with_advance(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
        advance_rows: usize,
    ) -> Self {
        Self::new_inner(
            source,
            distorted,
            parallel,
            pool,
            FrontEnd::Sdr,
            None,
            Some(advance_rows),
            crate::ssim_form::active_revision(),
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        source: &'a S,
        distorted: &'a D,
        parallel: bool,
        pool: &mut Vec<Vec<f32>>,
        front_end: FrontEnd,
        ref_planes: Option<&'a [crate::streaming::XybPyramidLevel]>,
        advance_override: Option<usize>,
        revision: crate::feature_defs::FormulaRevision,
        omit_scale0_xb: bool,
    ) -> Self {
        let linear = linear_pyramid();
        assert!(
            !linear || (matches!(front_end, FrontEnd::Sdr) && ref_planes.is_none()),
            "ZENSIM_PYRAMID=linear supports the SDR walk without a cached reference"
        );
        // The linear pyramid converts every level itself: scale-0 chroma is kept.
        let omit_scale0_xb = omit_scale0_xb
            && !linear
            && revision >= crate::feature_defs::FormulaRevision::Rev5
            && matches!(front_end, FrontEnd::Sdr);
        let (w0, h0) = (source.width(), source.height());
        debug_assert_eq!(w0, distorted.width());
        debug_assert_eq!(h0, distorted.height());

        let mut scales = Vec::with_capacity(crate::NUM_SCALES);
        let (mut w, mut h) = (w0, h0);
        for _ in 0..crate::NUM_SCALES {
            scales.push(ScaleState {
                plane_w: w,
                plane_h: h,
                next_ks: 0,
            });
            w /= 2;
            h /= 2;
        }

        let advance_rows = advance_override.unwrap_or_else(|| advance_rows_for(parallel, h0));
        assert_eq!(
            advance_rows % CONVERT_CHUNK_ROWS,
            0,
            "producer advance must keep every conversion chunk boundary on the \
             {CONVERT_CHUNK_ROWS}-row lattice (chunk HEIGHT is semantics)"
        );
        let planes = std::array::from_fn(|_| {
            std::array::from_fn(|ch| {
                scales
                    .iter()
                    .enumerate()
                    .map(|(s, st)| {
                        RollingPlane::from_pooled(
                            pool.pop().unwrap_or_default(),
                            st.plane_w,
                            if omit_scale0_xb && s == 0 && ch != 1 {
                                2
                            } else {
                                scale_capacity_rows(s, st.plane_h, advance_rows)
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            })
        });

        let lin = linear.then(|| {
            Box::new(std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    scales
                        .iter()
                        .enumerate()
                        .map(|(s, st)| {
                            RollingPlane::from_pooled(
                                pool.pop().unwrap_or_default(),
                                st.plane_w,
                                scale_capacity_rows(s, st.plane_h, advance_rows),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            }))
        });

        Self {
            source,
            distorted,
            sampled: None,
            omit_scale0_xb,
            planes,
            scales,
            parallel,
            front_end,
            hdr_row: match front_end {
                FrontEnd::Sdr => Vec::new(),
                FrontEnd::Hdr(_) => vec![[0.0f32; 3]; w0],
            },
            max_held_rows: vec![0; crate::NUM_SCALES],
            ref_planes,
            advance_rows,
            revision,
            lin,
            lin_buf: Vec::new(),
        }
    }

    #[inline]
    fn plane(&self, side: Side, ch: usize, scale: usize) -> &RollingPlane {
        let si = match side {
            Side::Source => 0,
            Side::Distorted => 1,
        };
        &self.planes[si][ch][scale]
    }

    /// Emit the next ready kernel strip (shallowest ready scale first;
    /// strictly ascending strip order within a scale), producing more
    /// pyramid rows as needed. Returns `None` when every kernel strip of
    /// every scale has been emitted. The returned windows (via
    /// [`Self::wide_window`] / [`Self::fill_wide`] / [`Self::rows`]) stay
    /// valid until the NEXT `next_strip` call.
    pub(crate) fn next_strip(&mut self) -> Option<StripInfo> {
        if self.sampled.is_some() {
            for (scale, st) in self.scales.iter_mut().enumerate() {
                let y0 = st.next_ks * STRIP_ROWS;
                if y0 < st.plane_h {
                    st.next_ks += 1;
                    return Some(StripInfo {
                        scale,
                        y0,
                        strip_h: STRIP_ROWS.min(st.plane_h - y0),
                        plane_w: st.plane_w,
                        plane_h: st.plane_h,
                    });
                }
            }
            return None;
        }
        // Retire rows consumed by everything emitted so far, then look
        // for a ready strip; if none, produce more rows and repeat.
        loop {
            self.retire();
            if let Some(info) = self.find_ready() {
                self.scales[info.scale].next_ks += 1;
                return Some(info);
            }
            if !self.produce() {
                debug_assert!(
                    self.scales
                        .iter()
                        .all(|st| { st.next_ks * STRIP_ROWS >= st.plane_h }),
                    "production exhausted with unemitted strips"
                );
                return None;
            }
        }
    }

    /// Shallowest scale with a ready, unemitted kernel strip.
    fn find_ready(&self) -> Option<StripInfo> {
        for (scale, st) in self.scales.iter().enumerate() {
            let y0 = st.next_ks * STRIP_ROWS;
            if y0 >= st.plane_h {
                continue; // scale fully emitted
            }
            let strip_h = STRIP_ROWS.min(st.plane_h - y0);
            let need_hi = (y0 + strip_h + HALO_P).min(st.plane_h);
            let hi = self.plane(Side::Source, 0, scale).hi;
            if hi >= need_hi {
                return Some(StripInfo {
                    scale,
                    y0,
                    strip_h,
                    plane_w: st.plane_w,
                    plane_h: st.plane_h,
                });
            }
        }
        None
    }

    /// Advance production: convert up to `self.advance_rows` new scale-0
    /// rows on both sides and cascade downscales. Returns false when
    /// scale 0 was already complete (nothing produced).
    fn produce(&mut self) -> bool {
        let h0 = self.scales[0].plane_h;
        let hi0 = self.planes[0][0][0].hi;
        if hi0 >= h0 {
            return false;
        }
        let n_new = self.advance_rows.min(h0 - hi0);

        if !self.omit_scale0_xb {
            crate::fold_timing::work(crate::fold_timing::Work::Scale0XbStoredRows, 4 * n_new);
        }
        // --- Scale-0 conversion, both sides. ---
        //
        // MT (fold-MT lane): the two sides write DISJOINT plane sets and read
        // disjoint sources, so running them concurrently is a pure schedule
        // change — no accumulation, no shared scratch, not one byte moved.
        // It is worth doing because the inner fan-out inside
        // `convert_source_to_xyb_into_slices` chunks at 64 rows and this call
        // hands it `self.advance_rows` rows — as few as TWO chunks on a small
        // pool ([`advance_rows_for`]); two sides in parallel double that.
        // The HDR route keeps the serial loop — it shares one `hdr_row`
        // scratch, and a second buffer is not this lane's measurement.
        let __t_conv = crate::fold_timing::start();
        if self.omit_scale0_xb {
            let width = self.scales[0].plane_w;
            let source = self.source;
            let distorted = self.distorted;
            let revision = self.revision;
            let parallel = self.parallel;
            let ref_planes = self.ref_planes;
            let (head, tail) = self.planes.split_at_mut(1);
            let (sp, dp) = (&mut head[0], &mut tail[0]);
            let mut convert_source = || {
                if let Some(rp) = ref_planes {
                    // Scale-zero chroma is unread. Feed the cached coarse planes
                    // directly rather than copying full-resolution X/B first.
                    debug_assert!(rp[0].0[1].len() >= (hi0 + n_new) * width);
                    sp[1][0].hi = hi0 + n_new;
                    for ch in [0, 2] {
                        sp[ch][0].hi = hi0 + n_new;
                    }
                } else {
                    convert_side_y_and_downscale(source, sp, hi0, n_new, width, revision, parallel);
                }
            };
            let mut convert_distorted = || {
                convert_side_y_and_downscale(distorted, dp, hi0, n_new, width, revision, parallel)
            };
            // Each side owns its two-row chroma scratch and rolling planes.
            // Keep the pointwise conversion/downscale calls intact; only their
            // schedule changes, as in the full scale-zero producer below.
            #[cfg(feature = "threads")]
            if self.parallel {
                rayon::join(convert_source, convert_distorted);
            } else {
                convert_source();
                convert_distorted();
            }
            #[cfg(not(feature = "threads"))]
            {
                convert_source();
                convert_distorted();
            }
        } else {
            let front_end = self.front_end;
            #[cfg(feature = "threads")]
            let sides_parallel = self.parallel && matches!(front_end, FrontEnd::Sdr);
            #[cfg(not(feature = "threads"))]
            let sides_parallel = false;
            #[cfg(feature = "threads")]
            if sides_parallel {
                let width = self.scales[0].plane_w;
                let source = self.source;
                let distorted = self.distorted;
                let ref_planes = self.ref_planes;
                let inner_parallel = self.parallel;
                let revision = self.revision;
                let (h, t) = self.planes.split_at_mut(1);
                let (sp, dp) = (&mut h[0], &mut t[0]);
                rayon::join(
                    || match ref_planes {
                        Some(rp) => advance_cached_scale0(rp, sp, hi0, n_new, width),
                        None => convert_side_scale0(
                            source,
                            sp,
                            hi0,
                            n_new,
                            width,
                            inner_parallel,
                            revision,
                        ),
                    },
                    || {
                        convert_side_scale0(
                            distorted,
                            dp,
                            hi0,
                            n_new,
                            width,
                            inner_parallel,
                            revision,
                        )
                    },
                );
            }
            for si in (0..2).take(if sides_parallel { 0 } else { 2 }) {
                let width = self.scales[0].plane_w;
                // REF-CACHED FEED: the source side's scale-0 rows already exist
                // in the caller's pyramid — borrow them instead of decoding and
                // converting. Bytes are identical by construction (see the
                // `ref_planes` field doc); the distorted side never takes this
                // branch.
                if si == 0
                    && let Some(rp) = self.ref_planes
                {
                    let (planes, cw, _) = &rp[0];
                    debug_assert_eq!(*cw, width, "cached ref stride must equal the plane width");
                    let [c0, c1, c2] = &mut self.planes[0];
                    for (ch, plane) in [c0, c1, c2].into_iter().enumerate() {
                        debug_assert!(planes[ch].len() >= (hi0 + n_new) * width);
                        plane[0].hi = hi0 + n_new;
                    }
                    continue;
                }
                let [c0, c1, c2] = &mut self.planes[si];
                let (p0, p1, p2) = (
                    c0[0].append_rows(n_new),
                    c1[0].append_rows(n_new),
                    c2[0].append_rows(n_new),
                );
                match front_end {
                    FrontEnd::Sdr => match si {
                        0 => {
                            let sub = SubsetView::new(self.source, hi0, n_new);
                            crate::streaming::convert_source_to_xyb_into_slices(
                                &sub,
                                p0,
                                p1,
                                p2,
                                width,
                                self.parallel,
                                hi0,
                                self.revision,
                            );
                        }
                        _ => {
                            let sub = SubsetView::new(self.distorted, hi0, n_new);
                            crate::streaming::convert_source_to_xyb_into_slices(
                                &sub,
                                p0,
                                p1,
                                p2,
                                width,
                                self.parallel,
                                hi0,
                                self.revision,
                            );
                        }
                    },
                    FrontEnd::Hdr(encoding) => {
                        for k in 0..n_new {
                            let y = hi0 + k;
                            let row_off = k * width;
                            if si == 0 {
                                hdr_source_row_to_nits(
                                    self.source,
                                    y,
                                    encoding,
                                    &mut self.hdr_row,
                                    self.revision,
                                );
                            } else {
                                hdr_source_row_to_nits(
                                    self.distorted,
                                    y,
                                    encoding,
                                    &mut self.hdr_row,
                                    self.revision,
                                );
                            }
                            crate::color::linear_to_pu_xyb_planar_into_at_revision(
                                &self.hdr_row[..width],
                                &mut p0[row_off..row_off + width],
                                &mut p1[row_off..row_off + width],
                                &mut p2[row_off..row_off + width],
                                self.revision,
                            );
                        }
                    }
                }
            }
        }
        crate::fold_timing::stop(__t_conv, crate::fold_timing::Phase::ProdConvert, 0);

        // `ZENSIM_PYRAMID=linear`: the same rows in linear RGB, the parents of
        // every coarse level.
        if let Some(lin) = self.lin.as_deref_mut() {
            let width = self.scales[0].plane_w;
            for (si, side) in lin.iter_mut().enumerate() {
                let [r, g, b] = side;
                let out = [
                    r[0].append_rows(n_new),
                    g[0].append_rows(n_new),
                    b[0].append_rows(n_new),
                ];
                if si == 0 {
                    srgb8_rows_to_linear(self.source, hi0, n_new, width, out);
                } else {
                    srgb8_rows_to_linear(self.distorted, hi0, n_new, width, out);
                }
            }
        }

        // --- Cascade downscales: fill each deeper scale from complete
        //     even row pairs of its parent. ---
        //
        // MT (fold-MT lane): within ONE scale the six (side, channel) planes
        // are fully independent — each reads its own parent plane and writes
        // its own child plane, and `downscale_2x_into` is a pure function of
        // the rows handed to it. Scales stay SEQUENTIAL because scale `s`
        // consumes rows scale `s-1` produced in this same call. So the fan-out
        // is 6-way per scale, which is a pure schedule change: no value is
        // summed across the boundary, no byte moves.
        let __t_ds = crate::fold_timing::start();
        if let Some(lin) = self.lin.as_deref_mut() {
            // Each coarse level: 2×2 averages of the parent's linear rows (the
            // XYB cascade's `downscale_2x_into` and row targets), then positive
            // XYB of the new rows.
            for scale in 1..crate::NUM_SCALES {
                let (plane_w, plane_h) = (self.scales[scale].plane_w, self.scales[scale].plane_h);
                let parent_w = self.scales[scale - 1].plane_w;
                for (side, planes) in lin.iter_mut().zip(self.planes.iter_mut()) {
                    let child_hi = side[0][scale].hi;
                    let target = (side[0][scale - 1].hi / 2).min(plane_h);
                    if target <= child_hi {
                        continue;
                    }
                    let n_out = target - child_hi;
                    for ch in side.iter_mut() {
                        let (head, tail) = ch.split_at_mut(scale);
                        let src_rows = head[scale - 1].rows(child_hi * 2, target * 2);
                        let dst = tail[0].append_rows(n_out);
                        crate::blur::downscale_2x_into(src_rows, parent_w, dst, plane_w, n_out);
                    }
                    let [r, g, b] = &*side;
                    let [x, y, bb] = planes;
                    linear_rows_to_xyb(
                        [
                            r[scale].rows(child_hi, target),
                            g[scale].rows(child_hi, target),
                            b[scale].rows(child_hi, target),
                        ],
                        n_out,
                        plane_w,
                        [
                            x[scale].append_rows(n_out),
                            y[scale].append_rows(n_out),
                            bb[scale].append_rows(n_out),
                        ],
                        self.revision,
                        &mut self.lin_buf,
                    );
                }
            }
            crate::fold_timing::stop(__t_ds, crate::fold_timing::Phase::ProdDownscale, 0);
            for (scale, held) in self.max_held_rows.iter_mut().enumerate() {
                let p = &self.planes[0][0][scale];
                *held = (*held).max(p.hi - p.lo);
            }
            return true;
        }
        let ref_planes = self.ref_planes;
        #[cfg(feature = "threads")]
        let cascade_parallel = self.parallel;
        let (h, t) = self.planes.split_at_mut(1);
        let mut chans: [(usize, usize, &mut Vec<RollingPlane>); 6] = {
            let [s0, s1, s2] = &mut h[0];
            let [d0, d1, d2] = &mut t[0];
            [
                (0, 0, s0),
                (0, 1, s1),
                (0, 2, s2),
                (1, 0, d0),
                (1, 1, d1),
                (1, 2, d2),
            ]
        };
        for scale in 1..crate::NUM_SCALES {
            let (plane_w, plane_h) = (self.scales[scale].plane_w, self.scales[scale].plane_h);
            let parent_w = self.scales[scale - 1].plane_w;
            let one = |(si, ch, planes): &mut (usize, usize, &mut Vec<RollingPlane>)| {
                let parent_hi = planes[scale - 1].hi;
                let child_hi = planes[scale].hi;
                let target = (parent_hi / 2).min(plane_h);
                if target <= child_hi {
                    return;
                }
                let n_out = target - child_hi;
                // REF-CACHED FEED: the cache holds every level, built by
                // the same `downscale_2x_into` this cascade runs, so the
                // source side advances its borrowed view instead of downscaling.
                if *si == 0
                    && let Some(rp) = ref_planes
                {
                    let (cached, cw, _) = &rp[scale];
                    debug_assert_eq!(*cw, plane_w, "cached ref stride at scale {scale}");
                    debug_assert!(cached[*ch].len() >= (child_hi + n_out) * plane_w);
                    planes[scale].hi = child_hi + n_out;
                    return;
                }
                // Split borrow: parent (read) vs child (write).
                let (head, tail) = planes.split_at_mut(scale);
                let parent = &head[scale - 1];
                let child = &mut tail[0];
                let src_rows = parent.rows(child_hi * 2, child_hi * 2 + n_out * 2);
                let dst = child.append_rows(n_out);
                crate::blur::downscale_2x_into(src_rows, parent_w, dst, plane_w, n_out);
            };
            #[cfg(feature = "threads")]
            if cascade_parallel {
                use rayon::prelude::*;
                chans.par_iter_mut().for_each(one);
                continue;
            }
            chans.iter_mut().for_each(one);
        }

        crate::fold_timing::stop(__t_ds, crate::fold_timing::Phase::ProdDownscale, 0);

        for (scale, held) in self.max_held_rows.iter_mut().enumerate() {
            let p = &self.planes[0][0][scale];
            *held = (*held).max(p.hi - p.lo);
        }
        true
    }

    /// Retire rows no longer needed by (a) the next unemitted kernel
    /// strip's wide window at each scale, or (b) the parent rows the next
    /// downscale production step still has to read.
    fn retire(&mut self) {
        for scale in 0..crate::NUM_SCALES {
            let st = &self.scales[scale];
            // (a) Wide-window floor for the next strip to emit. The top
            //     reflection only ever reads rows ≥ 0 that a top strip
            //     still holds (lo starts at 0 and this floor is 0 until
            //     the first strip retires).
            let ks_floor = if st.next_ks * STRIP_ROWS >= st.plane_h {
                st.plane_h // scale fully emitted — no kernel need
            } else {
                (st.next_ks * STRIP_ROWS).saturating_sub(HALO_P)
            };
            // (b) Downscale floor: the child's next output row `j` reads
            //     parent rows [2j, 2j+2).
            let ds_floor = if scale + 1 < crate::NUM_SCALES {
                let child_hi = self.planes[0][0][scale + 1].hi;
                let child_h = self.scales[scale + 1].plane_h;
                if child_hi >= child_h {
                    st.plane_h
                } else {
                    child_hi * 2
                }
            } else {
                st.plane_h
            };
            let floor = ks_floor.min(ds_floor);
            for si in 0..2 {
                for ch in 0..3 {
                    self.planes[si][ch][scale].retire_below(floor);
                }
            }
            // Linear rows are read only by the next level's downscale; the
            // last level's are converted as they are made.
            if let Some(lin) = self.lin.as_deref_mut() {
                for side in lin.iter_mut() {
                    let floor = if scale + 1 < crate::NUM_SCALES {
                        let child_hi = side[0][scale + 1].hi;
                        if child_hi >= self.scales[scale + 1].plane_h {
                            st.plane_h
                        } else {
                            child_hi * 2
                        }
                    } else {
                        side[0][scale].hi
                    };
                    for ch in side.iter_mut() {
                        ch[scale].retire_below(floor);
                    }
                }
            }
        }
    }

    /// Zero-copy wide window for an INTERIOR strip (no reflection):
    /// contiguous rows `[y0 − HALO_P, y0 + strip_h + HALO_P)`. `None`
    /// when the strip touches the true top/bottom — use
    /// [`Self::fill_wide`] there.
    pub(crate) fn wide_window(&self, side: Side, ch: usize, info: &StripInfo) -> Option<&[f32]> {
        if !info.interior() {
            return None;
        }
        Some(self.rows(
            side,
            ch,
            info.scale,
            info.y0 - HALO_P,
            info.y0 + info.strip_h + HALO_P,
        ))
    }

    /// Materialize the strip's wide window into `dst`
    /// (`plane_w × wide_h()` elements) with [`gather_strip_halo`]'s exact
    /// semantics: buffer row `i` = plane row
    /// `reflect_101(y0 − HALO_P + i, plane_h)`.
    pub(crate) fn fill_wide(&self, side: Side, ch: usize, info: &StripInfo, dst: &mut [f32]) {
        let w = info.plane_w;
        let wide_h = info.wide_h();
        debug_assert!(dst.len() >= w * wide_h);
        for i in 0..wide_h {
            let gy = info.y0 as isize - HALO_P as isize + i as isize;
            let gy_r = reflect_101(gy, info.plane_h);
            dst[i * w..(i + 1) * w].copy_from_slice(self.rows(
                side,
                ch,
                info.scale,
                gy_r,
                gy_r + 1,
            ));
        }
    }

    /// Contiguous plane rows `[r0, r1)` at a scale (for the strip-fed
    /// blockiness/gradient consumers — rows must lie inside the current
    /// window, which the emission/retire order guarantees for
    /// `[y0 − 1, y0 + strip_h + 1)` of the just-emitted strip).
    pub(crate) fn rows(&self, side: Side, ch: usize, scale: usize, r0: usize, r1: usize) -> &[f32] {
        if let Some(sampled) = &self.sampled {
            let si = match side {
                Side::Source => 0,
                Side::Distorted => 1,
            };
            let p = &sampled[si][scale];
            return &p.0[ch][r0 * p.1..r1 * p.1];
        }
        if side == Side::Source
            && let Some(cached) = self.ref_planes
        {
            let held = self.plane(side, ch, scale);
            debug_assert!(held.lo <= r0 && r0 <= r1 && r1 <= held.hi);
            let (planes, width, _) = &cached[scale];
            return &planes[ch][r0 * width..r1 * width];
        }
        self.plane(side, ch, scale).rows(r0, r1)
    }

    /// Peak rows held per scale (test telemetry).
    #[cfg(all(test, feature = "feature-regime-v2"))]
    pub(crate) fn max_held_rows(&self) -> &[usize] {
        &self.max_held_rows
    }

    /// Return every rolling-plane buffer to `pool` for the next pair.
    pub(crate) fn recycle(self, pool: &mut Vec<Vec<f32>>) {
        for side in self.lin.into_iter().flat_map(|lin| *lin) {
            for ch in side {
                pool.extend(ch.into_iter().map(|plane| plane.buf));
            }
        }
        for side in self.planes.into_iter().rev() {
            for ch in side.into_iter().rev() {
                for plane in ch.into_iter().rev() {
                    pool.push(plane.buf);
                }
            }
        }
    }
}

/// Read one source row as absolute-linear cd/m² RGB triples per the
/// declared [`HdrEncoding`]. Accepted pixel formats (validated at the HDR
/// entry): `LinearF32Rgba` (f32 values used directly — cd/m² for
/// `Linear`, code values for `Pq`/`Hlg`; alpha ignored, Opaque required)
/// and `Srgb16Rgba` (u16 code values normalized by 65535 — `Pq`/`Hlg`
/// code-value containers like cICP-spliced 16-bit PNG).
///
/// Every reachable caller (`streaming::PrecomputedReference::for_candidate_inner`,
/// `attribution::Fused944Session::planned_features`) sits behind
/// `custom-profiles` in addition to this module's own `feature-regime-v2`
/// gate; `allow` rather than `cfg` because `planned_features` itself is
/// gated on `feature-regime-v2` alone and still calls this unconditionally,
/// so cfg-stripping the function under `feature-regime-v2`-only would break
/// that compile rather than just leave it unreached.
#[cfg(feature = "feature-regime-v2")]
#[cfg_attr(not(feature = "custom-profiles"), allow(dead_code))]
pub(crate) fn hdr_source_to_xyb(
    source: &impl ImageSource,
    encoding: HdrEncoding,
    out: &mut [Vec<f32>; 3],
    revision: crate::feature_defs::FormulaRevision,
) {
    let width = source.width();
    let mut row = vec![[0.0; 3]; width];
    let [x, y, b] = out;
    for r in 0..source.height() {
        hdr_source_row_to_nits(source, r, encoding, &mut row, revision);
        let range = r * width..(r + 1) * width;
        crate::color::linear_to_pu_xyb_planar_into_at_revision(
            &row,
            &mut x[range.clone()],
            &mut y[range.clone()],
            &mut b[range],
            revision,
        );
    }
}

fn hdr_source_row_to_nits(
    src: &impl ImageSource,
    y: usize,
    encoding: HdrEncoding,
    out: &mut [[f32; 3]],
    revision: crate::feature_defs::FormulaRevision,
) {
    let width = src.width();
    let row_bytes = &src.row_bytes(y)[..width * src.pixel_format().bytes_per_pixel()];
    let mut already_nits = false;
    match src.pixel_format() {
        PixelFormat::LinearF32Rgba => {
            let px = crate::source::packed_row::<[f32; 4]>(row_bytes, width);
            for (o, p) in out[..width].iter_mut().zip(&px[..width]) {
                *o = [p[0], p[1], p[2]];
            }
        }
        PixelFormat::Srgb16Rgba => {
            if let HdrEncoding::Pq { peak_nits } = encoding {
                crate::transfer::decode_pq_u16_rgba_row_at_revision(
                    row_bytes,
                    &mut out[..width],
                    peak_nits,
                    revision,
                );
                already_nits = true;
            } else {
                const INV: f32 = 1.0 / 65535.0;
                for (x, o) in out[..width].iter_mut().enumerate() {
                    let off = x * 8;
                    let r = u16::from_ne_bytes([row_bytes[off], row_bytes[off + 1]]);
                    let g = u16::from_ne_bytes([row_bytes[off + 2], row_bytes[off + 3]]);
                    let b = u16::from_ne_bytes([row_bytes[off + 4], row_bytes[off + 5]]);
                    *o = [r as f32 * INV, g as f32 * INV, b as f32 * INV];
                }
            }
        }
        other => unreachable!(
            "HDR route accepts LinearF32Rgba/Srgb16Rgba only (validated at the entry); got {other:?}"
        ),
    }
    if !already_nits {
        match encoding {
            HdrEncoding::Linear => {}
            HdrEncoding::Pq { peak_nits } => {
                crate::transfer::decode_pq_row_at_revision(&mut out[..width], peak_nits, revision)
            }
            HdrEncoding::Hlg {
                peak_nits,
                ambient_lux,
            } => crate::transfer::decode_hlg_row_in_primaries_at_revision(
                &mut out[..width],
                peak_nits,
                ambient_lux,
                src.color_primaries(),
                revision,
            ),
        }
    }
    // Absolute light must remain absolute: never apply the SDR [0, 1]
    // display clamp here. The opsin matrix is defined in linear sRGB.
    for pixel in &mut out[..width] {
        crate::color::apply_gamut_matrix(
            pixel,
            src.color_primaries(),
            crate::source::GamutMapping::Preserve,
        );
    }
}

// ============================================================================
// C1 unit gate: producer windows byte-equal the materialized-path planes.
// ============================================================================

#[cfg(all(test, feature = "feature-regime-v2"))]
mod tests {
    use super::*;
    use crate::source::RgbSlice;

    #[test]
    fn rev5_downscaled_chroma_matches_full_conversion() {
        // Tier switches affect the whole process. Isolate them from the
        // ordinary parallel suite, whose two-pass parity tests need a fixed tier.
        if std::env::var("REV5_CHROMA_CHILD").as_deref() != Ok("1") {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "feature_v2_stream::tests::rev5_downscaled_chroma_matches_full_conversion",
                    "--exact",
                    "--nocapture",
                ])
                .env("REV5_CHROMA_CHILD", "1")
                .env("ZENSIM_FORMULA_REV", "5")
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(String::from_utf8_lossy(&out.stdout).contains("REV5-CHROMA-PASS"));
            return;
        }
        use crate::feature_defs::FormulaRevision;
        let _guard = archmage::testing::lock_token_testing();
        let _ = archmage::testing::for_each_token_permutation(
            archmage::testing::CompileTimePolicy::Warn,
            |_| {
                for (w, h) in [
                    (64, 64),
                    (97, 63),
                    (64, 129),
                    (97, 193),
                    (257, 257),
                    (257, 289),
                ] {
                    let pixels = textured_image(w, h, 71);
                    let image = RgbSlice::new(&pixels, w, h);
                    let cached_planes = materialize(&image);
                    for (cached, parallel) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        let feed = cached.then_some(cached_planes.as_slice());
                        let (mut pool_a, mut pool_b) = (Vec::new(), Vec::new());
                        let mut a = StripPlaneProducer::new_with_ref_feed(
                            &image,
                            &image,
                            parallel,
                            &mut pool_a,
                            FrontEnd::Sdr,
                            feed,
                            None,
                            FormulaRevision::Rev5,
                            false,
                        );
                        let mut b = StripPlaneProducer::new_with_ref_feed(
                            &image,
                            &image,
                            parallel,
                            &mut pool_b,
                            FrontEnd::Sdr,
                            feed,
                            None,
                            FormulaRevision::Rev5,
                            true,
                        );
                        while let Some(ia) = a.next_strip() {
                            let ib = b.next_strip().unwrap();
                            assert_eq!(
                                (ia.scale, ia.y0, ia.strip_h),
                                (ib.scale, ib.y0, ib.strip_h)
                            );
                            for ch in 0..3 {
                                if ia.scale == 0 && ch != 1 {
                                    continue;
                                }
                                for side in [Side::Source, Side::Distorted] {
                                    assert_eq!(
                                        a.rows(side, ch, ia.scale, ia.y0, ia.y0 + ia.strip_h),
                                        b.rows(side, ch, ib.scale, ib.y0, ib.y0 + ib.strip_h),
                                        "{w}x{h} scale={} ch={ch}",
                                        ia.scale
                                    );
                                }
                            }
                        }
                        assert!(b.next_strip().is_none());
                        assert_eq!(b.planes[0][0][0].buf.len(), 2 * w);
                    }
                }
            },
        );
        println!("REV5-CHROMA-PASS");
    }

    /// Deterministic textured content (same family as feature_v2's tests).
    fn textured_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut px = vec![[0u8; 3]; w * h];
        let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for y in 0..h {
            for x in 0..w {
                let n = next();
                let gx = ((x * 5) & 0x7F) as u8;
                let gy = ((y * 3) & 0x7F) as u8;
                px[y * w + x] = [
                    gx.wrapping_add((n & 0x3F) as u8),
                    gy.wrapping_add(((n >> 8) & 0x3F) as u8),
                    ((x ^ y) & 0xFF) as u8 ^ ((n >> 16) & 0x1F) as u8,
                ];
            }
        }
        px
    }

    fn distort(src: &[[u8; 3]]) -> Vec<[u8; 3]> {
        src.iter()
            .map(|&[r, g, b]| [(r & !7) | 4, g.saturating_add(3), b & !3])
            .collect()
    }

    /// Materialized pyramid for one side, built EXACTLY like the
    /// prepared-reference path (`build_v2_ref_scales`: whole-plane XYB +
    /// `downscale_2x_into` per level). The materialized walk's distorted
    /// side uses `downscale_2x_inplace` instead — documented (and
    /// asserted here) bit-identical arithmetic.
    fn materialize(img: &impl ImageSource) -> Vec<([Vec<f32>; 3], usize, usize)> {
        let mut width = img.width();
        let mut height = img.height();
        let mut scales: Vec<([Vec<f32>; 3], usize, usize)> = Vec::new();
        scales.push((
            crate::streaming::convert_source_to_xyb(
                img,
                width,
                false,
                crate::ssim_form::active_revision(),
            ),
            width,
            height,
        ));
        for _ in 1..crate::NUM_SCALES {
            let new_w = width / 2;
            let new_h = height / 2;
            let mut next: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; new_w * new_h]);
            {
                let (prev, _, _) = scales.last().unwrap();
                for ch in 0..3 {
                    crate::blur::downscale_2x_into(&prev[ch], width, &mut next[ch], new_w, new_h);
                }
            }
            scales.push((next, new_w, new_h));
            width = new_w;
            height = new_h;
        }
        scales
    }

    /// In-place downscale (the materialized walk's distorted-side
    /// operator) produces the same bytes as `downscale_2x_into`.
    #[test]
    fn downscale_into_matches_inplace() {
        let (w, h) = (127, 93);
        let src = textured_image(w, h, 3);
        let img = RgbSlice::new(&src, w, h);
        let mut planes = crate::streaming::convert_source_to_xyb(
            &img,
            w,
            false,
            crate::ssim_form::active_revision(),
        );
        let reference = materialize(&img);
        let (mut cw, mut ch_) = (w, h);
        #[allow(clippy::needless_range_loop)] // scale drives geometry AND indexes reference/planes
        for scale in 1..crate::NUM_SCALES {
            for p in planes.iter_mut() {
                crate::blur::downscale_2x_inplace(p, cw, ch_);
            }
            cw /= 2;
            ch_ /= 2;
            let (ref_planes, rw, rh) = &reference[scale];
            assert_eq!((cw, ch_), (*rw, *rh));
            for c in 0..3 {
                assert_eq!(planes[c].len(), ref_planes[c].len());
                for (i, (a, b)) in planes[c].iter().zip(ref_planes[c].iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "scale {scale} ch {c} idx {i}");
                }
            }
        }
    }

    /// The advance is a SCHEDULE knob, not semantics: sweeping it changes how
    /// many rows one `produce()` call converts (and therefore the cross-scale
    /// interleave of emitted strips), and every strip's wide window on every
    /// side and channel must come back BIT-identical.
    ///
    /// This is the direct gate on [`advance_rows_for`] making the advance
    /// depend on the rayon pool size. The values swept are every one that
    /// function can return (64/128/192/256) plus 512, which is past its
    /// ceiling — all multiples of [`CONVERT_CHUNK_ROWS`], which is the
    /// invariant that keeps the conversion byte-identical to the whole-image
    /// call (chunk HEIGHT is semantics; see
    /// `crate::streaming::tests::convert_chunk_rows_is_semantics_not_a_knob`).
    #[test]
    fn producer_windows_are_advance_invariant() {
        // Heights that straddle every swept advance, plus a sub-advance one.
        let cases = [(96usize, 64usize), (127, 93), (200, 150), (96, 1030)];
        for (w, h) in cases {
            let src = textured_image(w, h, 23);
            let dst = distort(&src);
            let s_img = RgbSlice::new(&src, w, h);
            let d_img = RgbSlice::new(&dst, w, h);
            // (scale, y0, side, ch) -> window bits, from the reference advance.
            let mut baseline: Vec<(usize, usize, usize, usize, Vec<u32>)> = Vec::new();
            for (ai, advance) in [64usize, 128, 192, 256, 512].into_iter().enumerate() {
                let mut pool = Vec::new();
                let mut prod =
                    StripPlaneProducer::new_with_advance(&s_img, &d_img, false, &mut pool, advance);
                let mut got: Vec<(usize, usize, usize, usize, Vec<u32>)> = Vec::new();
                let mut wide = Vec::new();
                while let Some(info) = prod.next_strip() {
                    let n_wide = info.plane_w * info.wide_h();
                    for (si, side) in [Side::Source, Side::Distorted].into_iter().enumerate() {
                        for ch in 0..3 {
                            wide.clear();
                            wide.resize(n_wide, 0.0f32);
                            prod.fill_wide(side, ch, &info, &mut wide[..n_wide]);
                            got.push((
                                info.scale,
                                info.y0,
                                si,
                                ch,
                                wide.iter().map(|v| v.to_bits()).collect(),
                            ));
                        }
                    }
                }
                // Emission ORDER may differ across advances (the cross-scale
                // interleave is exactly what the advance changes); the SET of
                // (scale, y0, side, ch) windows and their bits may not.
                got.sort_by_key(|t| (t.0, t.1, t.2, t.3));
                if ai == 0 {
                    baseline = got;
                    assert!(!baseline.is_empty(), "{w}x{h}: no strips emitted");
                } else {
                    assert_eq!(
                        got.len(),
                        baseline.len(),
                        "{w}x{h} advance {advance}: strip/window count moved"
                    );
                    for (g, b) in got.iter().zip(baseline.iter()) {
                        assert_eq!(
                            (g.0, g.1, g.2, g.3),
                            (b.0, b.1, b.2, b.3),
                            "{w}x{h} advance {advance}: window identity moved"
                        );
                        assert_eq!(
                            g.4, b.4,
                            "{w}x{h} advance {advance}: scale {} y0 {} side {} ch {} moved bits",
                            g.0, g.1, g.2, g.3
                        );
                    }
                }
            }
        }
    }

    /// THE C1 gate: every emitted strip's wide window — both sides, all
    /// 3 channels — is byte-equal to `gather_strip_halo` run against the
    /// materialized planes; interior strips' zero-copy windows equal the
    /// filled ones; every kernel strip of every scale is emitted exactly
    /// once, ascending per scale; capacity budget holds.
    /// `ZENSIM_PYRAMID=linear` (a child process, the toggle being read once
    /// per process): every strip window of both producers, both sides, every
    /// channel and scale, bit for bit against the whole-image form (scale 0
    /// as the default conversion; each coarse level the 2×2 average of the
    /// parent level's linear RGB, converted to positive XYB in one call), at
    /// revisions 5 and 7, with the walk's scale-0 chroma omission requested
    /// and refused. Negative control: the coarse levels differ from the
    /// XYB-averaged pyramid.
    #[cfg(feature = "threads")]
    #[test]
    fn linear_pyramid_producers_match_the_whole_image_form() {
        if std::env::var("ZENSIM_PYRAMID").as_deref() != Ok("linear") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "feature_v2_stream::tests::linear_pyramid_producers_match_the_whole_image_form",
                    "--exact",
                ])
                .env("ZENSIM_PYRAMID", "linear")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        use crate::feature_defs::FormulaRevision;
        use crate::feature_v2_blocks::{BlockPool, CHUNK_ROWS, gather_window, produce_chunk};
        use std::sync::Arc;
        assert!(linear_pyramid());
        let whole = |img: &RgbSlice<'_>, revision| {
            let (mut w, mut h) = (img.width(), img.height());
            let mut out = vec![(
                crate::streaming::convert_source_to_xyb(img, w, false, revision),
                w,
                h,
            )];
            let mut lin: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; w * h]);
            for y in 0..h {
                let row: &[[u8; 3]] = bytemuck::cast_slice(&img.row_bytes(y)[..w * 3]);
                for (x, p) in row.iter().enumerate() {
                    for ch in 0..3 {
                        lin[ch][y * w + x] = crate::color::srgb_u8_to_linear(p[ch]);
                    }
                }
            }
            for _ in 1..crate::NUM_SCALES {
                let (nw, nh) = (w / 2, h / 2);
                let next: [Vec<f32>; 3] = std::array::from_fn(|ch| {
                    let mut v = vec![0.0; nw * nh];
                    crate::blur::downscale_2x_into(&lin[ch], w, &mut v, nw, nh);
                    v
                });
                let pixels: Vec<[f32; 3]> = (0..nw * nh)
                    .map(|i| [next[0][i], next[1][i], next[2][i]])
                    .collect();
                let mut xyb: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; nw * nh]);
                {
                    let [x, y, b] = &mut xyb;
                    crate::color::linear_to_positive_xyb_planar_into(&pixels, x, y, b, revision);
                }
                out.push((xyb, nw, nh));
                (lin, w, h) = (next, nw, nh);
            }
            out
        };
        let cases = [
            (96usize, 64usize),
            (127, 93),
            (208, 144),
            (96, 1030),
            (320, 517),
            (77, 300),
        ];
        for revision in [FormulaRevision::Rev5, FormulaRevision::Rev7] {
            for (w, h) in cases {
                let src = textured_image(w, h, 11);
                let dst = distort(&src);
                let s_img = RgbSlice::new(&src, w, h);
                let d_img = RgbSlice::new(&dst, w, h);
                let expected = [whole(&s_img, revision), whole(&d_img, revision)];
                // Negative control: averaging XYB instead gives other levels.
                let mut avg = expected[1][0].0.clone();
                for (scale, (planes, cw, chh)) in expected[1].iter().enumerate().skip(1) {
                    let pw = w >> (scale - 1);
                    avg = std::array::from_fn(|ch| {
                        let mut v = vec![0.0; cw * chh];
                        crate::blur::downscale_2x_into(&avg[ch], pw, &mut v, *cw, *chh);
                        v
                    });
                    assert_ne!(*planes, avg, "{w}x{h} scale {scale}");
                }
                let dims: Vec<(usize, usize)> =
                    (0..crate::NUM_SCALES).map(|s| (w >> s, h >> s)).collect();
                let pool = Arc::new(BlockPool::default());
                let chunks: Vec<[crate::feature_v2_blocks::ChunkPlanes; 2]> = (0..h
                    .div_ceil(CHUNK_ROWS))
                    .map(|k| {
                        [
                            produce_chunk(&s_img, k, &dims, false, revision, &pool),
                            produce_chunk(&d_img, k, &dims, false, revision, &pool),
                        ]
                    })
                    .collect();
                for omit in [false, true] {
                    let mut spool = Vec::new();
                    let mut prod = StripPlaneProducer::new_with_ref_feed(
                        &s_img,
                        &d_img,
                        false,
                        &mut spool,
                        FrontEnd::Sdr,
                        None,
                        None,
                        revision,
                        omit,
                    );
                    let (mut wide, mut wide_ref, mut wide_blk) =
                        (Vec::new(), Vec::new(), Vec::new());
                    let mut strips = 0;
                    while let Some(info) = prod.next_strip() {
                        strips += 1;
                        let n_wide = info.plane_w * info.wide_h();
                        wide.resize(n_wide, 0.0);
                        wide_ref.resize(n_wide, 0.0);
                        wide_blk.resize(n_wide, 0.0);
                        for (side, si) in [(Side::Source, 0), (Side::Distorted, 1)] {
                            let (planes, pw, ph) = &expected[si][info.scale];
                            assert_eq!((*pw, *ph), (info.plane_w, info.plane_h));
                            for ch in 0..3 {
                                gather_strip_halo(
                                    &planes[ch],
                                    *pw,
                                    *ph,
                                    info.y0,
                                    info.wide_h(),
                                    HALO_P,
                                    &mut wide_ref,
                                );
                                prod.fill_wide(side, ch, &info, &mut wide);
                                let blocks: Vec<_> = chunks
                                    .iter()
                                    .filter_map(|c| c[si][ch][info.scale].clone())
                                    .collect();
                                gather_window(&blocks, &info, &mut wide_blk);
                                for i in 0..n_wide {
                                    let at = || {
                                        format!(
                                            "{revision:?} omit={omit} {w}x{h} scale {} y0={} \
                                             side {side:?} ch {ch} idx {i}",
                                            info.scale, info.y0
                                        )
                                    };
                                    assert_eq!(
                                        wide[i].to_bits(),
                                        wide_ref[i].to_bits(),
                                        "{}",
                                        at()
                                    );
                                    assert_eq!(
                                        wide_blk[i].to_bits(),
                                        wide_ref[i].to_bits(),
                                        "{}",
                                        at()
                                    );
                                }
                            }
                        }
                    }
                    assert!(strips > 0);
                    prod.recycle(&mut spool);
                }
            }
        }
    }

    /// The block producer (`crate::feature_v2_blocks`, the parallel walk's
    /// rows) against this producer: every strip window of every scale, both
    /// sides and every stored channel, bit for bit, at revisions 5 and 7 with
    /// and without scale-0 chroma, over odd, tall, short and non-multiple-of-64
    /// heights.
    #[cfg(feature = "threads")]
    #[test]
    fn block_producer_windows_match_the_streaming_producer() {
        use crate::feature_defs::FormulaRevision;
        use crate::feature_v2_blocks::{BlockPool, CHUNK_ROWS, gather_window, produce_chunk};
        use std::sync::Arc;
        let cases = [
            (96usize, 64usize),
            (127, 93),
            (208, 144),
            (200, 150),
            (96, 1030),
            (320, 517),
            (77, 300),
        ];
        for revision in [FormulaRevision::Rev5, FormulaRevision::Rev7] {
            for omit in [false, true] {
                for (w, h) in cases {
                    let src = textured_image(w, h, 11);
                    let dst = distort(&src);
                    let s_img = RgbSlice::new(&src, w, h);
                    let d_img = RgbSlice::new(&dst, w, h);
                    let dims: Vec<(usize, usize)> =
                        (0..crate::NUM_SCALES).map(|s| (w >> s, h >> s)).collect();
                    let pool = Arc::new(BlockPool::default());
                    let chunks: Vec<[crate::feature_v2_blocks::ChunkPlanes; 2]> = (0..h
                        .div_ceil(CHUNK_ROWS))
                        .map(|k| {
                            [
                                produce_chunk(&s_img, k, &dims, omit, revision, &pool),
                                produce_chunk(&d_img, k, &dims, omit, revision, &pool),
                            ]
                        })
                        .collect();
                    let mut spool = Vec::new();
                    let mut prod = StripPlaneProducer::new_with_ref_feed(
                        &s_img,
                        &d_img,
                        false,
                        &mut spool,
                        FrontEnd::Sdr,
                        None,
                        None,
                        revision,
                        omit,
                    );
                    let (mut wide, mut wide_blk) = (Vec::new(), Vec::new());
                    let mut strips = 0;
                    while let Some(info) = prod.next_strip() {
                        strips += 1;
                        let n_wide = info.plane_w * info.wide_h();
                        wide.resize(n_wide, 0.0);
                        wide_blk.resize(n_wide, 0.0);
                        for (side, si) in [(Side::Source, 0), (Side::Distorted, 1)] {
                            for ch in 0..3 {
                                if omit && info.scale == 0 && ch != 1 {
                                    continue;
                                }
                                let blocks: Vec<_> = chunks
                                    .iter()
                                    .filter_map(|c| c[si][ch][info.scale].clone())
                                    .collect();
                                gather_window(&blocks, &info, &mut wide_blk);
                                prod.fill_wide(side, ch, &info, &mut wide);
                                for i in 0..n_wide {
                                    assert_eq!(
                                        wide_blk[i].to_bits(),
                                        wide[i].to_bits(),
                                        "{revision:?} omit={omit} {w}x{h} scale {} y0={} \
                                         side {side:?} ch {ch} idx {i}",
                                        info.scale,
                                        info.y0
                                    );
                                }
                            }
                        }
                    }
                    assert!(strips > 0);
                }
            }
        }
    }

    #[test]
    fn producer_windows_byte_equal_materialized() {
        // Odd dims, sub-one-strip, multi-strip, tall-thin, and >2-strip
        // heights; all ≥ MIN_PYRAMID_DIM (sub-64 pads before the
        // producer, covered by entry-level tests).
        let cases = [
            (96usize, 64usize),
            (127, 93),
            (208, 144),
            (200, 150),
            (96, 1030),
            (320, 517),
        ];
        for (w, h) in cases {
            let src = textured_image(w, h, 11);
            let dst = distort(&src);
            let s_img = RgbSlice::new(&src, w, h);
            let d_img = RgbSlice::new(&dst, w, h);
            let mat_s = materialize(&s_img);
            let mat_d = materialize(&d_img);

            let mut pool = Vec::new();
            let mut prod = StripPlaneProducer::new(&s_img, &d_img, false, &mut pool);
            let mut emitted: Vec<Vec<usize>> = vec![Vec::new(); crate::NUM_SCALES];
            let mut wide = Vec::new();
            let mut wide_ref = Vec::new();
            while let Some(info) = prod.next_strip() {
                emitted[info.scale].push(info.y0);
                let n_wide = info.plane_w * info.wide_h();
                wide.resize(n_wide, 0.0);
                wide_ref.resize(n_wide, 0.0);
                for (side, mat) in [(Side::Source, &mat_s), (Side::Distorted, &mat_d)] {
                    let (planes, pw, ph) = &mat[info.scale];
                    assert_eq!((*pw, *ph), (info.plane_w, info.plane_h));
                    #[allow(clippy::needless_range_loop)] // ch feeds fill_wide AND indexes planes
                    for ch in 0..3 {
                        prod.fill_wide(side, ch, &info, &mut wide);
                        gather_strip_halo(
                            &planes[ch],
                            *pw,
                            *ph,
                            info.y0,
                            info.wide_h(),
                            HALO_P,
                            &mut wide_ref,
                        );
                        for i in 0..n_wide {
                            assert_eq!(
                                wide[i].to_bits(),
                                wide_ref[i].to_bits(),
                                "{w}x{h} scale {} strip y0={} side {side:?} ch {ch} idx {i}",
                                info.scale,
                                info.y0
                            );
                        }
                        if let Some(win) = prod.wide_window(side, ch, &info) {
                            assert!(info.interior());
                            for i in 0..n_wide {
                                assert_eq!(
                                    win[i].to_bits(),
                                    wide[i].to_bits(),
                                    "in-place window mismatch at idx {i}"
                                );
                            }
                        } else {
                            assert!(!info.interior());
                        }
                    }
                }
            }

            // Full, ordered coverage of every scale's strip tiling.
            let (mut pw, mut ph) = (w, h);
            #[allow(clippy::needless_range_loop)] // scale drives tiling AND indexes emitted
            for scale in 0..crate::NUM_SCALES {
                let expect: Vec<usize> = (0..ph.div_ceil(STRIP_ROWS))
                    .map(|k| k * STRIP_ROWS)
                    .collect();
                assert_eq!(
                    emitted[scale], expect,
                    "{w}x{h} scale {scale} emission order/coverage"
                );
                pw /= 2;
                ph /= 2;
            }
            let _ = pw;

            // Capacity budget: rolling windows never outgrew their
            // construction-time budget (no fallback growth).
            for (scale, &held) in prod.max_held_rows().iter().enumerate() {
                let budget = scale_capacity_rows(
                    scale,
                    if scale == 0 { h } else { h >> scale },
                    advance_rows_for(false, h),
                );
                assert!(
                    held <= budget,
                    "{w}x{h} scale {scale}: held {held} rows > budget {budget}"
                );
            }
        }
    }

    /// Translucent RGBA: the producer's strip conversion must keep the
    /// alpha noise-background in the PARENT image's row phase
    /// (`abs_row_offset`), matching whole-image conversion bit-for-bit.
    #[test]
    fn producer_alpha_phase_matches_whole_image() {
        let (w, h) = (96, 517);
        let base = textured_image(w, h, 29);
        let rgba: Vec<[u8; 4]> = base
            .iter()
            .enumerate()
            .map(|(i, &[r, g, b])| [r, g, b, (((i * 37) % 256) as u8)])
            .collect();
        let img = crate::source::RgbaSlice::new(&rgba, w, h);
        let mat = materialize(&img);

        let mut pool = Vec::new();
        let mut prod = StripPlaneProducer::new(&img, &img, false, &mut pool);
        let mut wide = Vec::new();
        let mut wide_ref = Vec::new();
        while let Some(info) = prod.next_strip() {
            let n_wide = info.plane_w * info.wide_h();
            wide.resize(n_wide, 0.0);
            wide_ref.resize(n_wide, 0.0);
            let (planes, pw, ph) = &mat[info.scale];
            #[allow(clippy::needless_range_loop)] // ch feeds fill_wide AND indexes planes
            for ch in 0..3 {
                prod.fill_wide(Side::Source, ch, &info, &mut wide);
                gather_strip_halo(
                    &planes[ch],
                    *pw,
                    *ph,
                    info.y0,
                    info.wide_h(),
                    HALO_P,
                    &mut wide_ref,
                );
                for i in 0..n_wide {
                    assert_eq!(
                        wide[i].to_bits(),
                        wide_ref[i].to_bits(),
                        "alpha-phase mismatch: scale {} y0 {} ch {ch} idx {i}",
                        info.scale,
                        info.y0
                    );
                }
            }
        }
    }
}
