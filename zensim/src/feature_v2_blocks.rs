//! Pyramid rows produced by parallel chunk tasks, for the strip-job pipeline of
//! plain walks at formula revision 5 and later (SDR, no cached reference, no
//! sampling, nothing retained).
//!
//! The streaming producer ([`crate::feature_v2_stream::StripPlaneProducer`])
//! converts rows and runs the downscale cascade on the walk's thread, and the
//! walk's thread copied every strip window into its job; at eight threads that
//! thread was the walk's critical path. Here one task produces one 64-row chunk
//! of one image: its scale-0 rows (the same colour conversion calls, on the same
//! 64-row lattice, with the same absolute row offset) and every row of scales
//! 1–3 those rows determine (`downscale_2x_into` on row pairs; a chunk's span at
//! every scale is even, so no pair crosses a chunk). Each plane's rows land in
//! an immutable [`RowBlock`], and strip jobs copy their windows out of the
//! blocks under them. Every value is the streaming producer's.

use std::sync::{Arc, Mutex};

use crate::feature_defs::FormulaRevision;
use crate::feature_v2::{HALO_P, reflect_101};
use crate::feature_v2_stream::StripInfo;
use crate::source::{ImageSource, SubsetView};

/// Scale-0 rows per production task: the conversion lattice.
pub(crate) const CHUNK_ROWS: usize = crate::streaming::DEFAULT_CONVERT_CHUNK_ROWS;

/// Recycled block buffers, one free list per scale (a scale's blocks share
/// one size). It lives in the walk's scratch, so steady-state walks reuse
/// the previous walk's buffers instead of faulting in fresh pages.
#[derive(Default)]
pub(crate) struct BlockPool {
    free: [Mutex<Vec<Vec<f32>>>; crate::NUM_SCALES],
}

impl BlockPool {
    fn take(&self, scale: usize, len: usize) -> Vec<f32> {
        let mut free = self.free[scale].lock().unwrap_or_else(|e| e.into_inner());
        // A smaller buffer is left over from a narrower image: drop it.
        while let Some(v) = free.pop() {
            if v.len() >= len {
                return v;
            }
        }
        drop(free);
        vec![0.0; len]
    }

    fn put(&self, scale: usize, v: Vec<f32>) {
        self.free[scale]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(v);
    }
}

/// Rows `[r0, r0 + rows)` of one pyramid plane, immutable once produced. The
/// buffer goes back to its pool when the last holder drops the block.
pub(crate) struct RowBlock {
    r0: usize,
    rows: usize,
    width: usize,
    scale: usize,
    data: Vec<f32>,
    pool: Arc<BlockPool>,
}

impl RowBlock {
    fn new(pool: &Arc<BlockPool>, scale: usize, r0: usize, rows: usize, width: usize) -> Self {
        let data = pool.take(scale, (CHUNK_ROWS >> scale) * width);
        Self {
            r0,
            rows,
            width,
            scale,
            data,
            pool: Arc::clone(pool),
        }
    }

    pub(crate) fn r0(&self) -> usize {
        self.r0
    }

    /// One past the block's last row.
    pub(crate) fn end(&self) -> usize {
        self.r0 + self.rows
    }

    /// Plane rows `[a, b)`, contiguous.
    pub(crate) fn rows(&self, a: usize, b: usize) -> &[f32] {
        debug_assert!(self.r0 <= a && a <= b && b <= self.end());
        &self.data[(a - self.r0) * self.width..(b - self.r0) * self.width]
    }

    fn rows_mut(&mut self) -> &mut [f32] {
        &mut self.data[..self.rows * self.width]
    }
}

impl Drop for RowBlock {
    fn drop(&mut self) {
        self.pool.put(self.scale, core::mem::take(&mut self.data));
    }
}

/// One chunk's blocks, `[channel][scale]`: `None` where the chunk determines no
/// rows, and for scale-0 X and B when the walk omits them (their rows are only
/// downscaled).
pub(crate) type ChunkPlanes = [[Option<Arc<RowBlock>>; crate::NUM_SCALES]; 3];

/// Produce chunk `k` of `source`: scale-0 rows `[64k, min(64k + 64, h0))` and
/// every row of the deeper scales they determine. `dims` are the walk's plane
/// sizes. With `omit_scale0_xb`, X and B are converted two rows at a time and
/// downscaled at once ([`crate::feature_v2_stream`]'s
/// `convert_side_y_and_downscale`); otherwise the chunk is one conversion
/// call, as the streaming producer's 64-row chunks are.
pub(crate) fn produce_chunk<S: ImageSource>(
    source: &S,
    k: usize,
    dims: &[(usize, usize)],
    omit_scale0_xb: bool,
    revision: FormulaRevision,
    pool: &Arc<BlockPool>,
) -> ChunkPlanes {
    let (w0, h0) = dims[0];
    let r0 = k * CHUNK_ROWS;
    debug_assert!(r0 < h0);
    let n0 = CHUNK_ROWS.min(h0 - r0);
    let mut planes: [[Option<RowBlock>; crate::NUM_SCALES]; 3] = Default::default();
    let mut y = RowBlock::new(pool, 0, r0, n0, w0);
    if omit_scale0_xb {
        let w1 = dims[1].0;
        let mut x1 = RowBlock::new(pool, 1, r0 / 2, n0 / 2, w1);
        let mut b1 = RowBlock::new(pool, 1, r0 / 2, n0 / 2, w1);
        let mut xs = vec![0.0f32; 2 * w0];
        let mut bs = vec![0.0f32; 2 * w0];
        let (ys, x1s, b1s) = (y.rows_mut(), x1.rows_mut(), b1.rows_mut());
        for dy in (0..n0).step_by(2) {
            let count = (n0 - dy).min(2);
            let n = w0 * count;
            crate::streaming::convert_source_to_xyb_into_slices(
                &SubsetView::new(source, r0 + dy, count),
                &mut xs[..n],
                &mut ys[dy * w0..dy * w0 + n],
                &mut bs[..n],
                w0,
                false,
                r0 + dy,
                revision,
            );
            if count == 2 {
                let out = (dy / 2) * w1;
                crate::blur::downscale_2x_into(&xs[..n], w0, &mut x1s[out..out + w1], w1, 1);
                crate::blur::downscale_2x_into(&bs[..n], w0, &mut b1s[out..out + w1], w1, 1);
            }
        }
        if n0 >= 2 {
            planes[0][1] = Some(x1);
            planes[2][1] = Some(b1);
        }
    } else {
        let mut x = RowBlock::new(pool, 0, r0, n0, w0);
        let mut b = RowBlock::new(pool, 0, r0, n0, w0);
        crate::streaming::convert_source_to_xyb_into_slices(
            &SubsetView::new(source, r0, n0),
            x.rows_mut(),
            y.rows_mut(),
            b.rows_mut(),
            w0,
            false,
            r0,
            revision,
        );
        planes[0][0] = Some(x);
        planes[2][0] = Some(b);
    }
    planes[1][0] = Some(y);
    // `ZENSIM_PYRAMID=linear`: every coarse level from 2×2 averages of the
    // parent's linear rows, then positive XYB (the streaming producer's form).
    if crate::feature_v2_stream::linear_pyramid() {
        debug_assert!(!omit_scale0_xb);
        let mut lin: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; n0 * w0]);
        {
            let [r, g, b] = &mut lin;
            crate::feature_v2_stream::srgb8_rows_to_linear(
                source,
                r0,
                n0,
                w0,
                [&mut r[..], &mut g[..], &mut b[..]],
            );
        }
        let (mut lr0, mut lrows) = (r0, n0);
        let mut buf = Vec::new();
        for s in 1..dims.len() {
            let (ws, hs) = dims[s];
            let pw = dims[s - 1].0;
            let (c0, c1) = (lr0 / 2, ((lr0 + lrows) / 2).min(hs));
            if c1 <= c0 {
                break;
            }
            debug_assert_eq!(c0 * 2, lr0);
            let n = c1 - c0;
            let next: [Vec<f32>; 3] = std::array::from_fn(|ch| {
                let mut v = vec![0.0; n * ws];
                crate::blur::downscale_2x_into(&lin[ch][..2 * n * pw], pw, &mut v, ws, n);
                v
            });
            let mut blocks: [RowBlock; 3] =
                std::array::from_fn(|_| RowBlock::new(pool, s, c0, n, ws));
            {
                let [bx, by, bb] = &mut blocks;
                crate::feature_v2_stream::linear_rows_to_xyb(
                    [&next[0][..], &next[1][..], &next[2][..]],
                    n,
                    ws,
                    [bx.rows_mut(), by.rows_mut(), bb.rows_mut()],
                    revision,
                    &mut buf,
                );
            }
            for (ch, block) in blocks.into_iter().enumerate() {
                planes[ch][s] = Some(block);
            }
            (lin, lr0, lrows) = (next, c0, n);
        }
    }
    // The cascade: child row `j` reads parent rows `2j` and `2j + 1`, and the
    // streaming producer's target `min(parent rows / 2, plane height)`.
    for s in 1..dims.len() {
        let (ws, hs) = dims[s];
        let pw = dims[s - 1].0;
        for scales in &mut planes {
            if scales[s].is_some() {
                continue;
            }
            let Some(parent) = scales[s - 1].as_ref() else {
                continue;
            };
            let c0 = parent.r0() / 2;
            let c1 = (parent.end() / 2).min(hs);
            if c1 <= c0 {
                continue;
            }
            let n = c1 - c0;
            let mut child = RowBlock::new(pool, s, c0, n, ws);
            crate::blur::downscale_2x_into(
                parent.rows(parent.r0(), parent.r0() + 2 * n),
                pw,
                child.rows_mut(),
                ws,
                n,
            );
            scales[s] = Some(child);
        }
    }
    planes.map(|scales| scales.map(|b| b.map(Arc::new)))
}

/// Copy a strip's wide window out of `blocks` (consecutive blocks of the
/// strip's plane covering every row the window reads): buffer row `i` is plane
/// row `reflect_101(y0 − HALO_P + i, plane_h)`, the rows
/// [`crate::feature_v2::gather_strip_halo`] copies.
pub(crate) fn gather_window(blocks: &[Arc<RowBlock>], info: &StripInfo, dst: &mut [f32]) {
    let w = info.plane_w;
    let wide_h = info.wide_h();
    debug_assert!(dst.len() >= w * wide_h);
    let span = CHUNK_ROWS >> info.scale;
    let first = blocks[0].r0();
    let row_at = |i: usize| {
        reflect_101(
            info.y0 as isize - HALO_P as isize + i as isize,
            info.plane_h,
        )
    };
    let mut i = 0;
    while i < wide_h {
        let gy = row_at(i);
        let block = &blocks[(gy - first) / span];
        debug_assert!(block.r0() <= gy && gy < block.end());
        // The longest run of consecutive plane rows inside this block.
        let mut run = 1;
        while i + run < wide_h && gy + run < block.end() && row_at(i + run) == gy + run {
            run += 1;
        }
        dst[i * w..(i + run) * w].copy_from_slice(block.rows(gy, gy + run));
        i += run;
    }
}
