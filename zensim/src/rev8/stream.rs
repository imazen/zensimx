use core::ops::Range;

#[cfg(feature = "threads")]
use rayon::prelude::*;

use super::{COMMON, Features, RgbRows, Variant, kernels::RowKernels};

pub(super) const RADIUS: usize = 5;
pub(super) const LANES: usize = 16;
pub(super) const MOMENTS: usize = 6;
pub(super) const PLANES: usize = 6;
const TAPS: usize = 2 * RADIUS + 1;
const RING: usize = TAPS;
const PARALLEL_PIXELS: usize = 192 * 192;
pub(super) const TAU: f64 = 0.05;

pub(super) type Moments = [Vec<f32>; MOMENTS];
type PixelRow = [Vec<f32>; PLANES];
type StripTotals = [[RowTotals; 3]; 5];

pub(super) struct PixelStencil<'a> {
    pub before: [&'a [f32]; 2],
    pub center: [&'a [f32]; 2],
    pub after: [&'a [f32]; 2],
    pub width: usize,
    pub height: usize,
    pub y: usize,
}

#[derive(Clone, Copy, Default)]
pub(super) struct RowTotals {
    pub features: Features,
    pub orientation: [f64; 2],
    pub forward: [f64; 4],
    pub reverse: [f64; 4],
    #[cfg(test)]
    pub orientation_parts: [[f64; 3]; 2],
    #[cfg(test)]
    pub mean_fragility: f64,
}

impl RowTotals {
    fn merge(&mut self, other: Self) {
        for (own, rhs) in self.features[..COMMON].iter_mut().zip(other.features) {
            *own += rhs;
        }
        for (own, rhs) in self.orientation.iter_mut().zip(other.orientation) {
            *own += rhs;
        }
        for (own, rhs) in self.forward.iter_mut().zip(other.forward) {
            *own += rhs;
        }
        for (own, rhs) in self.reverse.iter_mut().zip(other.reverse) {
            *own += rhs;
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Geometry {
    width: usize,
    height: usize,
}

impl Geometry {
    fn next(self) -> Self {
        Self {
            width: self.width.div_ceil(2),
            height: self.height.div_ceil(2),
        }
    }
}

#[cfg(feature = "feature-regime-v2")]
use crate::feature_v2::reflect_101;

// The same periodic boundary map for a minimal rev8-proto build.
#[cfg(not(feature = "feature-regime-v2"))]
fn reflect_101(i: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let last = (len - 1) as isize;
    let phase = i.rem_euclid(2 * last);
    phase.min(2 * last - phase) as usize
}

fn y_tap(y: isize, height: usize) -> usize {
    if (y as usize) < height {
        y as usize
    } else {
        reflect_101(y, height)
    }
}

fn planes<const N: usize>(len: usize) -> [Vec<f32>; N] {
    core::array::from_fn(|_| vec![0.0; len])
}

struct Level {
    geometry: Geometry,
    padded: usize,
    pixels: Vec<PixelRow>,
    blurred: Vec<[Moments; 3]>,
    horizontal_scratch: Moments,
    // Each channel has a separate vertical recurrence.
    vertical_bank: [Moments; 3],
    input: Range<usize>,
    output: Range<usize>,
    next_output: usize,
}

impl Level {
    fn new(geometry: Geometry) -> Self {
        let padded = geometry.width.next_multiple_of(LANES);
        let pixel_len = padded + LANES;
        let ring = geometry.height.min(RING);
        Self {
            geometry,
            padded,
            pixels: (0..ring).map(|_| planes(pixel_len)).collect(),
            blurred: (0..ring)
                .map(|_| core::array::from_fn(|_| planes(padded)))
                .collect(),
            horizontal_scratch: planes(pixel_len),
            vertical_bank: core::array::from_fn(|_| planes(padded)),
            input: 0..0,
            output: 0..0,
            next_output: 0,
        }
    }

    fn reset(&mut self, output: Range<usize>, child_input: Option<&Range<usize>>) {
        self.input =
            output.start.saturating_sub(RADIUS)..(output.end + RADIUS).min(self.geometry.height);
        if let Some(child) = child_input {
            self.input.start = self.input.start.min(child.start * 2);
            self.input.end = self
                .input
                .end
                .max((child.end * 2).min(self.geometry.height));
        }
        self.next_output = output.start;
        self.output = output;
    }

    fn padded_row(&mut self, y: usize) -> &mut PixelRow {
        let slot = y % self.pixels.len();
        &mut self.pixels[slot]
    }

    fn load_base<K: RowKernels>(&mut self, math: K, y: usize, a: RgbRows<'_>, b: RgbRows<'_>) {
        let row = self.padded_row(y);
        let (a_channels, b_channels) = row.split_at_mut(3);
        math.color(a.row(y), a_channels);
        math.color(b.row(y), b_channels);
    }

    fn downsample(&mut self, parent: &Self, y: usize) {
        let width = self.geometry.width;
        let parent_width = parent.geometry.width;
        let top = (2 * y) % parent.pixels.len();
        let bottom = (2 * y + 1).min(parent.geometry.height - 1) % parent.pixels.len();
        let out = self.padded_row(y);
        for (channel, plane) in out.iter_mut().enumerate() {
            let a = &parent.pixels[top][channel][RADIUS..RADIUS + parent_width];
            let b = &parent.pixels[bottom][channel][RADIUS..RADIUS + parent_width];
            let (pairs_a, tail_a) = a.as_chunks::<2>();
            let (pairs_b, tail_b) = b.as_chunks::<2>();
            let row = &mut plane[RADIUS..RADIUS + width];
            for (([a0, a1], [b0, b1]), value) in pairs_a.iter().zip(pairs_b).zip(&mut *row) {
                *value = ((*a0 + *a1) + (*b0 + *b1)) * 0.25;
            }
            if let ([a], [b]) = (tail_a, tail_b) {
                row[width - 1] = ((*a + *a) + (*b + *b)) * 0.25;
            }
        }
    }

    fn push<K: RowKernels, const HYBRID: bool>(
        &mut self,
        math: K,
        y: usize,
        totals: &mut [RowTotals; 3],
    ) {
        let Geometry { width, height } = self.geometry;
        // The propagated coarse-scale halo can extend beyond this level's
        // output halo. Those rows feed downsampling without moment work.
        if y < self.output.start.saturating_sub(RADIUS) || y >= self.output.end + RADIUS {
            return;
        }
        let slot = y % self.pixels.len();
        let row = &mut self.pixels[slot];
        for x in (0..RADIUS).chain(RADIUS + width..self.padded + LANES) {
            let source = RADIUS + reflect_101(x as isize - RADIUS as isize, width);
            for plane in &mut *row {
                plane[x] = plane[source];
            }
        }
        for c in 0..3 {
            math.horizontal(
                &row[c],
                &row[c + 3],
                super::kernels::channel_center(c),
                &mut self.horizontal_scratch,
                &mut self.blurred[slot][c],
                self.padded,
            );
        }

        while self.next_output < self.output.end && (self.next_output + RADIUS).min(height - 1) <= y
        {
            let center = self.next_output as isize;
            let ring = self.pixels.len();
            let taps = core::array::from_fn::<_, TAPS, _>(|i| {
                y_tap(center + i as isize - RADIUS as isize, height) % ring
            });
            let [before, current, after] =
                [-1, 0, 1].map(|delta| &self.pixels[y_tap(center + delta, height) % ring]);
            for (c, total) in totals.iter_mut().enumerate() {
                math.vertical(
                    &taps.map(|i| &self.blurred[i][c]),
                    &mut self.vertical_bank[c],
                    self.padded,
                );
                math.terms::<HYBRID>(
                    &self.vertical_bank[c],
                    PixelStencil {
                        before: [&before[c], &before[c + 3]],
                        center: [&current[c], &current[c + 3]],
                        after: [&after[c], &after[c + 3]],
                        width,
                        height,
                        y: self.next_output,
                    },
                    c,
                    total,
                );
            }
            self.next_output += 1;
        }
    }
}

struct Strip {
    levels: Vec<Level>,
}

impl Strip {
    fn new(geometry: &[Geometry]) -> Self {
        Self {
            levels: geometry.iter().copied().map(Level::new).collect(),
        }
    }

    fn scan<K: RowKernels, const HYBRID: bool>(
        &mut self,
        math: K,
        a: RgbRows<'_>,
        b: RgbRows<'_>,
        rows: Range<usize>,
    ) -> StripTotals {
        let mut child = None;
        for (s, level) in self.levels.iter_mut().enumerate().rev() {
            let output = (rows.start >> s)..rows.end.div_ceil(1 << s);
            level.reset(output, child.as_ref());
            child = Some(level.input.clone());
        }

        let mut totals = StripTotals::default();
        for y in self.levels[0].input.clone() {
            self.levels[0].load_base(math, y, a, b);
            let mut row = y;
            for (s, total) in totals[..self.levels.len()].iter_mut().enumerate() {
                let (processed, rest) = self.levels.split_at_mut(s + 1);
                let level = &mut processed[s];
                level.push::<_, HYBRID>(math, row, total);
                let Some(child) = rest.first_mut() else { break };
                let pair_complete = row % 2 == 1 || row == level.geometry.height - 1;
                if !pair_complete || !child.input.contains(&(row / 2)) {
                    break;
                }
                row /= 2;
                child.downsample(level, row);
            }
        }
        debug_assert!(self.levels.iter().all(|l| l.next_output == l.output.end));
        totals
    }
}

fn extract_mode<K: RowKernels, const HYBRID: bool>(
    math: K,
    a: RgbRows<'_>,
    b: RgbRows<'_>,
    scales: usize,
    parallel: bool,
) -> [[Features; 3]; 5] {
    let mut geometry = [Geometry::default(); 5];
    geometry[0] = Geometry {
        width: a.width,
        height: a.height,
    };
    for s in 1..scales {
        geometry[s] = geometry[s - 1].next();
    }
    let active = &geometry[..scales];
    let use_strips = a.width * a.height >= PARALLEL_PIXELS;
    // Fixed partitions are a property of geometry. Both serial and Rayon
    // visit the same core rows and merge in increasing strip order.
    let step = if use_strips {
        a.height.div_ceil(20).next_multiple_of(32).max(32)
    } else {
        a.height
    };
    let ranges = (0..a.height)
        .step_by(step)
        .map(|start| start..(start + step).min(a.height))
        .collect::<Vec<_>>();

    let mut pooled = StripTotals::default();
    let mut merge = |strip: StripTotals| {
        for (pooled, partial) in pooled.iter_mut().zip(strip) {
            for (pooled, partial) in pooled.iter_mut().zip(partial) {
                pooled.merge(partial);
            }
        }
    };

    #[cfg(not(feature = "threads"))]
    let _ = parallel;
    #[cfg(feature = "threads")]
    let threaded = use_strips && parallel && rayon::current_num_threads() > 1 && ranges.len() > 1;
    #[cfg(not(feature = "threads"))]
    let threaded = false;
    if threaded {
        #[cfg(feature = "threads")]
        ranges
            .into_par_iter()
            .map_init(
                || Strip::new(active),
                |strip, rows| strip.scan::<_, HYBRID>(math, a, b, rows),
            )
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(&mut merge);
    } else {
        let mut strip = Strip::new(active);
        for rows in ranges {
            merge(strip.scan::<_, HYBRID>(math, a, b, rows));
        }
    }

    let base_samples = (a.width * a.height) as f64;
    let [phi_a, phi_b] = pooled[0][1].orientation.map(|v| v / base_samples);
    for (scale, shape) in pooled.iter_mut().zip(active) {
        let inv_n = 1.0 / (shape.width * shape.height) as f64;
        for group in scale {
            if HYBRID {
                group.features[COMMON..].copy_from_slice(&blend(
                    phi_a,
                    phi_b,
                    group.forward,
                    group.reverse,
                ));
            }
            for term in &mut group.features {
                *term *= inv_n;
            }
        }
    }
    pooled.map(|scale| scale.map(|group| group.features))
}

pub(super) fn blend(a: f64, b: f64, forward: [f64; 4], reverse: [f64; 4]) -> [f64; 4] {
    // Canonicalize the arithmetic order after the dense walk. Equal
    // orientation uses one commutative addition, including exact ties.
    if a == b {
        return core::array::from_fn(|i| (forward[i] + reverse[i]) * 0.5);
    }
    let (strong, weak) = if a > b {
        (forward, reverse)
    } else {
        (reverse, forward)
    };
    let weight = 1.0 / (1.0 + libm::exp(-(a - b).abs() / TAU));
    core::array::from_fn(|i| weight * strong[i] + (1.0 - weight) * weak[i])
}

pub(super) fn extract<K: RowKernels>(
    math: K,
    a: RgbRows<'_>,
    b: RgbRows<'_>,
    variant: Variant,
    parallel: bool,
) -> [[Features; 3]; 5] {
    if variant.hybrid() {
        extract_mode::<_, true>(math, a, b, variant.scales(), parallel)
    } else {
        extract_mode::<_, false>(math, a, b, variant.scales(), parallel)
    }
}

#[cfg(test)]
pub(super) fn diagnostic(a: RgbRows<'_>, b: RgbRows<'_>) -> RowTotals {
    let geometry = [Geometry {
        width: a.width,
        height: a.height,
    }];
    let mut strip = Strip::new(&geometry);
    let row = strip.scan::<_, true>(
        super::kernels::RowMath(archmage::ScalarToken),
        a,
        b,
        0..a.height,
    )[0][1];
    let n = (a.width * a.height) as f64;
    RowTotals {
        features: row.features.map(|v| v / n),
        orientation: row.orientation.map(|v| v / n),
        orientation_parts: row.orientation_parts.map(|parts| parts.map(|v| v / n)),
        mean_fragility: row.mean_fragility / n,
        ..RowTotals::default()
    }
}
