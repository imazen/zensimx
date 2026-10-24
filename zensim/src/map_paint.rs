//! Research painting modes for the candidate attribution map
//! (`BakeScorer::compute_with_ref_and_attribution`; exploration record
//! `benchmarks/byconstruction_2026-10-22.md`, "Maps").
//!
//! The map adds, per pyramid scale, a pixel-allocated plane (`id`) and a
//! window-supported plane (`win`: signals that summarize an 11-tap window,
//! such as the SSIM error), then paints the scale onto the source grid. Two
//! choices shape what the map looks like:
//!
//! - **window** — where a windowed signal's mass goes. `box` (the default,
//!   unchanged bits) spreads it uniformly over its window, so one defect
//!   becomes an 11-tap plateau at its scale (88 source pixels at scale 3).
//!   `center` keeps it at the window's own pixel, as DVIFM keeps a block's
//!   term at its block. `guided` sends it back to the window's pixels in
//!   proportion to the guide, so the mass lands where the two images actually
//!   differ. Every mode conserves the plane's sum.
//! - **paint** — how a coarse pixel covers its `2^s × 2^s` source footprint.
//!   `nearest` (the default) spreads it uniformly, as DVIFM paints a block.
//!   `guided` splits it in proportion to the source-resolution guide inside
//!   the footprint (per-pixel refinement); it applies only to a per-pixel map
//!   (`bin == 1`).
//! - **guide** — what the guided modes weigh a pixel by: its squared
//!   difference (`sq`, the default; each channel over the reference's
//!   variance at that scale) or its absolute difference (`abs`; each channel
//!   over the standard deviation). A squared guide squares the contrast of
//!   fine structure: a row-dithered gradient's error, whose row means vary
//!   by 40%, paints coarse mass onto alternate rows at twice that.
//! - **split** — how a pooled feature's value is divided among its pixels.
//!   `repair` (the default, today's map) gives each pixel its first-order
//!   removal effect, what the feature loses when that one pixel is made
//!   perfect: exact for mean pools, `1/p` of the feature in total for an Lp
//!   pool, and for the SSIM deviation pools a near-uniform field (making a
//!   pixel perfect moves it away from the mean) whose total is not the
//!   feature. `exact` gives each pixel its share of the feature, as DVIFM
//!   splits its block terms: `xᵖ/Σxᵖ` of an Lp pool (Euler's theorem), `d·∂f/∂d`
//!   of a deviation pool, `w·v/W` of a soft peak and the secant share of the
//!   global and HF-gain ratios, so every such pool's map sums to its feature.
//!   Linearized cross-scale terms (edge width) and maxima keep their repair
//!   form in both.
//!
//! A scorer selects them per map (`BakeScorer::with_map_paint`,
//! `with_map_guide`, `with_map_smooth`); research command lines can set
//! `ZENSIM_MAP_WINDOW=box|center|guided`, `ZENSIM_MAP_PAINT=nearest|guided`,
//! `ZENSIM_MAP_SPLIT=repair|exact`, `ZENSIM_MAP_GUIDE=sq|abs` and
//! `ZENSIM_MAP_SMOOTH=<σ>` instead. Nothing in the score path reads them.

/// Where a window-supported signal's mass goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowMode {
    Box,
    Center,
    Guided,
}

/// How a coarse pixel covers its source footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PaintMode {
    Nearest,
    Guided,
}

/// What the guided modes weigh a pixel by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuideNorm {
    /// The squared difference, each channel over the reference's variance.
    Squared,
    /// The absolute difference, each channel over the reference's standard
    /// deviation.
    Absolute,
}

impl GuideNorm {
    /// Parse `sq|abs`.
    pub(crate) fn parse(norm: &str) -> Option<Self> {
        match norm {
            "sq" => Some(Self::Squared),
            "abs" => Some(Self::Absolute),
            _ => None,
        }
    }
}

/// How a pooled feature's value is divided among its pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SplitMode {
    /// Each pixel's first-order removal effect.
    Repair,
    /// Each pixel's share of the feature; the shares sum to it.
    Exact,
    /// For steering (research): each nonlinear pool (Lp and L8 norms, soft
    /// peaks, deviation pools) distributed as a mean pool of its per-pixel
    /// quantity, `−s·v_i/N` (`−s·|v_i − μ|/N` for a deviation pool). A pixel's
    /// value then responds to its own error as the pool does to a uniform
    /// change, where a share of an Lp pool goes as `v^p` locally and as `v`
    /// globally; for uniform damage the total equals the exact split's.
    /// Linear, weighted, ratio and global terms keep their exact shares.
    Amplitude,
}

/// The painting choices of one map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MapPaint {
    pub(crate) window: WindowMode,
    pub(crate) paint: PaintMode,
    pub(crate) split: SplitMode,
    /// What the guided window and paint weigh a pixel by.
    pub(crate) guide: GuideNorm,
    /// The final per-pixel map's perceptual integration: a sum-preserving
    /// Gaussian of this σ in thousandths of a source pixel (0: none). The eye
    /// integrates error over a few pixels; without it, per-pixel terms (the
    /// squared error, the JND transducers) show a row-dithered gradient's
    /// error as one stripe per row.
    pub(crate) smooth_milli: u16,
}

impl Default for MapPaint {
    /// The uniform window spread, nearest-neighbour fold and removal split:
    /// today's map.
    fn default() -> Self {
        Self {
            window: WindowMode::Box,
            paint: PaintMode::Nearest,
            split: SplitMode::Repair,
            guide: GuideNorm::Squared,
            smooth_milli: 0,
        }
    }
}

impl MapPaint {
    /// Parse `box|center|guided`, `nearest|guided` and `repair|exact|amplitude`.
    pub(crate) fn parse(window: &str, paint: &str, split: &str) -> Option<Self> {
        let window = match window {
            "box" => WindowMode::Box,
            "center" => WindowMode::Center,
            "guided" => WindowMode::Guided,
            _ => return None,
        };
        let paint = match paint {
            "nearest" => PaintMode::Nearest,
            "guided" => PaintMode::Guided,
            _ => return None,
        };
        let split = match split {
            "repair" => SplitMode::Repair,
            "exact" => SplitMode::Exact,
            "amplitude" => SplitMode::Amplitude,
            _ => return None,
        };
        Some(Self {
            window,
            paint,
            split,
            ..Self::default()
        })
    }

    /// `σ` in source pixels as [`Self::smooth_milli`]; `None` outside `[0, 16]`.
    pub(crate) fn smooth_milli_of(sigma: f32) -> Option<u16> {
        (sigma.is_finite() && (0.0..=16.0).contains(&sigma))
            .then(|| (sigma * 1000.0).round() as u16)
    }

    /// The process's research selection (`ZENSIM_MAP_WINDOW`,
    /// `ZENSIM_MAP_PAINT`, `ZENSIM_MAP_SPLIT`, `ZENSIM_MAP_GUIDE`,
    /// `ZENSIM_MAP_SMOOTH`), read once; unset means [`Self::default`] and an
    /// unknown value panics.
    pub(crate) fn from_env() -> Self {
        use std::sync::OnceLock;
        static MODES: OnceLock<MapPaint> = OnceLock::new();
        *MODES.get_or_init(|| {
            let window = std::env::var("ZENSIM_MAP_WINDOW").unwrap_or_else(|_| "box".into());
            let paint = std::env::var("ZENSIM_MAP_PAINT").unwrap_or_else(|_| "nearest".into());
            let split = std::env::var("ZENSIM_MAP_SPLIT").unwrap_or_else(|_| "repair".into());
            let smooth = std::env::var("ZENSIM_MAP_SMOOTH").unwrap_or_else(|_| "0".into());
            let guide = std::env::var("ZENSIM_MAP_GUIDE").unwrap_or_else(|_| "sq".into());
            let guide = GuideNorm::parse(&guide)
                .unwrap_or_else(|| panic!("ZENSIM_MAP_GUIDE={guide:?}: expected sq|abs"));
            let smooth_milli = smooth.parse::<f32>().ok().and_then(Self::smooth_milli_of).unwrap_or_else(|| {
                panic!("ZENSIM_MAP_SMOOTH={smooth:?}: expected a sigma in 0..=16 source pixels")
            });
            Self::parse(&window, &paint, &split).map(|p| Self { smooth_milli, guide, ..p }).unwrap_or_else(|| {
                panic!(
                    "ZENSIM_MAP_WINDOW={window:?} / ZENSIM_MAP_PAINT={paint:?} / ZENSIM_MAP_SPLIT={split:?}: \
                     expected box|center|guided, nearest|guided and repair|exact|amplitude"
                )
            })
        })
    }
}

/// Clipped box sum of radius `r` (each output sums the in-bounds inputs of
/// its `(2r+1)²` window), in f64. Each window is summed afresh, separably,
/// rather than as a running sum: a running sum subtracts every value it
/// added, so a large value leaves its rounding error behind in every later
/// window of small ones (the guide spans many decades).
fn box_sum(src: &[f64], w: usize, h: usize, r: usize, out: &mut Vec<f64>) {
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            tmp[y * w + x] = row[x.saturating_sub(r)..(x + r + 1).min(w)].iter().sum();
        }
    }
    out.clear();
    out.resize(w * h, 0.0);
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (y.saturating_sub(r)..(y + r + 1).min(h))
                .map(|yy| tmp[yy * w + x])
                .sum();
        }
    }
}

/// Sum-preserving separable Gaussian of `σ = sigma` pixels over a `w × h` map:
/// each value is scattered to its in-bounds neighbours with the kernel
/// renormalized over them, so the map's total (and every row's, then every
/// column's) is kept exactly; radius `⌈3σ⌉`.
pub(crate) fn smooth_sum_preserving(map: &mut [f64], w: usize, h: usize, sigma: f64) {
    if sigma <= 0.0 || map.is_empty() {
        return;
    }
    let r = (3.0 * sigma).ceil() as usize;
    let k: Vec<f64> = (0..=r)
        .map(|d| (-((d * d) as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let scatter = |src: &[f64], dst: &mut [f64], n: usize, stride: usize, base: usize| {
        for i in 0..n {
            let v = src[base + i * stride];
            if v == 0.0 {
                continue;
            }
            let (lo, hi) = (i.saturating_sub(r), (i + r).min(n - 1));
            let z: f64 = (lo..=hi).map(|j| k[i.abs_diff(j)]).sum();
            for j in lo..=hi {
                dst[base + j * stride] += v * k[i.abs_diff(j)] / z;
            }
        }
    };
    let mut tmp = vec![0.0f64; map.len()];
    for y in 0..h {
        scatter(map, &mut tmp, w, 1, y * w);
    }
    map.fill(0.0);
    for x in 0..w {
        scatter(&tmp, map, h, w, x);
    }
}

/// The guide: per-pixel squared (or absolute) difference, each channel
/// normalized by the reference's variance (or standard deviation) on this
/// plane, plus a small uniform floor so a window without any difference
/// still keeps (and spreads) its mass.
pub(crate) fn difference_guide(
    src: [&[f32]; 3],
    dst: [&[f32]; 3],
    n: usize,
    norm: GuideNorm,
) -> Vec<f64> {
    let mut g = vec![0.0f64; n];
    for c in 0..3 {
        let (s, d) = (src[c], dst[c]);
        if s.len() < n || d.len() < n {
            continue; // a channel this walk did not keep (scale-0 chroma)
        }
        let mean = s[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = s[..n]
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        match norm {
            GuideNorm::Squared => {
                let k = 1.0 / (var + 1e-6);
                for i in 0..n {
                    let e = (s[i] - d[i]) as f64;
                    g[i] += k * e * e;
                }
            }
            GuideNorm::Absolute => {
                let k = 1.0 / (var + 1e-6).sqrt();
                for i in 0..n {
                    g[i] += k * ((s[i] - d[i]) as f64).abs();
                }
            }
        }
    }
    let mean = g.iter().sum::<f64>() / n as f64;
    let floor = 1e-3 * mean + 1e-30;
    for v in &mut g {
        *v += floor;
    }
    g
}

/// Merge `win` into `id` under `mode`. `Box` is the existing fused spread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_window(
    mode: WindowMode,
    win: &mut [f32],
    id: &mut [f32],
    w: usize,
    h: usize,
    radius: usize,
    guide: impl FnOnce() -> Vec<f64>,
    spread_tmp: &mut Vec<f32>,
    spread_out: &mut Vec<f32>,
    parallel: bool,
) {
    let n = w * h;
    match mode {
        WindowMode::Box => crate::blur::box_spread_merge_f32(
            &mut win[..n],
            &mut id[..n],
            w,
            h,
            radius,
            spread_tmp,
            spread_out,
            parallel,
        ),
        WindowMode::Center => {
            for i in 0..n {
                id[i] += win[i];
            }
            win[..n].fill(0.0);
        }
        WindowMode::Guided => {
            // out(q) = g(q) · Σ_{p ∋ q} win(p) / Σ_{q' ∈ W(p)} g(q').
            let g = guide();
            let mut s = Vec::new();
            box_sum(&g, w, h, radius, &mut s);
            let ratio: Vec<f64> = (0..n).map(|i| win[i] as f64 / s[i]).collect();
            let mut back = Vec::new();
            box_sum(&ratio, w, h, radius, &mut back);
            for i in 0..n {
                id[i] += (g[i] * back[i]) as f32;
            }
            win[..n].fill(0.0);
        }
    }
}

/// Paint a scale plane onto a `w0 × h0` source grid, splitting each coarse
/// pixel over its `2^scale` footprint in proportion to `guide0` (source
/// resolution); footprint cells past the image edge are dropped, as the
/// nearest-neighbour fold drops them.
pub(crate) fn paint_guided(
    plane: &[f32],
    ws: usize,
    hs: usize,
    scale: usize,
    guide0: &[f64],
    w0: usize,
    h0: usize,
) -> Vec<f32> {
    let f = 1usize << scale;
    let mut out = vec![0.0f32; w0 * h0];
    for cy in 0..hs {
        let y0 = cy * f;
        if y0 >= h0 {
            break;
        }
        let y1 = (y0 + f).min(h0);
        for cx in 0..ws {
            let x0 = cx * f;
            if x0 >= w0 {
                break;
            }
            let x1 = (x0 + f).min(w0);
            let m = plane[cy * ws + cx] as f64;
            if m == 0.0 {
                continue;
            }
            let mut total = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    total += guide0[y * w0 + x];
                }
            }
            for y in y0..y1 {
                for x in x0..x1 {
                    out[y * w0 + x] += (m * guide0[y * w0 + x] / total) as f32;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(w: usize, h: usize) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
        let s: [Vec<f32>; 3] = core::array::from_fn(|c| {
            (0..w * h)
                .map(|i| 0.2 + 0.01 * c as f32 + ((i * 7 % 13) as f32) * 0.01)
                .collect()
        });
        let mut d = s.clone();
        d[1][(h / 2) * w + w / 2] += 0.3; // one defect pixel
        (s, d)
    }

    #[test]
    fn guide_norms_scale_as_their_powers() {
        // One kept channel; doubling the difference quadruples the squared
        // guide and doubles the absolute one (the floor scales with the mean).
        let n = 64;
        let s: Vec<f32> = (0..n).map(|i| (i % 8) as f32 * 0.1).collect();
        let shifted = |e: f32| -> Vec<f32> {
            s.iter()
                .enumerate()
                .map(|(i, &v)| v + if i % 3 == 0 { e } else { 0.0 })
                .collect()
        };
        let (d1, d2) = (shifted(0.01), shifted(0.02));
        let none: [f32; 0] = [];
        for (norm, want) in [(GuideNorm::Squared, 4.0), (GuideNorm::Absolute, 2.0)] {
            let g1 = difference_guide([&s, &none, &none], [&d1, &none, &none], n, norm);
            let g2 = difference_guide([&s, &none, &none], [&d2, &none, &none], n, norm);
            let ratio = g2[0] / g1[0];
            assert!(
                (ratio - want).abs() < 1e-3 * want,
                "{norm:?}: ratio {ratio}"
            );
            assert!(
                g1[1] > 0.0 && g1[1] < 1e-2 * g1[0],
                "{norm:?}: floor {}",
                g1[1]
            );
        }
    }

    #[test]
    fn smoothing_conserves_mass_and_zero_is_identity() {
        let (w, h) = (37, 29);
        let map: Vec<f64> = (0..w * h)
            .map(|i| ((i * 31 % 17) as f64) * 0.01 - 0.05)
            .collect();
        let total: f64 = map.iter().sum();
        for sigma in [0.5, 1.0, 1.5, 3.0, 8.0] {
            let mut m = map.clone();
            smooth_sum_preserving(&mut m, w, h, sigma);
            let t: f64 = m.iter().sum();
            assert!(
                (t - total).abs() <= 1e-12 * total.abs().max(1.0),
                "sigma {sigma}: {t} vs {total}"
            );
        }
        let mut m = map.clone();
        smooth_sum_preserving(&mut m, w, h, 0.0);
        assert_eq!(m, map);
        // A row-alternating pattern is flattened.
        let mut stripes: Vec<f64> = (0..w * h).map(|i| ((i / w) % 2) as f64).collect();
        smooth_sum_preserving(&mut stripes, w, h, 1.5);
        let row = |y: usize| stripes[y * w + w / 2];
        assert!(
            (row(h / 2) - row(h / 2 + 1)).abs() < 0.05,
            "{} vs {}",
            row(h / 2),
            row(h / 2 + 1)
        );
    }

    #[test]
    fn every_window_mode_conserves_mass() {
        let (w, h) = (37, 29);
        let (s, d) = planes(w, h);
        let win0: Vec<f32> = (0..w * h).map(|i| ((i * 31 % 17) as f32) * 0.01).collect();
        let total: f64 = win0.iter().map(|&v| v as f64).sum();
        for mode in [WindowMode::Box, WindowMode::Center, WindowMode::Guided] {
            let mut win = win0.clone();
            let mut id = vec![0.0f32; w * h];
            let (mut t, mut o) = (Vec::new(), Vec::new());
            let sr = [&s[0][..], &s[1][..], &s[2][..]];
            let dr = [&d[0][..], &d[1][..], &d[2][..]];
            merge_window(
                mode,
                &mut win,
                &mut id,
                w,
                h,
                5,
                || difference_guide(sr, dr, w * h, GuideNorm::Squared),
                &mut t,
                &mut o,
                false,
            );
            let got: f64 = id.iter().map(|&v| v as f64).sum();
            assert!(
                (got - total).abs() <= 1e-4 * total,
                "{mode:?}: {got} vs {total}"
            );
        }
    }

    #[test]
    fn guided_window_mass_concentrates_on_the_difference() {
        let (w, h) = (41, 41);
        let (s, d) = planes(w, h);
        // A plateau: every pixel whose window holds the defect carries error.
        let (cx, cy) = (w / 2, h / 2);
        let mut win = vec![0.0f32; w * h];
        for y in cy - 5..=cy + 5 {
            for x in cx - 5..=cx + 5 {
                win[y * w + x] = 1.0;
            }
        }
        let mut id = vec![0.0f32; w * h];
        let (mut t, mut o) = (Vec::new(), Vec::new());
        let sr = [&s[0][..], &s[1][..], &s[2][..]];
        let dr = [&d[0][..], &d[1][..], &d[2][..]];
        merge_window(
            WindowMode::Guided,
            &mut win,
            &mut id,
            w,
            h,
            5,
            || difference_guide(sr, dr, w * h, GuideNorm::Squared),
            &mut t,
            &mut o,
            false,
        );
        let total: f32 = id.iter().sum();
        assert!(
            id[cy * w + cx] > 0.9 * total,
            "defect pixel holds {} of {total}",
            id[cy * w + cx]
        );
    }

    #[test]
    fn guided_paint_conserves_each_footprint() {
        let (w0, h0) = (21, 13);
        let scale = 2;
        let (ws, hs) = (w0 >> scale, h0 >> scale);
        let plane: Vec<f32> = (0..ws * hs).map(|i| 1.0 + i as f32).collect();
        let guide0: Vec<f64> = (0..w0 * h0).map(|i| 0.5 + (i % 5) as f64).collect();
        let out = paint_guided(&plane, ws, hs, scale, &guide0, w0, h0);
        let f = 1 << scale;
        for cy in 0..hs {
            for cx in 0..ws {
                let mut sum = 0.0f64;
                for y in cy * f..cy * f + f {
                    for x in cx * f..cx * f + f {
                        sum += out[y * w0 + x] as f64;
                    }
                }
                let want = plane[cy * ws + cx] as f64;
                assert!(
                    (sum - want).abs() <= 1e-5 * want,
                    "cell {cx},{cy}: {sum} vs {want}"
                );
            }
        }
    }
}
