//! Model-bound experimental XYB sampling. Kernels belong to zenresize;
//! extraction and attribution remain in their existing owners.
#![cfg_attr(
    not(all(feature = "custom-profiles", feature = "feature-regime-v2")),
    allow(dead_code)
)]

use crate::streaming::XybPyramidLevel;
use crate::{ImageSource, ZensimError};
use zenresize::filter::InterpolationDetails;
use zenresize::weights::F32WeightTable;
use zenresize::{Filter, PixelFormat, ResizeConfig, Resizer};

pub(crate) const KEY: &str = "zentrain.sampling";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sampling {
    pub(crate) keep_y: bool,
    kernel: u8,
    num: usize,
    den: usize,
    /// Direct-from-original divisors; None retains the historical cascade.
    divisors: Option<[usize; 4]>,
}

impl Sampling {
    pub(crate) fn from_model(model: &crate::mlp::Model) -> Result<Option<Self>, ZensimError> {
        if model.metadata().get(KEY).is_none() {
            return Ok(None);
        }
        let bad = || ZensimError::ModelLoadFailed {
            reason: "unknown zentrain.sampling contract",
        };
        let value = model.metadata().get_utf8(KEY).map_err(|_| bad())?;
        let parts: Vec<_> = value.split(':').collect();
        if parts.len() != 4 || !matches!(parts[0], "v1" | "v2") {
            return Err(bad());
        }
        let keep_y = match parts[1] {
            "y" => true,
            "xyb" => false,
            _ => return Err(bad()),
        };
        let kernel = match parts[2] {
            "triangle" => 0,
            "mitchell" => 1,
            "robidouxsharp" => 2,
            _ => return Err(bad()),
        };
        if parts[0] == "v2" {
            if keep_y {
                return Err(bad());
            }
            let divisors = match parts[3] {
                "1,2,4,8" => [1, 2, 4, 8],
                "1,3,5,7" => [1, 3, 5, 7],
                "1,2,3,5" => [1, 2, 3, 5],
                _ => return Err(bad()),
            };
            return Ok(Some(Self {
                keep_y,
                kernel,
                num: 1,
                den: 1,
                divisors: Some(divisors),
            }));
        }
        let (num, den) = match parts[3] {
            "3/2" => (3, 2),
            "2" => (2, 1),
            "3" => (3, 1),
            _ => return Err(bad()),
        };
        Ok(Some(Self {
            keep_y,
            kernel,
            num,
            den,
            divisors: None,
        }))
    }

    pub(crate) fn is_direct(self) -> bool {
        self.divisors.is_some()
    }

    fn filter(self) -> Filter {
        match self.kernel {
            0 => Filter::Triangle,
            1 => Filter::Mitchell,
            _ => Filter::RobidouxSharp,
        }
    }

    pub(crate) fn min_dim(self) -> usize {
        if let Some(d) = self.divisors {
            return 8 * d[3];
        }
        // Four real levels, each at least eight pixels on both axes.
        (if self.keep_y { 32 } else { 64 }) * self.num / self.den
    }

    pub(crate) fn dims(self, w: usize, h: usize) -> Vec<(usize, usize)> {
        let (mut w, mut h) = (w.max(self.min_dim()), h.max(self.min_dim()));
        if let Some(d) = self.divisors {
            return d.into_iter().map(|div| (w / div, h / div)).collect();
        }
        let mut dims = Vec::with_capacity(4);
        if !self.keep_y {
            w = w * self.den / self.num;
            h = h * self.den / self.num;
        }
        dims.push((w, h));
        for level in 1..4 {
            if level == 1 && self.keep_y {
                w = w * self.den / self.num;
                h = h * self.den / self.num;
            } else {
                w /= 2;
                h /= 2;
            }
            dims.push((w, h));
        }
        dims
    }

    fn resize(
        self,
        previous: &XybPyramidLevel,
        w: usize,
        h: usize,
        parallel: bool,
        revision: crate::feature_defs::FormulaRevision,
    ) -> XybPyramidLevel {
        if revision >= crate::feature_defs::FormulaRevision::Rev5 {
            // Keep the established zenresize coefficients, but give Rev5 one
            // tier-independent ordered f64 reduction per output, rounded to f32
            // after each separable axis. Historical SIMD resize is unchanged.
            let details = InterpolationDetails::create(self.filter());
            let x = F32WeightTable::new(previous.1 as u32, w as u32, &details);
            let y = F32WeightTable::new(previous.2 as u32, h as u32, &details);
            let planes = core::array::from_fn(|ch| {
                let mut horizontal = vec![0.0f32; w * previous.2];
                for row in 0..previous.2 {
                    for col in 0..w {
                        let mut sum = 0.0f64;
                        for (j, &weight) in x.weights(col).iter().enumerate() {
                            if weight != 0.0 {
                                sum += f64::from(
                                    previous.0[ch][row * previous.1 + x.left[col] as usize + j],
                                ) * f64::from(weight);
                            }
                        }
                        horizontal[row * w + col] = sum as f32;
                    }
                }
                let mut out = vec![0.0f32; w * h];
                for row in 0..h {
                    for col in 0..w {
                        let mut sum = 0.0f64;
                        for (j, &weight) in y.weights(row).iter().enumerate() {
                            if weight != 0.0 {
                                sum += f64::from(horizontal[(y.left[row] as usize + j) * w + col])
                                    * f64::from(weight);
                            }
                        }
                        out[row * w + col] = sum as f32;
                    }
                }
                out
            });
            return (planes, w, h);
        }
        let config =
            ResizeConfig::builder(previous.1 as u32, previous.2 as u32, w as u32, h as u32)
                .format(PixelFormat::GRAYF32_LINEAR)
                .filter(self.filter())
                .build();
        #[cfg(feature = "threads")]
        if parallel && previous.1 * previous.2 >= 65_536 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            let mut planes: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; w * h]);
            planes.par_iter_mut().enumerate().for_each(|(c, out)| {
                Resizer::new(&config).resize_f32_into(&previous.0[c], out);
            });
            return (planes, w, h);
        }
        let _ = parallel;
        let mut resizer = Resizer::new(&config);
        let planes = core::array::from_fn(|c| {
            let mut out = vec![0.0; w * h];
            resizer.resize_f32_into(&previous.0[c], &mut out);
            out
        });
        (planes, w, h)
    }

    /// `revision` is the walk's formula revision (canonical opsin at Rev4).
    pub(crate) fn pyramid(
        self,
        source: &(impl ImageSource + ?Sized),
        parallel: bool,
        revision: crate::feature_defs::FormulaRevision,
    ) -> Vec<XybPyramidLevel> {
        let dims = self.dims(source.width(), source.height());
        let padded;
        let planes = if source.width() < self.min_dim() || source.height() < self.min_dim() {
            padded = crate::metric::reflect_pad_to_size(source, self.min_dim());
            crate::streaming::convert_source_to_xyb(&padded, padded.width(), parallel, revision)
        } else {
            crate::streaming::convert_source_to_xyb(source, source.width(), parallel, revision)
        };
        let original = (
            planes,
            source.width().max(self.min_dim()),
            source.height().max(self.min_dim()),
        );
        if self.is_direct() {
            let mut levels: Vec<_> = dims[1..]
                .iter()
                .map(|&(w, h)| self.resize(&original, w, h, parallel, revision))
                .collect();
            levels.insert(0, original);
            return levels;
        }
        let first = if self.keep_y {
            original
        } else {
            self.resize(&original, dims[0].0, dims[0].1, parallel, revision)
        };
        let mut levels = vec![first];
        for &(w, h) in &dims[1..] {
            levels.push(self.resize(levels.last().unwrap(), w, h, parallel, revision));
        }
        levels
    }

    pub(crate) fn reference(
        self,
        source: &(impl ImageSource + ?Sized),
        parallel: bool,
        revision: crate::feature_defs::FormulaRevision,
    ) -> crate::PrecomputedReference {
        crate::PrecomputedReference {
            formula_revision: revision,
            scales: self.pyramid(source, parallel, revision),
            ref_width: source.width(),
            ref_height: source.height(),
            sampling: Some(self),
            sampling_geometry: Some(Geometry {
                width: source.width(),
                height: source.height(),
                axes: (0..4)
                    .map(|level| {
                        [
                            self.axis(source.width(), level),
                            self.axis(source.height(), level),
                        ]
                    })
                    .collect(),
            }),
        }
    }

    /// Compose nonnegative, normalized squared-tap ownership back to original
    /// coordinates. This is a removal-mass model, not the resizer's derivative.
    fn axis(self, logical: usize, level: usize) -> Vec<Vec<(usize, f64)>> {
        let extended = logical.max(self.min_dim());
        let mut map: Vec<Vec<(usize, f64)>> = (0..extended)
            .map(|i| vec![(crate::metric::reflect_index(i, logical), 1.0)])
            .collect();
        let dims = self.dims(logical, logical);
        for (i, &(len, _)) in dims.iter().enumerate().take(level + 1) {
            if (i == 0 && (self.keep_y || self.is_direct())) || (self.is_direct() && i != level) {
                continue;
            }
            let table = F32WeightTable::new(
                map.len() as u32,
                len as u32,
                &InterpolationDetails::create(self.filter()),
            );
            let next = (0..len)
                .map(|o| {
                    let weights = table.weights(o);
                    let norm: f64 = weights.iter().map(|&w| f64::from(w).powi(2)).sum();
                    let mut sums = std::collections::BTreeMap::<usize, f64>::new();
                    for (j, &weight) in weights.iter().enumerate() {
                        if weight == 0.0 {
                            continue;
                        }
                        let a = f64::from(weight).powi(2) / norm;
                        for &(x, mass) in &map[table.left[o] as usize + j] {
                            *sums.entry(x).or_default() += a * mass;
                        }
                    }
                    sums.into_iter().collect()
                })
                .collect();
            map = next;
        }
        map
    }
}

type Axis = Vec<Vec<(usize, f64)>>;
/// Cache spatial ownership once per reference; it depends only on the model's
/// sampling contract and the logical dimensions, never on image values.
pub(crate) struct Geometry {
    width: usize,
    height: usize,
    axes: Vec<[Axis; 2]>,
}
impl Geometry {
    pub(crate) fn logical_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    pub(crate) fn footprints(&self, axis: usize, level: usize) -> Vec<(usize, usize)> {
        self.axes[level][axis]
            .iter()
            .map(|a| (a.first().unwrap().0, a.last().unwrap().0))
            .collect()
    }
    pub(crate) fn project(&self, plane: &[f32], level: usize) -> Vec<f32> {
        let (width, height) = (self.width, self.height);
        let xs = &self.axes[level][0];
        let ys = &self.axes[level][1];
        assert_eq!(plane.len(), xs.len() * ys.len());
        // Separable transpose keeps work linear in axis tap counts rather
        // than expanding every 2D footprint into a Cartesian product.
        let mut tmp = vec![0.0f64; width * ys.len()];
        for y in 0..ys.len() {
            for (sx, weights) in xs.iter().enumerate() {
                let v = f64::from(plane[y * xs.len() + sx]);
                for &(x, weight) in weights {
                    tmp[y * width + x] += v * weight;
                }
            }
        }
        let mut out = vec![0.0f64; width * height];
        for (sy, weights) in ys.iter().enumerate() {
            for &(y, weight) in weights {
                for x in 0..width {
                    out[y * width + x] += tmp[sy * width + x] * weight;
                }
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_scales_match_quantized_separable_f64_tap_algebra() {
        for divisors in [[1, 2, 4, 8], [1, 3, 5, 7], [1, 2, 3, 5]] {
            for kernel in 0..3 {
                let sampling = Sampling {
                    keep_y: false,
                    kernel,
                    num: 1,
                    den: 1,
                    divisors: Some(divisors),
                };
                let (w, h) = (97usize, 131usize);
                let pixels: Vec<_> = (0..w * h)
                    .map(|i| [(i % 251) as u8, (i * 7 % 239) as u8, (i * 13 % 233) as u8])
                    .collect();
                let source = crate::RgbSlice::new(&pixels, w, h);
                let levels = sampling.pyramid(&source, false, crate::ssim_form::active_revision());
                let mut max_unquantized_delta = 0f64;
                for (level, (planes, ow, oh)) in levels.iter().enumerate() {
                    assert_eq!((*ow, *oh), (w / divisors[level], h / divisors[level]));
                    let xs = sampling.axis(w, level);
                    let ys = sampling.axis(h, level);
                    for a in xs.iter().chain(&ys) {
                        assert!((a.iter().map(|p| p.1).sum::<f64>() - 1.).abs() < 1e-12);
                    }
                    if level == 0 {
                        continue;
                    }
                    let tx = F32WeightTable::new(
                        w as u32,
                        *ow as u32,
                        &InterpolationDetails::create(sampling.filter()),
                    );
                    let ty = F32WeightTable::new(
                        h as u32,
                        *oh as u32,
                        &InterpolationDetails::create(sampling.filter()),
                    );
                    for (ch, plane) in planes.iter().enumerate() {
                        for y in 0..*oh {
                            for x in 0..*ow {
                                let mut reference = 0f64;
                                let mut unquantized = 0f64;
                                for (dy, &wy) in ty.weights(y).iter().enumerate() {
                                    for (dx, &wx) in tx.weights(x).iter().enumerate() {
                                        if wx == 0.0 || wy == 0.0 {
                                            continue;
                                        }
                                        let index = (ty.left[y] as usize + dy) * w
                                            + tx.left[x] as usize
                                            + dx;
                                        let value = f64::from(levels[0].0[ch][index]);
                                        // Independently round to binary16: 10 fraction bits,
                                        // with constant 2^-24 spacing in the subnormal range.
                                        let exponent = if value == 0.0 {
                                            -24
                                        } else {
                                            (value.abs().log2().floor() as i32 - 10).max(-24)
                                        };
                                        let step = 2f64.powi(exponent);
                                        let quantized = (value / step).round_ties_even() * step;
                                        reference += f64::from(wy) * f64::from(wx) * quantized;
                                        unquantized += f64::from(wy) * f64::from(wx) * value;
                                    }
                                }
                                max_unquantized_delta = max_unquantized_delta
                                    .max((unquantized - f64::from(plane[y * ow + x])).abs());
                                assert!(
                                    (reference - f64::from(plane[y * ow + x])).abs() < 2e-6,
                                    "{sampling:?} level={level} ({x},{y}) actual={} reference={reference} unquantized={unquantized}",
                                    plane[y * ow + x]
                                );
                            }
                        }
                    }
                }
                assert!(
                    max_unquantized_delta > 1e-6,
                    "unquantized negative control must fail"
                );
                eprintln!("{sampling:?}: max error vs unquantized f64={max_unquantized_delta:e}");
            }
        }
    }

    #[cfg(feature = "threads")]
    #[test]
    fn parallel_pyramids_match_serial_bits() {
        // Serial vs parallel is a two-computation comparison: a forced-tier permutation test in this
        // binary can switch SIMD tokens between the two calls and make them run different tiers.
        let _tokens = archmage::testing::lock_token_testing();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let (w, h) = (257, 259);
        let pixels: Vec<_> = (0..w * h)
            .map(|i| {
                [
                    (i % 251) as u8,
                    ((i * 7) % 239) as u8,
                    ((i * 13) % 233) as u8,
                ]
            })
            .collect();
        let source = crate::RgbSlice::new(&pixels, w, h);
        for divisors in [[1, 2, 4, 8], [1, 3, 5, 7], [1, 2, 3, 5]] {
            for kernel in 0..3 {
                let sampling = Sampling {
                    keep_y: false,
                    kernel,
                    num: 1,
                    den: 1,
                    divisors: Some(divisors),
                };
                assert_eq!(
                    sampling.pyramid(&source, false, crate::ssim_form::active_revision()),
                    pool.install(|| sampling.pyramid(
                        &source,
                        true,
                        crate::ssim_form::active_revision()
                    )),
                    "{sampling:?}"
                );
            }
        }
        for keep_y in [true, false] {
            for kernel in 0..3 {
                for (num, den) in [(3, 2), (2, 1), (3, 1)] {
                    let sampling = Sampling {
                        keep_y,
                        kernel,
                        num,
                        den,
                        divisors: None,
                    };
                    let serial =
                        sampling.pyramid(&source, false, crate::ssim_form::active_revision());
                    let parallel = pool.install(|| {
                        sampling.pyramid(&source, true, crate::ssim_form::active_revision())
                    });
                    assert_eq!(serial, parallel, "{sampling:?}");
                }
            }
        }
    }

    #[test]
    fn spatial_ownership_conserves_signed_mass_and_contains_all_taps() {
        for keep_y in [false, true] {
            for kernel in 0..3 {
                for (num, den) in [(3, 2), (2, 1), (3, 1)] {
                    let sampling = Sampling {
                        keep_y,
                        kernel,
                        num,
                        den,
                        divisors: None,
                    };
                    for (width, height) in [(7, 11), (97, 131)] {
                        let geometry = Geometry {
                            width,
                            height,
                            axes: (0..4)
                                .map(|level| {
                                    [sampling.axis(width, level), sampling.axis(height, level)]
                                })
                                .collect(),
                        };
                        for level in 0..4 {
                            for axis in 0..2 {
                                let bounds = geometry.footprints(axis, level);
                                for (i, weights) in geometry.axes[level][axis].iter().enumerate() {
                                    assert!(
                                        (weights.iter().map(|p| p.1).sum::<f64>() - 1.0).abs()
                                            < 1e-12
                                    );
                                    assert!(weights.iter().all(|&(x, weight)| weight > 0.0
                                        && x >= bounds[i].0
                                        && x <= bounds[i].1));
                                }
                            }
                            let n = geometry.axes[level][0].len() * geometry.axes[level][1].len();
                            let plane: Vec<f32> = (0..n).map(|i| (i % 17) as f32 - 6.0).collect();
                            let sum = plane.iter().map(|&v| f64::from(v)).sum::<f64>();
                            let projected = geometry.project(&plane, level);
                            let total = projected.iter().map(|&v| f64::from(v)).sum::<f64>();
                            assert!(
                                (sum - total).abs() < n as f64 * 2e-6,
                                "{sampling:?} {width}x{height} level {level}: {sum} != {total}"
                            );
                        }
                    }
                }
            }
        }
    }
}
