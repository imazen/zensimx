use archmage::ScalarToken;
use archmage_attune::{attune, attuned};
use magetypes::simd::{backends::F32x16Convert, generic::f32x16 as V};

use super::{COMMON, Features, Fx2Extraction, IntensityTarget, RgbRows, Variant, stream};
use crate::color::{
    K_B0, K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22, srgb_u8_to_linear,
};
use stream::{LANES, MOMENTS, Moments, PLANES, PixelStencil, RADIUS};

const C2: f32 = 0.0009;
const EPS: f32 = 1e-12;
pub(super) const EXP_CUTOFF: f32 = -32.0;
pub(super) const ALPHA: f32 = 0.5;
pub(super) const BETA: f32 = 0.35;
const MEAN: f32 = 1.0 / 121.0;
const CBRT_BIAS: f32 = 0.155_954_2;
const OFFSETS: [f32; 3] = [0.42, 0.45, 0.55];
const MUL_500: f64 = 500.0 / 255.0;
const SCALE_500: f32 = 0.770_926_36;

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn scan(
    token: Token,
    a: RgbRows<'_>,
    b: RgbRows<'_>,
    variant: Variant,
    parallel: bool,
) -> [[Features; 3]; 5] {
    stream::extract(RowMath(token), a, b, variant, parallel)
}

pub(super) fn extract(
    a: RgbRows<'_>,
    b: RgbRows<'_>,
    variant: Variant,
    parallel: bool,
) -> [[Features; 3]; 5] {
    attuned!(
        scan(a, b, variant, parallel),
        [
            _v4x(cfg(avx512)),
            _v4(cfg(avx512)),
            _v3,
            _neon,
            _wasm128,
            _scalar
        ]
    )
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn scan_fx2(
    token: Token,
    a: RgbRows<'_>,
    b: RgbRows<'_>,
    variant: Variant,
    target: IntensityTarget,
    parallel: bool,
) -> Fx2Extraction {
    stream::extract_fx2(RowMath(token), a, b, variant, target, parallel)
}

pub(super) fn extract_fx2(
    a: RgbRows<'_>,
    b: RgbRows<'_>,
    variant: Variant,
    target: IntensityTarget,
    parallel: bool,
) -> Fx2Extraction {
    attuned!(
        scan_fx2(a, b, variant, target, parallel),
        [
            _v4x(cfg(avx512)),
            _v4(cfg(avx512)),
            _v3,
            _neon,
            _wasm128,
            _scalar
        ]
    )
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn tier_name(token: Token) -> &'static str {
    core::any::type_name_of_val(&token)
}

pub(super) fn tier() -> &'static str {
    attuned!(
        tier_name(),
        [
            _v4x(cfg(avx512)),
            _v4(cfg(avx512)),
            _v3,
            _neon,
            _wasm128,
            _scalar
        ]
    )
    .rsplit("::")
    .next()
    .unwrap()
}

#[derive(Clone, Copy)]
pub(super) struct RowMath<T>(pub(super) T);

impl<T: F32x16Convert> RowMath<T> {
    #[inline(always)]
    fn splat(self, x: f32) -> V<T> {
        V::splat_t(self.0, x)
    }

    #[inline(always)]
    fn load(self, x: &[f32], start: usize) -> V<T> {
        V::load_t(self.0, x[start..start + LANES].try_into().unwrap())
    }

    #[inline(always)]
    fn store(self, v: V<T>, x: &mut [f32], start: usize) {
        v.store((&mut x[start..start + LANES]).try_into().unwrap());
    }

    #[inline(always)]
    fn fma(self, a: V<T>, b: V<T>, c: V<T>) -> V<T> {
        if cfg!(target_arch = "wasm32")
            || core::any::TypeId::of::<T>() == core::any::TypeId::of::<ScalarToken>()
        {
            let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
            V::from_array_t(self.0, core::array::from_fn(|i| a[i].mul_add(b[i], c[i])))
        } else {
            a.mul_add_relaxed(b, c)
        }
    }

    /// Use the hardware estimate where available. Scalar/WASM use the portable
    /// seed and an extra step to reach the hardware seed's precision floor.
    #[inline(always)]
    pub(super) fn rcp_nr1(self, input: V<T>, floor: f32) -> V<T> {
        // Positive normal floor (>= 1e-30) before the estimate protects
        // zero and subnormal lanes. The in-use upper shoulder is < 2^126.
        let normal = input.max(self.splat(floor));
        let finite = normal.simd_lt(self.splat(f32::INFINITY));
        let x = V::blend(finite, normal, self.splat(1.0));
        let native = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
            && core::any::TypeId::of::<T>() != core::any::TypeId::of::<archmage::ScalarToken>();
        let one = self.splat(1.0);
        let refine = |r: V<T>| self.fma(r, self.fma(-x, r, one), r);
        let r = if native {
            x.rcp_approx()
        } else {
            refine(x.rcp_approx_portable())
        };
        // An infinity must map to zero, even when x*r in Newton's step is NaN.
        V::blend(finite, refine(r), self.splat(0.0))
    }

    #[inline(always)]
    pub(super) fn negative_exp(self, x: V<T>) -> V<T> {
        // The discarded tail is bounded by exp(-32), below 1.27e-14.
        // An unordered argument or a positive argument is outside this profile.
        let zero = self.splat(0.0);
        let cutoff = self.splat(EXP_CUTOFF);
        let exp = (x.max(cutoff).min(zero) * self.splat(core::f32::consts::LOG2_E)).exp2_midp();
        let exp = V::blend(x.simd_le(cutoff), zero, exp);
        V::blend(x.simd_le(zero), exp, self.splat(f32::NAN))
    }

    #[inline(always)]
    pub(super) fn cube_root_and_add(self, x: V<T>, add: f32) -> V<T> {
        // Use the Rev7 reciprocal cube-root seed, then finish the direct
        // cube-root refinement with one estimated reciprocal instead of div.
        let a = x.abs().max(self.splat(1e-30));
        let third = self.splat(1.0 / 3.0);
        let one = self.splat(1.0);
        let q = (V::from_i32_t(self.0, a.bitcast_to_i32()) * third).to_i32();
        let magic = self.splat(f32::from_bits(crate::det_math::CBRT_NEWTON_MAGIC as u32));
        let r = V::from_i32_bitcast_t(self.0, magic.bitcast_to_i32() - q);
        let t = a * r;
        let e = self.fma(-t, r * r, one);
        let r = self.fma(r * e, third, r);
        let y = (a * r) * r;
        let y2 = y * y;
        let residual = self.fma(-y2, y, a);
        let result = self.fma(
            y,
            residual * self.rcp_nr1(self.fma(y2 + y2, y, a), 1e-30),
            y,
        );
        let sign = x & self.splat(-0.0);
        let root = V::blend(x.simd_eq(self.splat(0.0)), x, result | sign);
        // Adding +0 would erase the sign of an exact -0.
        if add == 0.0 {
            root
        } else {
            root + self.splat(add)
        }
    }

    #[inline(always)]
    pub(super) fn color_math<const NITS500: bool>(self, rgb: &[[u8; 3]], out: &mut [Vec<f32>]) {
        let scale = if NITS500 {
            self.splat(SCALE_500)
        } else {
            self.splat(1.0)
        };
        let mul = |w: f32| {
            if NITS500 {
                (f64::from(w) * MUL_500) as f32
            } else {
                w
            }
        };
        for chunk in (0..rgb.len()).step_by(LANES) {
            let n = (rgb.len() - chunk).min(LANES);
            let to_linear = |c: usize| {
                V::from_array_t(
                    self.0,
                    core::array::from_fn(|i| {
                        if i < n {
                            srgb_u8_to_linear(rgb[chunk + i][c])
                        } else {
                            0.0
                        }
                    }),
                )
            };
            let r = to_linear(0);
            let g = to_linear(1);
            let b = to_linear(2);
            let mix = |m0, m1, m2| {
                self.fma(
                    self.splat(mul(m0)),
                    r,
                    self.fma(
                        self.splat(mul(m1)),
                        g,
                        self.fma(self.splat(mul(m2)), b, self.splat(K_B0)),
                    ),
                )
            };
            let mut c0 = self.cube_root_and_add(mix(K_M00, K_M01, K_M02), -CBRT_BIAS);
            let mut c1 = self.cube_root_and_add(mix(K_M10, K_M11, K_M12), -CBRT_BIAS);
            let mut c2 = self.cube_root_and_add(mix(K_M20, K_M21, K_M22), -CBRT_BIAS);
            if NITS500 {
                c0 *= scale;
                c1 *= scale;
                c2 *= scale;
            }
            let black = (r + g + b).simd_eq(self.splat(0.0));
            let y = (c0 + c1) * self.splat(0.5);
            self.store(
                self.fma(c0 - c1, self.splat(7.0), self.splat(0.42)),
                &mut out[0],
                chunk + RADIUS,
            );
            self.store(
                V::blend(black, self.splat(0.01), y + self.splat(0.01)),
                &mut out[1],
                chunk + RADIUS,
            );
            self.store(
                V::blend(black, self.splat(0.55), c2 - y + self.splat(0.55)),
                &mut out[2],
                chunk + RADIUS,
            );
        }
    }

    #[inline(always)]
    pub(super) fn horizontal_math(
        self,
        a: &[f32],
        b: &[f32],
        center: f32,
        scratch: &mut Moments,
        out: &mut Moments,
        padded: usize,
    ) {
        let center = self.splat(center);
        for x in (0..a.len()).step_by(LANES) {
            let a = self.load(a, x);
            let b = self.load(b, x);
            // Difference is formed before either centering operation.
            let e = a - b;
            let a = a - center;
            let b = b - center;
            for (plane, value) in scratch.iter_mut().zip([a, b, e, a * a, b * b, e * e]) {
                self.store(value, plane, x);
            }
        }
        for (ph, raw) in out.iter_mut().zip(scratch) {
            for x in (0..padded).step_by(LANES) {
                let load = |k| self.load(raw, x + k);
                // A fixed sliding tap tree prevents horizontal drift and
                // evaluates the same taps for all strips and runtime tiers.
                let lo = ((load(0) + load(1)) + (load(2) + load(3)))
                    + ((load(4) + load(5)) + (load(6) + load(7)));
                let hi = (load(8) + load(9)) + load(10);
                self.store(lo + hi, ph, x);
            }
        }
    }

    #[inline(always)]
    pub(super) fn vertical_math(self, taps: &[&Moments; 11], out: &mut Moments, padded: usize) {
        for (m, plane) in out.iter_mut().enumerate() {
            for x in (0..padded).step_by(LANES) {
                let load = |k: usize| self.load(&taps[k][m], x);
                // Fresh global-coordinate block on every output row.
                // A small entering tap survives once the large taps leave.
                let value = (((load(0) + load(1)) + (load(2) + load(3)))
                    + ((load(4) + load(5)) + (load(6) + load(7))))
                    + ((load(8) + load(9)) + load(10));
                self.store(value, plane, x);
            }
        }
    }

    #[inline(always)]
    pub(super) fn moment_variance(self, sum: V<T>, sum_sq: V<T>) -> V<T> {
        let mean = sum * self.splat(MEAN);
        (sum_sq * self.splat(MEAN) - mean * mean).max(self.splat(0.0))
    }

    #[inline(always)]
    fn visibility_power<const P: usize>(self, sigma: V<T>, threshold: f32, amplitude: f32) -> V<T> {
        let one = self.splat(1.0);
        // Select a safe argument before the reciprocal and powers.
        let full = sigma.simd_le(self.splat(threshold));
        let inv = self.rcp_nr1(V::blend(full, one, sigma), threshold);
        let inv2 = inv * inv;
        let inv_pow = match P {
            1 => inv,
            2 => inv2,
            _ => inv2 * inv2,
        };
        V::blend(full, one, (self.splat(amplitude) * inv_pow).min(one))
    }

    #[inline(always)]
    pub(super) fn visibility(self, sigma: V<T>) -> [V<T>; 3] {
        [
            self.visibility_power::<1>(sigma, 0.03, 0.03),
            self.visibility_power::<2>(sigma, 0.05, 0.0025),
            self.visibility_power::<4>(sigma, 0.10, 0.0001),
        ]
    }

    #[inline(always)]
    fn spatial_terms<const HYBRID: bool>(
        self,
        pixels: &PixelStencil<'_>,
        side: usize,
        x: usize,
        block_x: V<T>,
        block_y: bool,
    ) -> [V<T>; 6] {
        let zero = self.splat(0.0);
        let i = x + RADIUS;
        let center = self.load(pixels.center[side], i);
        let dx = self.load(pixels.center[side], i + 1) - center;
        let dy = self.load(pixels.after[side], i) - center;
        let back_x = center - self.load(pixels.center[side], i - 1);
        let back_y = center - self.load(pixels.before[side], i);
        let g = (dx * dx + dy * dy).sqrt();
        let curvature = (dx - back_x).abs() + (dy - back_y).abs();
        let hx = dx.abs();
        let hy = dy.abs();
        let px = back_x.abs();
        let py = back_y.abs();
        let boundary_y = if block_y { (hy - py).max(zero) } else { zero };
        let block = block_x * (hx - px).max(zero) + boundary_y;
        let ring = ((hx + px - (dx + back_x).abs()).max(zero)
            + (hy + py - (dy + back_y).abs()).max(zero))
            * self.splat(0.5);
        let interior = if HYBRID {
            hx * (self.splat(1.0) - block_x) + if block_y { zero } else { hy }
        } else {
            zero
        };
        [center, g, curvature, block, ring, interior]
    }

    #[inline(always)]
    fn directed_terms(
        self,
        source: V<T>,
        dest: V<T>,
        high_src: V<T>,
        high_dst: V<T>,
        high_norm: V<T>,
    ) -> [V<T>; 4] {
        let zero = self.splat(0.0);
        let loss = (source - dest).max(zero);
        let gain = (dest - source).max(zero);
        let texture = self.splat(1.0) - self.negative_exp(source * self.splat(-1.0 / 0.08));
        [
            loss * self.rcp_nr1(source + self.splat(EPS), EPS) * texture,
            gain * self.negative_exp(source * self.splat(-1.0 / 0.045)),
            (high_src - high_dst).max(zero) * high_norm,
            (high_dst - high_src).max(zero) * high_norm,
        ]
    }

    #[inline(always)]
    pub(super) fn terms_math<const HYBRID: bool, const SYM_GRAD: bool>(
        self,
        moments: &Moments,
        pixels: PixelStencil<'_>,
        channel: usize,
        rows: &mut stream::RowTotals,
    ) {
        let zero = self.splat(0.0);
        let one = self.splat(1.0);
        let c2 = self.splat(C2);
        let eps = self.splat(EPS);
        let mut totals = [zero; COMMON];
        let mut orientation = [zero; 2];
        let mut sym_grad = zero;
        let mut powers = [[0.0_f64; LANES]; 7];
        #[cfg(test)]
        let mut orientation_parts = [[zero; 3]; 2];
        #[cfg(test)]
        let mut mean_fragility = zero;
        let mut forward = [zero; 4];
        let mut reverse = [zero; 4];
        let width = pixels.width;
        let offset = self.splat(OFFSETS[channel]);
        let phase_y = pixels.y % 8 == 7 && pixels.y + 1 < pixels.height;
        for x in (0..width).step_by(LANES) {
            let m: [V<T>; MOMENTS] = core::array::from_fn(|i| self.load(&moments[i], x));
            let mean_a = m[0] * self.splat(MEAN) + offset;
            let mean_b = m[1] * self.splat(MEAN) + offset;
            let mean_e = m[2] * self.splat(MEAN);
            let va = self.moment_variance(m[0], m[3]);
            let vb = self.moment_variance(m[1], m[4]);
            let ve = self.moment_variance(m[2], m[5]);
            let l = (mean_e * mean_e * self.rcp_nr1(mean_a * mean_a + mean_b * mean_b + c2, C2))
                .min(one);
            let structure = (ve * self.rcp_nr1(va + vb + c2, C2)).min(self.splat(2.0));
            let damage = (l + (one - l) * structure).max(zero);

            let block_x = V::from_array_t(
                self.0,
                core::array::from_fn(|i| f32::from((x + i) % 8 == 7 && x + i + 1 < width)),
            );
            let [a, ga, ca, ba, ra, ia] =
                self.spatial_terms::<HYBRID>(&pixels, 0, x, block_x, phase_y);
            let [b, gb, cb, bb, rb, ib] =
                self.spatial_terms::<HYBRID>(&pixels, 1, x, block_x, phase_y);
            let e = a - b;
            let g_delta = ga - gb;
            let gms = g_delta * g_delta * self.rcp_nr1(ga * ga + gb * gb + c2, C2);
            let edge =
                (ca * self.rcp_nr1(ca + ga + c2, C2) - cb * self.rcp_nr1(cb + gb + c2, C2)).abs();
            let low = va.min(vb).sqrt();
            let high = va.max(vb).sqrt();
            let delta = high - low;
            let [md, me, mf] = self.visibility(low);
            let smooth = self.negative_exp(low * self.splat(-1.0 / 0.045));
            let textured = one - self.negative_exp(low * self.splat(-1.0 / 0.08));
            let art = delta * smooth;
            let det = delta * self.rcp_nr1(high + eps, EPS) * textured;
            let frag_a = ra * self.rcp_nr1(ga + ra + c2, C2);
            let frag_b = rb * self.rcp_nr1(gb + rb + c2, C2);
            let fragility = frag_a.max(frag_b) * damage;
            let live = if width - x >= LANES {
                one
            } else {
                V::from_array_t(self.0, core::array::from_fn(|i| f32::from(x + i < width)))
            };
            let terms = [
                (0, l),
                (3, structure),
                (7, e.abs()),
                (9, gms),
                (10, edge),
                (11, (ba - bb).abs()),
                (12, (ra - rb).abs()),
                (13, damage * md),
                (14, damage * me),
                (15, damage * mf),
                (16, art),
                (17, det),
                (18, fragility),
            ];
            for (idx, value) in terms {
                totals[idx] += value * live;
            }
            if SYM_GRAD {
                sym_grad += (ga + gb) * self.splat(0.5) * live;
            }
            let l = (l * live).to_array();
            let s = (structure.max(zero) * live).to_array();
            let d = (damage * live).to_array();
            let e = (e * live).to_array();
            for i in 0..LANES {
                let l = f64::from(l[i]);
                let s = f64::from(s[i]);
                let d = f64::from(d[i]);
                let e = f64::from(e[i]);
                let l2 = l * l;
                let d2 = d * d;
                let d4 = d2 * d2;
                for (acc, term) in
                    powers
                        .iter_mut()
                        .zip([l2, l2 * l2, s * s, d2, d4, e * e, d4 * d4])
                {
                    acc[i] += term;
                }
            }
            #[cfg(test)]
            {
                mean_fragility += (frag_a + frag_b) * self.splat(0.5) * damage * live;
            }

            if HYBRID {
                let sa = va.sqrt();
                let sb = vb.sqrt();
                let high_norm = self.rcp_nr1(ca + cb + c2, C2);
                for (total, term) in forward
                    .iter_mut()
                    .zip(self.directed_terms(sa, sb, ca, cb, high_norm))
                {
                    *total += term * live;
                }
                for (total, term) in reverse
                    .iter_mut()
                    .zip(self.directed_terms(sb, sa, cb, ca, high_norm))
                {
                    *total += term * live;
                }
                let spatial = [(ia, ba, ra), (ib, bb, rb)];
                for (side, (total, (interior, bk, rg))) in
                    orientation.iter_mut().zip(spatial).enumerate()
                {
                    let activity = interior * self.rcp_nr1(interior + self.splat(0.05), 0.05);
                    *total += (activity - self.splat(ALPHA) * bk - self.splat(BETA) * rg) * live;
                    #[cfg(test)]
                    for (part, value) in orientation_parts[side].iter_mut().zip([activity, bk, rg])
                    {
                        *part += value * live;
                    }
                    #[cfg(not(test))]
                    let _ = side;
                }
            }
        }
        // Each term crosses from f32 to f64 once per row and channel.
        for (total, lanes) in rows.features[..COMMON].iter_mut().zip(totals) {
            *total += f64::from(lanes.reduce_add());
        }
        for (i, lanes) in [1, 2, 4, 5, 6, 8, 19].into_iter().zip(powers) {
            rows.features[i] += lanes.into_iter().sum::<f64>();
        }
        if SYM_GRAD {
            rows.sym_grad += f64::from(sym_grad.reduce_add());
        }
        #[cfg(test)]
        {
            rows.mean_fragility += f64::from(mean_fragility.reduce_add());
            for (parts, vectors) in rows.orientation_parts.iter_mut().zip(orientation_parts) {
                for (total, lanes) in parts.iter_mut().zip(vectors) {
                    *total += f64::from(lanes.reduce_add());
                }
            }
        }
        if HYBRID {
            for (total, lanes) in rows.orientation.iter_mut().zip(orientation) {
                *total += f64::from(lanes.reduce_add());
            }
            for (total, lanes) in rows.forward.iter_mut().zip(forward) {
                *total += f64::from(lanes.reduce_add());
            }
            for (total, lanes) in rows.reverse.iter_mut().zip(reverse) {
                *total += f64::from(lanes.reduce_add());
            }
        }
    }
}

// Re-enter each dense row at the feature boundary. Strip orchestration
// (including Rayon workers) does not inherit target features from scan.
pub(super) trait RowKernels: Copy + Send + Sync {
    fn color<const NITS500: bool>(self, rgb: &[[u8; 3]], out: &mut [Vec<f32>]);
    fn horizontal(
        self,
        a: &[f32],
        b: &[f32],
        center: f32,
        scratch: &mut Moments,
        out: &mut Moments,
        padded: usize,
    );
    fn vertical(self, taps: &[&Moments; 11], out: &mut Moments, padded: usize);
    fn terms<const HYBRID: bool, const SYM_GRAD: bool>(
        self,
        moments: &Moments,
        pixels: PixelStencil<'_>,
        channel: usize,
        totals: &mut stream::RowTotals,
    );
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn color_row<const NITS500: bool>(token: Token, rgb: &[[u8; 3]], out: &mut [Vec<f32>]) {
    RowMath(token).color_math::<NITS500>(rgb, out);
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn horizontal_row(
    token: Token,
    a: &[f32],
    b: &[f32],
    center: f32,
    scratch: &mut Moments,
    out: &mut Moments,
    padded: usize,
) {
    RowMath(token).horizontal_math(a, b, center, scratch, out, padded);
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn vertical_row(token: Token, taps: &[&Moments; 11], out: &mut Moments, padded: usize) {
    RowMath(token).vertical_math(taps, out, padded);
}

#[attune(_*_t, +v4(cfg(avx512)), +v4x(cfg(avx512)))]
fn terms_row<const HYBRID: bool, const SYM_GRAD: bool>(
    token: Token,
    moments: &Moments,
    pixels: PixelStencil<'_>,
    channel: usize,
    totals: &mut stream::RowTotals,
) {
    RowMath(token).terms_math::<HYBRID, SYM_GRAD>(moments, pixels, channel, totals);
}

macro_rules! row_kernels {
    ($token:ty, $color:ident, $horizontal:ident, $vertical:ident, $terms:ident) => {
        impl RowKernels for RowMath<$token> {
            fn color<const NITS500: bool>(self, rgb: &[[u8; 3]], out: &mut [Vec<f32>]) {
                $color::<NITS500>(self.0, rgb, out);
            }

            fn horizontal(
                self,
                a: &[f32],
                b: &[f32],
                center: f32,
                scratch: &mut Moments,
                out: &mut Moments,
                padded: usize,
            ) {
                $horizontal(self.0, a, b, center, scratch, out, padded);
            }

            fn vertical(self, taps: &[&Moments; 11], out: &mut Moments, padded: usize) {
                $vertical(self.0, taps, out, padded);
            }

            fn terms<const HYBRID: bool, const SYM_GRAD: bool>(
                self,
                moments: &Moments,
                pixels: PixelStencil<'_>,
                channel: usize,
                totals: &mut stream::RowTotals,
            ) {
                $terms::<HYBRID, SYM_GRAD>(self.0, moments, pixels, channel, totals);
            }
        }
    };
}

row_kernels!(
    ScalarToken,
    color_row_scalar_t,
    horizontal_row_scalar_t,
    vertical_row_scalar_t,
    terms_row_scalar_t
);
#[cfg(target_arch = "x86_64")]
row_kernels!(
    archmage::X64V3Token,
    color_row_v3_t,
    horizontal_row_v3_t,
    vertical_row_v3_t,
    terms_row_v3_t
);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
row_kernels!(
    archmage::X64V4Token,
    color_row_v4_t,
    horizontal_row_v4_t,
    vertical_row_v4_t,
    terms_row_v4_t
);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
row_kernels!(
    archmage::X64V4xToken,
    color_row_v4x_t,
    horizontal_row_v4x_t,
    vertical_row_v4x_t,
    terms_row_v4x_t
);
#[cfg(target_arch = "aarch64")]
row_kernels!(
    archmage::NeonToken,
    color_row_neon_t,
    horizontal_row_neon_t,
    vertical_row_neon_t,
    terms_row_neon_t
);
#[cfg(target_arch = "wasm32")]
row_kernels!(
    archmage::Wasm128Token,
    color_row_wasm128_t,
    horizontal_row_wasm128_t,
    vertical_row_wasm128_t,
    terms_row_wasm128_t
);

pub(super) fn channel_center(channel: usize) -> f32 {
    OFFSETS[channel]
}

// Keep the fixed plane layout visible at the SIMD ingress boundary.
const _: () = assert!(PLANES == 6);
