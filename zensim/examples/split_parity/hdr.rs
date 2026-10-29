// Deterministic synthetic HDR parity fixtures, shared verbatim between
// zensimx's `split_parity_golden` and the slim crate's `split_parity_check`
// (and the slim crate's in-tree parity dump). Seeded noise, gradients up to
// 4000 cd/m² and near-black ramps — no datasets.

/// Encoding index: 0 = Linear (f32 nits), 1 = Pq{1000}, 2 = Pq{10000},
/// 3 = Hlg{1000, 5 lux}.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct HdrCase {
    pub enc: u8,
    pub enc_name: &'static str,
    /// `true` → `Srgb16Rgba` (u16 code values); `false` → `LinearF32Rgba`.
    pub srgb16: bool,
    pub width: usize,
    pub height: usize,
    /// `true` → `ColorPrimaries::Bt2020`; `false` → `Srgb` (BT.709).
    pub bt2020: bool,
    /// Extra row-stride bytes past width*bpp (odd strides).
    pub pad: usize,
    pub parallel: bool,
    /// 0 = seeded noise, 1 = gradient to the peak, 2 = near-black ramp.
    pub kind: u8,
    pub seed: u64,
}

/// The folded walk's identity emit width at revision 7.
#[allow(dead_code)]
pub const HDR_FEATURE_WIDTH: usize = 720;

/// The case matrix: 4 encodings × 5 sizes × 2 primaries × 2 strides ×
/// 2 thread modes, plus a small `LinearF32Rgba`-carried PQ/HLG set at
/// 64×64 covering the f32 code-value decode path.
#[allow(dead_code)]
pub fn hdr_cases() -> Vec<HdrCase> {
    let sizes = [(8, 8), (33, 17), (64, 64), (257, 193), (512, 384)];
    let encs: [(u8, &'static str, bool); 4] = [
        (0, "lin", false),
        (1, "pq1k", true),
        (2, "pq10k", true),
        (3, "hlg", true),
    ];
    let mut cases = Vec::new();
    let mut n = 0usize;
    for &(enc, enc_name, srgb16) in &encs {
        for &(w, h) in &sizes {
            for bt2020 in [false, true] {
                for pad in [0usize, 7] {
                    for parallel in [false, true] {
                        cases.push(HdrCase {
                            enc,
                            enc_name,
                            srgb16,
                            width: w,
                            height: h,
                            bt2020,
                            pad,
                            parallel,
                            kind: (n % 3) as u8,
                            seed: 0x5eed_u64.wrapping_mul((n as u64) + 1),
                        });
                        n += 1;
                    }
                }
            }
        }
    }
    // f32 code-value containers for PQ/HLG at 64×64 (BT.2020, odd stride).
    for &(enc, enc_name, _) in &encs[1..] {
        for parallel in [false, true] {
            cases.push(HdrCase {
                enc,
                enc_name,
                srgb16: false,
                width: 64,
                height: 64,
                bt2020: true,
                pad: 3,
                parallel,
                kind: (n % 3) as u8,
                seed: 0x5eed_u64.wrapping_mul((n as u64) + 1),
            });
            n += 1;
        }
    }
    cases
}

/// Stable row-key for one case (goes in the TSV `pair` column before the
/// `:f<slot>` suffix).
#[allow(dead_code)]
pub fn hdr_case_id(c: &HdrCase) -> String {
    format!(
        "{}x{}-{}-pad{}-{}{}-k{}",
        c.width,
        c.height,
        if c.bt2020 { "bt2020" } else { "bt709" },
        c.pad,
        if c.parallel { "par" } else { "ser" },
        if c.srgb16 { "-u16" } else { "-f32" },
        c.kind,
    )
}

fn next_f32(state: &mut u64) -> f32 {
    // splitmix64, 24-bit mantissa quantised for a dense reproducible range.
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 40) & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Pixel bytes for one case — `dst` produces the distorted member of the
/// pair (a decorrelated variant of the same generator state).
#[allow(dead_code)]
pub fn gen_hdr_image(c: &HdrCase, dst: bool) -> (Vec<u8>, usize) {
    let bpp = if c.srgb16 { 8usize } else { 16usize };
    let stride = c.width * bpp + c.pad;
    let mut buf = vec![0u8; c.height * stride];
    let mut st = c.seed ^ if dst { 0xdead_beef_cafe_f00d } else { 0 };
    // Value ceilings: Linear emits absolute nits (gradient tops at
    // 4000 cd/m²); PQ/HLG code values live in [0, 1].
    let (noise_lo, noise_hi, grad_peak, dark_hi) = if c.enc == 0 {
        (0.005f32, 4000.0f32, 4000.0f32, 0.05f32)
    } else {
        (0.0f32, 1.0f32, 1.0f32, 0.01f32)
    };
    for y in 0..c.height {
        for x in 0..c.width {
            let px = match c.kind {
                // Seeded noise.
                0 => [
                    noise_lo + next_f32(&mut st) * (noise_hi - noise_lo),
                    noise_lo + next_f32(&mut st) * (noise_hi - noise_lo),
                    noise_lo + next_f32(&mut st) * (noise_hi - noise_lo),
                ],
                // Horizontal gradient up to the peak, per-channel phase.
                1 => {
                    let t = if c.width > 1 {
                        x as f32 / (c.width - 1) as f32
                    } else {
                        0.0
                    };
                    [
                        t * grad_peak,
                        (1.0 - t) * grad_peak,
                        (t * 0.7 + y as f32 / (c.height.max(1) as f32) * 0.3) * grad_peak,
                    ]
                }
                // Near-black ramp: faint left→right rise.
                _ => {
                    let t = x as f32 / c.width.max(1) as f32;
                    [t * dark_hi, t * dark_hi * 0.5, dark_hi * 0.25]
                }
            };
            let off = y * stride + x * bpp;
            if c.srgb16 {
                for ch in 0..3 {
                    let v = (px[ch].clamp(0.0, 1.0) * 65535.0).round() as u16;
                    buf[off + ch * 2..off + ch * 2 + 2].copy_from_slice(&v.to_ne_bytes());
                }
                buf[off + 6..off + 8].copy_from_slice(&65535u16.to_ne_bytes());
            } else {
                for ch in 0..3 {
                    buf[off + ch * 4..off + ch * 4 + 4]
                        .copy_from_slice(&px[ch].to_bits().to_ne_bytes());
                }
                buf[off + 12..off + 16].copy_from_slice(&1.0f32.to_bits().to_ne_bytes());
            }
        }
    }
    (buf, stride)
}
