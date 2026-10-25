//! Shared deterministic HDR inputs for the integration gates and bit-dump harness.

/// Absolute-nits fixture from `native_hdr_color::hdr_public_features_match_independent_common_primaries_without_sdr_clipping`.
pub fn native_color(w: usize, h: usize) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let a: Vec<[f32; 4]> = (0..w * h)
        .map(|i| {
            let t = (i * 37 % 251) as f32;
            [
                50. + t * 2.,
                20. + (i * 19 % 191) as f32,
                10. + (i * 13 % 173) as f32,
                1.,
            ]
        })
        .collect();
    let b: Vec<_> = a
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0] * 0.91, p[1] + (i % 7) as f32, p[2] * 1.08, 1.])
        .collect();
    (a, b)
}

/// A 0.1–1,800 cd/m² ramp with texture; every seventh sample brightens by 8%.
pub fn chdr_pu(w: usize, h: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let r: Vec<f32> = (0..w * h * 3)
        .map(|i| {
            let x = (i / 3) % w;
            let base = 0.1 + 1200.0 * (x as f32 / w as f32).powi(2);
            base * (0.5 + rnd())
        })
        .collect();
    let d = r
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 7 == 0 { v * 1.08 } else { *v })
        .collect();
    (r, d)
}
