// Extra surfaces for the repository-split parity gate: every public input path, not only
// 8-bit RGB through the five profiles. Byte-identical in imazen/zensimx
// (`split_parity_golden`) and imazen/zensim (`split_parity_check`); `include!`d after
// `pairs.rs` by both.
//
// Each line is `surface \t profile \t pair \t value`, where value is the f64/f32 bits of the
// result or `ERR <Debug>`. Pairs smaller than 8x8 are skipped by surfaces that need a
// precomputed reference (they refuse such inputs).

fn surf_fnv64(words: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn surf_hex(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn surf_result(r: Result<zensim::ZensimResult, zensim::ZensimError>) -> String {
    match r {
        Ok(r) => format!(
            "{} {} {:016x}",
            surf_hex(r.score()),
            surf_hex(r.raw_distance()),
            surf_fnv64(r.features().iter().map(|x| x.to_bits()))
        ),
        Err(e) => format!("ERR {e:?}"),
    }
}

fn surf_diffmap(r: Result<zensim::DiffmapResult, zensim::ZensimError>) -> String {
    match r {
        Ok(d) => format!(
            "{} {:016x}",
            surf_hex(d.score()),
            surf_fnv64(d.diffmap().iter().map(|x| x.to_bits() as u64))
        ),
        Err(e) => format!("ERR {e:?}"),
    }
}

/// Deterministic alpha in 128..=255 that varies with position.
fn surf_alpha(x: usize, y: usize) -> u8 {
    255 - ((x * 7 + y * 13) % 128) as u8
}

fn surf_rgba8(p: &[[u8; 3]], w: usize, bgra: bool, alpha: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(p.len() * 4);
    for (i, px) in p.iter().enumerate() {
        let a = if alpha { surf_alpha(i % w, i / w) } else { 255 };
        if bgra {
            out.extend_from_slice(&[px[2], px[1], px[0], a]);
        } else {
            out.extend_from_slice(&[px[0], px[1], px[2], a]);
        }
    }
    out
}

/// 16-bit RGBA, native endian: `v * 257` plus a position-dependent low-order offset so the
/// 16-bit path does not reduce to the 8-bit one.
fn surf_rgba16(p: &[[u8; 3]], w: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(p.len() * 8);
    for (i, px) in p.iter().enumerate() {
        let (x, y) = (i % w, i / w);
        for (c, &v8) in px.iter().enumerate() {
            let v = (v8 as u32 * 257).saturating_sub(((x + 2 * y + c) % 97) as u32) as u16;
            out.extend_from_slice(&v.to_ne_bytes());
        }
        let a = surf_alpha(x, y) as u16 * 257;
        out.extend_from_slice(&a.to_ne_bytes());
    }
    out
}

fn surf_srgb_to_linear(v: u8) -> f32 {
    let c = v as f64 / 255.0;
    (if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }) as f32
}

/// Linear-light f32 RGBA, native endian, opaque.
fn surf_linear_rgba(p: &[[u8; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(p.len() * 16);
    for px in p {
        for &v8 in px {
            out.extend_from_slice(&surf_srgb_to_linear(v8).to_ne_bytes());
        }
        out.extend_from_slice(&1.0f32.to_ne_bytes());
    }
    out
}

/// Three linear planes with a padded stride.
fn surf_linear_planes(p: &[[u8; 3]], w: usize, h: usize, stride: usize) -> [Vec<f32>; 3] {
    let mut planes = [vec![0.0f32; stride * h], vec![0.0f32; stride * h], vec![0.0f32; stride * h]];
    for y in 0..h {
        for x in 0..w {
            let px = p[y * w + x];
            for c in 0..3 {
                planes[c][y * stride + x] = surf_srgb_to_linear(px[c]);
            }
        }
    }
    planes
}

/// 8-bit RGB with a padded row stride.
fn surf_strided_rgb(p: &[[u8; 3]], w: usize, h: usize, pad: usize) -> Vec<u8> {
    let stride = w * 3 + pad;
    let mut out = vec![0x5au8; stride * h];
    for y in 0..h {
        for x in 0..w {
            out[y * stride + x * 3..y * stride + x * 3 + 3].copy_from_slice(&p[y * w + x]);
        }
    }
    out
}

fn surf_sb(
    data: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    f: zensim::PixelFormat,
    a: zensim::AlphaMode,
) -> zensim::StridedBytes<'_> {
    zensim::StridedBytes::with_alpha_mode(data, w, h, stride, f, a)
}

fn extra_surfaces(
    profiles: &[(&str, zensim::ZensimProfile)],
    pairs: &[SplitPair],
    line: &dyn Fn(&str, &str, &str, String),
) {
    use zensim::{
        AlphaMode, ColorPrimaries, DiffmapOptions, DiffmapWeighting, PixelFormat, RgbSlice, Zensim,
    };
    for pair in pairs {
        let (w, h) = (pair.width, pair.height);
        let id = pair.id.as_str();
        let srcp: &[[u8; 3]] = bytemuck::cast_slice(&pair.src);
        let dstp: &[[u8; 3]] = bytemuck::cast_slice(&pair.dst);
        let src_rgba = surf_rgba8(srcp, w, false, true);
        let dst_rgba = surf_rgba8(dstp, w, false, true);
        let src_bgra = surf_rgba8(srcp, w, true, false);
        let dst_bgra = surf_rgba8(dstp, w, true, false);
        let src_16 = surf_rgba16(srcp, w);
        let dst_16 = surf_rgba16(dstp, w);
        let src_lin = surf_linear_rgba(srcp);
        let dst_lin = surf_linear_rgba(dstp);
        let src_str = surf_strided_rgb(srcp, w, h, 7);
        let dst_str = surf_strided_rgb(dstp, w, h, 7);
        let src_flat: &[u8] = &pair.src;
        let dst_flat: &[u8] = &pair.dst;
        let lin_stride = w + 3;
        let src_planes = surf_linear_planes(srcp, w, h, lin_stride);
        let dst_planes = surf_linear_planes(dstp, w, h, lin_stride);
        let src = RgbSlice::new(srcp, w, h);
        let dst = RgbSlice::new(dstp, w, h);
        for (pname, profile) in profiles {
            let z = Zensim::new(*profile);
            let sb = |data, stride, f, a| surf_sb(data, w, h, stride, f, a);
            line(
                "rgba_straight",
                pname,
                id,
                surf_result(z.compute(
                    &sb(&src_rgba, w * 4, PixelFormat::Srgb8Rgba, AlphaMode::Straight),
                    &sb(&dst_rgba, w * 4, PixelFormat::Srgb8Rgba, AlphaMode::Straight),
                )),
            );
            line(
                "bgra_opaque",
                pname,
                id,
                surf_result(z.compute(
                    &sb(&src_bgra, w * 4, PixelFormat::Srgb8Bgra, AlphaMode::Opaque),
                    &sb(&dst_bgra, w * 4, PixelFormat::Srgb8Bgra, AlphaMode::Opaque),
                )),
            );
            line(
                "rgba16_straight",
                pname,
                id,
                surf_result(z.compute(
                    &sb(&src_16, w * 8, PixelFormat::Srgb16Rgba, AlphaMode::Straight),
                    &sb(&dst_16, w * 8, PixelFormat::Srgb16Rgba, AlphaMode::Straight),
                )),
            );
            line(
                "linear_f32",
                pname,
                id,
                surf_result(z.compute(
                    &sb(&src_lin, w * 16, PixelFormat::LinearF32Rgba, AlphaMode::Opaque),
                    &sb(&dst_lin, w * 16, PixelFormat::LinearF32Rgba, AlphaMode::Opaque),
                )),
            );
            for (name, prim) in [("p3", ColorPrimaries::DisplayP3), ("bt2020", ColorPrimaries::Bt2020)] {
                line(
                    name,
                    pname,
                    id,
                    surf_result(z.compute(
                        &sb(src_flat, w * 3, PixelFormat::Srgb8Rgb, AlphaMode::Opaque)
                            .with_color_primaries(prim),
                        &sb(dst_flat, w * 3, PixelFormat::Srgb8Rgb, AlphaMode::Opaque)
                            .with_color_primaries(prim),
                    )),
                );
            }
            line(
                "strided",
                pname,
                id,
                surf_result(z.compute(
                    &sb(&src_str, w * 3 + 7, PixelFormat::Srgb8Rgb, AlphaMode::Opaque),
                    &sb(&dst_str, w * 3 + 7, PixelFormat::Srgb8Rgb, AlphaMode::Opaque),
                )),
            );
            line(
                "serial",
                pname,
                id,
                surf_result(Zensim::new(*profile).with_parallel(false).compute(&src, &dst)),
            );
            // The research crate's options carry one more field than the slim crate's.
            #[allow(clippy::needless_update)]
            let full = DiffmapOptions {
                weighting: DiffmapWeighting::Balanced,
                masking_strength: Some(4.0),
                sqrt: true,
                include_edge_mse: true,
                include_hf: true,
                ..Default::default()
            };
            line(
                "diffmap_full_options",
                pname,
                id,
                surf_diffmap(z.compute_with_diffmap(&src, &dst, full)),
            );
            let custom = DiffmapOptions {
                weighting: DiffmapWeighting::Custom([0.2, 0.5, 0.3]),
                include_edge_mse: true,
                ..Default::default()
            };
            line(
                "diffmap_custom_edges",
                pname,
                id,
                surf_diffmap(z.compute_with_diffmap(&src, &dst, custom)),
            );
            let lp = z.precompute_reference_linear_planar(
                [&src_planes[0], &src_planes[1], &src_planes[2]],
                w,
                h,
                lin_stride,
            );
            line(
                "linear_planar_diffmap",
                pname,
                id,
                match &lp {
                    Ok(pre) => surf_diffmap(z.compute_with_ref_and_diffmap_linear_planar(
                        pre,
                        [&dst_planes[0], &dst_planes[1], &dst_planes[2]],
                        w,
                        h,
                        lin_stride,
                        DiffmapOptions::default(),
                    )),
                    Err(e) => format!("ERR {e:?}"),
                },
            );
            #[cfg(feature = "classification")]
            line(
                "classify",
                pname,
                id,
                match z.classify(&src, &dst) {
                    // The result's Debug names the profile, which the two crates spell
                    // differently; its numbers go through `surf_result`.
                    Ok(c) => format!(
                        "{} {:?} {:?}",
                        surf_result(Ok(c.result)),
                        c.classification,
                        c.delta_stats
                    ),
                    Err(e) => format!("ERR {e:?}"),
                },
            );
        }
        #[cfg(feature = "training")]
        {
            use zensim::{BlurKernel, ZensimConfig};
            let configs: Vec<(&str, ZensimConfig)> = {
                let base = ZensimConfig::default();
                let mut v = vec![("default", base)];
                let mut c = base;
                c.blur_radius = 3;
                v.push(("radius3", c));
                let mut c = base;
                c.blur_passes = 3;
                c.blur_kernel = BlurKernel::Box { passes: 3 };
                v.push(("passes3", c));
                let mut c = base;
                c.num_scales = 3;
                v.push(("scales3", c));
                let mut c = base;
                c.extended_features = true;
                v.push(("extended", c));
                let mut c = base;
                c.extended_features = true;
                c.compute_iw_features = true;
                v.push(("extended_iw", c));
                let mut c = base;
                c.compute_all_features = true;
                v.push(("all_features", c));
                let mut c = base;
                c.allow_multithreading = false;
                v.push(("serial", c));
                v
            };
            for (cname, config) in &configs {
                line(
                    "config",
                    cname,
                    id,
                    surf_result(zensim::compute_zensim_with_config(
                        srcp, dstp, w, h, *config,
                    )),
                );
                let pre = zensim::precompute_reference_with_scales(srcp, w, h, config.num_scales);
                line(
                    "config_with_ref",
                    cname,
                    id,
                    match &pre {
                        Ok(pre) => surf_result(zensim::compute_zensim_with_ref_and_config(
                            pre, dstp, w, h, *config,
                        )),
                        Err(e) => format!("ERR {e:?}"),
                    },
                );
            }
            for (pname, params) in [
                ("custom_v0_2", zensim::profile::ProfileParams::custom(zensim::WEIGHTS, 5, 1, 4, 18.0, 0.7)),
                ("custom_r3p3", zensim::profile::ProfileParams::custom(zensim::WEIGHTS, 3, 3, 4, 18.0, 0.7)),
            ] {
                line(
                    "with_params",
                    pname,
                    id,
                    surf_result(Zensim::compute_with_params(&params, &src, &dst)),
                );
            }
            if let Ok(r) = Zensim::new(zensim::ZensimProfile::PreviewV0_2).compute(&src, &dst) {
                line(
                    "score_from_features",
                    "PreviewV0_2",
                    id,
                    match zensim::try_score_from_features(r.features(), zensim::WEIGHTS) {
                        Ok((s, d)) => format!("{} {}", surf_hex(s), surf_hex(d)),
                        Err(e) => format!("ERR {e:?}"),
                    },
                );
            }
        }
    }
}
