// Export-only inspection of the planes retained by the production fold.
// No finalizer, accumulator or default build calls this module. Shared term
// helpers evaluate the same per-pixel expressions; pooling stays with the fold.
// This implementation intentionally refuses revisions other than 7.

impl Vf32 for f32 {
    type Tok = ();
    fn splat_t(_: (), v: f32) -> Self {
        v
    }
    fn zero_t(_: ()) -> Self {
        0.0
    }
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn simd_gt(self, other: Self) -> Self {
        if self > other { 1.0 } else { 0.0 }
    }
    fn simd_lt(self, other: Self) -> Self {
        if self < other { 1.0 } else { 0.0 }
    }
    fn blend(mask: Self, yes: Self, no: Self) -> Self {
        if mask != 0.0 { yes } else { no }
    }
}

pub(crate) fn viz_export_maps(
    ret: &FoldRetention,
    slots: &crate::feature_set_id::SlotSet,
    original: (usize, usize),
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    if ret.revision != FormulaRevision::Rev7 {
        return Err("atlas requires revision 7".into());
    }
    if active_ring_form() != RingForm::Raw {
        return Err("atlas currently requires the registered raw ringing form".into());
    }
    let out = std::path::PathBuf::from(
        std::env::var_os("ZENSIM_VIZ_MAP_DIR").ok_or("map directory unset")?,
    );
    std::fs::create_dir_all(&out)?;
    let mut records = Vec::new();
    for (scale, &(w, h)) in ret.dims.iter().enumerate() {
        for ch in 0..3 {
            let src = &ret.pyr_src[scale][ch];
            let dst = &ret.pyr_dst[scale][ch];
            let p = &ret.planes[scale][ch];
            if src.is_empty() || dst.is_empty() || p.mu1.is_empty() {
                continue;
            }
            let n = w * h;
            let basic_terms = if slots.iter_slots().any(|id| id < 156) && !ret.basic_sd.is_empty() {
                Some(viz_export_basic_terms(ret, scale, ch, original, &out)?)
            } else {
                None
            };
            // One source for each detector; features with different pooling share
            // the detector map. Numerators of weighted means retain their weight.
            let mut maps = vec![vec![0.0f32; n]; 29];
            let mut sh = Vec::with_capacity((h + 2) * w);
            let mut dh = Vec::with_capacity((h + 2) * w);
            for y in 0..h + 2 {
                let yy = reflect_101(y as isize - 1, h);
                sh.extend_from_slice(&src[yy * w..(yy + 1) * w]);
                dh.extend_from_slice(&dst[yy * w..(yy + 1) * w]);
            }
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let (t, _, _, _, _, _, mv, iv, kn, _) = dense_terms32(
                        src[i], dst[i], p.mu1[i], p.mu2[i], p.ssq[i], p.s12[i], p.act[i], true,
                        true, true,
                    );
                    for (j, k) in [0, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_iter().enumerate() {
                        maps[j][i] = t[k];
                    }
                    for j in 0..4 {
                        maps[10 + j][i] = mv[j];
                        maps[14 + j][i] = iv[j];
                    }
                    for j in 0..3 {
                        maps[18 + j][i] = kn[j];
                    }
                    let (gsrc, gms, ring, band) = if x == 0 || x + 1 == w {
                        let g = gradient_terms64::<false, false, false>(
                            &sh,
                            &dh,
                            &p.act,
                            &[],
                            &[],
                            &[],
                            w,
                            x,
                            y,
                            0.0,
                            0.0,
                            true,
                        );
                        (g.gsrc as f32, g.g as f32, g.ring as f32, g.band as f32)
                    } else {
                        let row = (y + 1) * w;
                        let g = gradient_terms32_v::<f32, false, false, false>(
                            (),
                            sh[row + x - 1],
                            sh[row + x + 1],
                            sh[row - w + x],
                            sh[row + w + x],
                            dh[row + x - 1],
                            dh[row + x + 1],
                            dh[row - w + x],
                            dh[row + w + x],
                            src[i],
                            dst[i],
                            p.act[i],
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            true,
                        );
                        (g.0, g.2, g.4, g.5)
                    };
                    maps[21][i] = gsrc;
                    maps[22][i] = gms;
                    maps[23][i] = ring;
                    maps[24][i] = band;
                    // Basic features use their own SSIM expression, retained by the
                    // fused kernel, and their unbounded artifact/detail ratio.
                    if !ret.basic_sd.is_empty() {
                        maps[25][i] = ret.basic_sd[scale][ch][i];
                    }
                    let e =
                        (1.0 + (dst[i] - p.mu2[i]).abs()) / (1.0 + (src[i] - p.mu1[i]).abs()) - 1.0;
                    maps[26][i] = e;
                    let mut lattice = 0.0;
                    if x > 0 && x.is_multiple_of(BLOCK_LATTICE) {
                        lattice += bounded_excess(
                            (f64::from(dst[i]) - f64::from(dst[i - 1])).abs(),
                            (f64::from(src[i]) - f64::from(src[i - 1])).abs(),
                            C_BLOCK,
                        );
                    }
                    if y > 0 && y.is_multiple_of(BLOCK_LATTICE) {
                        lattice += bounded_excess(
                            (f64::from(dst[i]) - f64::from(dst[i - w])).abs(),
                            (f64::from(src[i]) - f64::from(src[i - w])).abs(),
                            C_BLOCK,
                        );
                    }
                    maps[28][i] = lattice as f32;
                }
            }
            type MapRecord = (String, [f32; 2], usize, usize);
            let mut cache: std::collections::BTreeMap<(usize, i32), MapRecord> =
                std::collections::BTreeMap::new();
            for id in slots.iter_slots() {
                let Some(def) = crate::feature_defs::def_at(id, crate::NUM_SCALES) else {
                    continue;
                };
                if usize::from(def.scale) != scale || def.channel.as_str() != ["x", "y", "b"][ch] {
                    continue;
                }
                let (map_index, role) = if id < 156 {
                    match id % 13 {
                        0..=2 => (Some(25), "detector before mean/p-power pooling"),
                        3..=5 => (Some(26), "positive artifact excess before pooling"),
                        6..=8 => (Some(26), "positive detail loss before pooling"),
                        9 => (Some(27), "squared XYB error before mean pooling"),
                        _ => (None, "global ratio of high-frequency moments"),
                    }
                } else if (156..228).contains(&id) {
                    match (id - 156) % 6 {
                        0 | 3 => (Some(25), "SSIM detector before max/L8 pooling"),
                        1 | 4 => (Some(26), "artifact detector before max/L8 pooling"),
                        _ => (Some(26), "detail-loss detector before max/L8 pooling"),
                    }
                } else if (372..720).contains(&id) {
                    match (id - 372) % 29 {
                        0..=2 => (Some(0), "SSIM detector before mean/central-moment pooling"),
                        3 => (Some(1), "artifact detector"),
                        4 => (Some(2), "detail-loss detector"),
                        5 => (Some(3), "bounded squared error"),
                        6 => (Some(4), "high-frequency gain"),
                        7 => (Some(5), "high-frequency loss"),
                        8 => (Some(6), "high-frequency magnitude loss"),
                        9..=11 => (
                            Some(18 + (id - 372) % 29 - 9),
                            "saliency-weighted numerator; feature divides its global weight sum",
                        ),
                        12..=15 => (
                            Some(10 + (id - 372) % 29 - 12),
                            "mask-weighted numerator; feature divides its global weight sum",
                        ),
                        16..=19 => (
                            Some(14 + (id - 372) % 29 - 16),
                            "information-weighted numerator; feature divides its global weight sum",
                        ),
                        20 => (Some(7), "masked JND detector"),
                        21 => (None, "global nonlinear reference-gradient statistic"),
                        22 => (Some(22), "gradient magnitude similarity detector"),
                        23 => (Some(8), "low-k masked JND detector"),
                        24 => (Some(9), "high-k masked JND detector"),
                        25 => (Some(28), "sparse lattice step energy before mean pooling"),
                        26 => (Some(23), "raw ringing detector"),
                        27 => (Some(24), "banding detector"),
                        _ => (None, "global adjacent-scale gradient ratio"),
                    }
                } else {
                    (None, "map export not implemented for this family")
                };
                let Some(index) = map_index else {
                    records.push(serde_json::json!({"id":id,"map_file":null,"map_status":role}));
                    continue;
                };
                let variant = if id < 156 {
                    match id % 13 {
                        3..=5 => 1,
                        6..=8 => 2,
                        _ => 0,
                    }
                } else if id < 228 {
                    match (id - 156) % 6 {
                        1 | 4 => 1,
                        2 | 5 => 2,
                        _ => 0,
                    }
                } else {
                    0
                };
                let key = (index, variant);
                let (file, range, mw, mh) = if let Some(value) = cache.get(&key) {
                    value.clone()
                } else {
                    let values: Vec<f32> = if index == 27 {
                        src.iter()
                            .zip(dst)
                            .map(|(a, b)| (a - b) * (a - b))
                            .collect()
                    } else {
                        maps[index]
                            .iter()
                            .map(|&v| match variant {
                                1 => v.max(0.0),
                                2 => (-v).max(0.0),
                                _ => v,
                            })
                            .collect()
                    };
                    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let logical_w = (original.0 >> scale).max(1).min(w);
                    let logical_h = (original.1 >> scale).max(1).min(h);
                    let factor = logical_w.max(logical_h).div_ceil(512);
                    let mw = logical_w.div_ceil(factor);
                    let mh = logical_h.div_ceil(factor);
                    let mut rgb = Vec::with_capacity(mw * mh);
                    for yy in 0..mh {
                        for xx in 0..mw {
                            let v = values[(yy * factor).min(h - 1) * w + (xx * factor).min(w - 1)];
                            let q = if hi == lo {
                                0
                            } else {
                                ((v - lo) / (hi - lo) * 255.0).round() as u8
                            };
                            rgb.push(rgb::Rgb::new(q, q, q));
                        }
                    }
                    let file = format!("s{scale}-{ch}-{index}-{variant}.png");
                    let bytes = zenpng::encode_rgb8(
                        imgref::ImgRef::new(&rgb, mw, mh),
                        None,
                        &zenpng::EncodeConfig::default(),
                        &enough::Unstoppable,
                        &enough::Unstoppable,
                    )?;
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(out.join(&file))?
                        .write_all(&bytes)?;
                    let value = (file, [lo, hi], mw, mh);
                    cache.insert(key, value.clone());
                    value
                };
                records.push(serde_json::json!({"id":id,"map_file":file,"map_range":range,"map_width":mw,"map_height":mh,"native_grid":[(original.0>>scale).max(1).min(w),(original.1>>scale).max(1).min(h)],"pooling_grid":[w,h],"map_role":role,"terms_file":if id < 156 { basic_terms.as_ref() } else { None }}));
            }
        }
    }
    serde_json::to_writer(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("maps.json"))?,
        &records,
    )?;
    Ok(())
}

// Raw f32 terms are separate from quantized display maps. Sample coordinates,
// crop and stride exactly match those maps; no metric accumulator is changed.
fn viz_export_basic_terms(
    ret: &FoldRetention,
    scale: usize,
    ch: usize,
    original: (usize, usize),
    out: &std::path::Path,
) -> Result<String, Box<dyn std::error::Error>> {
    use std::io::Write;
    let (w, h) = ret.dims[scale];
    let src = &ret.pyr_src[scale][ch];
    let dst = &ret.pyr_dst[scale][ch];
    let p = &ret.planes[scale][ch];
    let logical = [
        (original.0 >> scale).max(1).min(w),
        (original.1 >> scale).max(1).min(h),
    ];
    let step = logical[0].max(logical[1]).div_ceil(512);
    let mw = logical[0].div_ceil(step);
    let mh = logical[1].div_ceil(step);
    let names = [
        "mu_r",
        "mu_d",
        "src",
        "dst",
        "square_moment",
        "error_moment",
        "variance_sum",
        "error_variance",
        "d1",
        "edge",
        "artifact",
        "detail",
        "mse",
    ];
    let mut planes: [Vec<f32>; 13] = core::array::from_fn(|_| Vec::with_capacity(mw * mh));
    for yy in 0..mh {
        for xx in 0..mw {
            let i = yy * step * w + xx * step;
            let m1 = p.mu1[i];
            let m2 = p.mu2[i];
            let delta = m1 - m2;
            let variance = (-m2).mul_add(m2, (-m1).mul_add(m1, p.ssq[i])).max(0.0);
            let error_variance = (p.s12[i] - delta * delta).max(0.0);
            let d1 = ret.basic_sd[scale][ch][i];
            let replay = crate::ssim_form::ssim_direct_raw_scalar(
                crate::ssim_form::SsimLumaForm::Clamp,
                m1,
                m2,
                p.ssq[i],
                p.s12[i],
            )
            .max(0.0);
            if replay.to_bits() != d1.to_bits() {
                return Err(
                    format!("basic formula replay differs at s{scale}/{ch}/({xx},{yy})").into(),
                );
            }
            let edge = (1.0 + (dst[i] - m2).abs()) / (1.0 + (src[i] - m1).abs()) - 1.0;
            let error = src[i] - dst[i];
            let values = [
                m1,
                m2,
                src[i],
                dst[i],
                p.ssq[i],
                p.s12[i],
                variance,
                error_variance,
                d1,
                edge,
                edge.max(0.0),
                (-edge).max(0.0),
                error * error,
            ];
            for (plane, value) in planes.iter_mut().zip(values) {
                if !value.is_finite() {
                    return Err("nonfinite basic formula term".into());
                }
                plane.push(value);
            }
        }
    }
    let file = format!("basic-terms-s{scale}-{ch}");
    let mut binary = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(format!("{file}.f32le")))?,
    );
    for plane in &planes {
        for value in plane {
            binary.write_all(&value.to_le_bytes())?;
        }
    }
    binary.flush()?;
    let header = format!("{file}.json");
    serde_json::to_writer(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(&header))?,
        &serde_json::json!({"schema":"zensim-basic-terms-v1","layout":"plane-major-f32le",
            "width":mw,"height":mh,"native_grid":logical,"pooling_grid":[w,h],"step":step,
            "scale":scale,"channel":ch,"planes":names,"data_file":format!("{file}.f32le"),
            "window":"same retained box window as extraction; reflected boundaries",
            "formula":"revision-7 Clamp direct-error; f32 mul_add; nonnegative result",
            "formula_replay_bits_checked":mw*mh,"formula_replay_bit_identity":true}),
    )?;
    Ok(header)
}
