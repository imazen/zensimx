use super::*;

// The retained canonical Rev5 path writes its five moment/activity cores
// directly. Owned input cores are still copied for snapshot reuse.

#[derive(Clone, Copy)]
pub(super) struct RetainedCoreWork {
    pub(super) toggles: V2NewFeatureToggles,
    pub(super) fold_v1: bool,
    pub(super) free: crate::fused::FreeExtrasWork,
}

pub(super) fn retained_core_admits(local: &ComputeSet) -> bool {
    local.formula_revision >= FormulaRevision::Rev5
        && matches!(
            crate::featcanon::mode(local.formula_revision),
            crate::featcanon::Mode::Canon64
        )
        && matches!(
            crate::featcanon::canon_blur_axis_for(
                crate::featcanon::mode(local.formula_revision),
                local.formula_revision
            ),
            crate::featcanon::BlurMode::Local
        )
        && local.v2_blocks
        && !(local.append
            || local.append2
            || local.csfw
            || local.gridblk
            || local.ringbasis
            || local.tailhist
            || local.arttype
            || local.gmsbank
            || local.gmsnative
            || local.mapdev
            || local.z1max)
}

impl RetainedRows<'_> {
    pub(super) fn moment_rows(&self) -> [&[f32]; 5] {
        core::array::from_fn(|i| &self.planes[i + 2][..])
    }
}

#[inline(never)]
pub(super) fn rev5_blur_into_retained(
    info: &crate::feature_v2_stream::StripInfo,
    scratch: &mut ScratchV2Strip,
    raw: [Option<&[f32]>; 2],
    output: &mut RetainedRows<'_>,
) {
    debug_assert!(output.bs2.is_none());
    let n = info.plane_w * info.wide_h();
    let width = info.plane_w;
    let height = info.wide_h();
    let first = HALO_P;
    let end = first + info.strip_h;
    let off = first * width;
    let core_n = info.strip_h * width;
    let ScratchV2Strip {
        src_wide,
        dst_wide,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        ..
    } = scratch;
    let src = raw[0].unwrap_or(&src_wide[..n]);
    let dst = raw[1].unwrap_or(&dst_wide[..n]);
    let [pyr_src, pyr_dst, m1, m2, sq, err, act] = &mut output.planes;
    pyr_src.copy_from_slice(&src[off..off + core_n]);
    pyr_dst.copy_from_slice(&dst[off..off + core_n]);

    let start = crate::fold_timing::start();
    fused_blur_h_ssim_banded(
        src,
        dst,
        &mut mu1_h[..n],
        &mut mu2_h[..n],
        &mut ssq_h[..n],
        &mut s12_h[..n],
        width,
        height,
        false,
        FormulaRevision::Rev5,
    );
    crate::fold_timing::stop(start, crate::fold_timing::Phase::BlurHWall, info.scale);
    let start = crate::fold_timing::start();

    // The activity chain needs mean-reference halo values. Its core lives
    // only in the final buffer; the two halo intervals remain in scratch.
    crate::blur::box_blur_v_local_rows(&mu1_h[..n], &mut mu1[..off], width, height, 0..first);
    crate::blur::box_blur_v_local_rows(&mu1_h[..n], m1, width, height, first..end);
    crate::blur::box_blur_v_local_rows(
        &mu1_h[..n],
        &mut mu1[end * width..n],
        width,
        height,
        end..height,
    );
    for (horizontal, core) in [
        (&mu2_h[..n], &mut **m2),
        (&ssq_h[..n], &mut **sq),
        (&s12_h[..n], &mut **err),
    ] {
        crate::blur::box_blur_v_local_rows(horizontal, core, width, height, first..end);
    }

    crate::simd_ops::abs_diff_into(&src[..off], &mu1[..off], &mut mu1_h[..off]);
    crate::simd_ops::abs_diff_into(&src[off..off + core_n], m1, &mut mu1_h[off..off + core_n]);
    crate::simd_ops::abs_diff_into(
        &src[end * width..n],
        &mu1[end * width..n],
        &mut mu1_h[end * width..n],
    );
    crate::blur::box_blur_h_at_revision(
        &mu1_h[..n],
        &mut mu2_h[..n],
        width,
        height,
        BLUR_RADIUS,
        FormulaRevision::Rev5,
    );
    crate::blur::box_blur_v_local_rows(&mu2_h[..n], act, width, height, first..end);
    // Five logical vertical planes; the mean-reference plane is partitioned
    // into two scratch halos and the retained core.
    crate::fold_timing::work(crate::fold_timing::Work::VerticalPlane, 5);
    crate::fold_timing::work(crate::fold_timing::Work::ActivityChain, 1);
    crate::fold_timing::work(crate::fold_timing::Work::V2Cell, 1);
    if let Some(sd) = &mut output.basic {
        incant!(retain_basic_sd(m1, m2, sq, err, sd), [v4x, v4, v3, scalar]);
    }
    crate::fold_timing::stop(start, crate::fold_timing::Phase::PhaseAV2Planes, info.scale);
}

#[inline(never)]
pub(super) fn rev5_retained_phase_b(
    info: &crate::feature_v2_stream::StripInfo,
    raw: [&[f32]; 2],
    planes: [&[f32]; 5],
    work: RetainedCoreWork,
    acc: &mut StreamChannelAccums,
    bands: &mut [V1BasicSums],
) {
    let width = info.plane_w;
    let n = width * info.strip_h;
    let off = HALO_P * width;
    let [m1, m2, sq, err, act] = planes;
    let start = crate::fold_timing::start();
    let d = dense_block_kernel_at_revision(
        &raw[0][off..off + n],
        &raw[1][off..off + n],
        m1,
        m2,
        sq,
        err,
        act,
        width,
        info.strip_h,
        work.toggles.transducer_bank,
        None,
        work.toggles.formula_revision,
    );
    crate::fold_timing::stop(start, crate::fold_timing::Phase::DenseKernel, info.scale);
    acc.dense[info.scale].accumulate(&d);
    let mut gradient = GradientAccum::default();
    if work.toggles.gradient_features {
        let start = crate::fold_timing::start();
        let g_off = (HALO_P - 1) * width;
        let g_n = width * (info.strip_h + 2);
        gradient = gradient_block_kernel_at_revision(
            &raw[0][g_off..g_off + g_n],
            &raw[1][g_off..g_off + g_n],
            act,
            Some((m1, m2)),
            width,
            info.strip_h,
            None,
            None,
            None,
            false,
            work.toggles.formula_revision,
        );
        crate::fold_timing::stop(start, crate::fold_timing::Phase::GradKernel, info.scale);
        acc.grad[info.scale].accumulate(&gradient);
    }
    if let Some(strips) = &mut acc.retained_strips
        && (1..=3).contains(&info.scale)
    {
        strips[info.scale].push((d, gradient));
    }
    if !work.fold_v1 {
        return;
    }
    let count = info.strip_h.div_ceil(V1_BAND_ROWS);
    let run = |(index, band): (usize, &mut V1BasicSums)| {
        let first = index * V1_BAND_ROWS;
        let end = (first + V1_BAND_ROWS).min(info.strip_h);
        let core = first * width..end * width;
        let input = off + first * width..off + end * width;
        *band = V1BasicSums::at_revision(work.toggles.formula_revision);
        // Radius zero reads only these inner rows. The same canonical owner
        // executes the same pixels in the same 32-row band order.
        band.accumulate(&crate::fused::fused_vblur_features_ssim(
            &m1[core.clone()],
            &m2[core.clone()],
            &sq[core.clone()],
            &err[core],
            &raw[0][input.clone()],
            &raw[1][input],
            width,
            end - first,
            0,
            end - first,
            0,
            &mut [],
            &mut [],
            false,
            &mut [],
            false,
            &mut [],
            &mut [],
            false,
            work.free,
            crate::fused::ExtPoolsWork::default(),
            &[],
        ));
    };
    bands[..count].iter_mut().enumerate().for_each(run);
}

// Keep retained-only scratch and dispatch out of the score walk's caller.
#[inline(never)]
pub(super) fn rev5_retained_single_channel<S: ImageSource, D: ImageSource>(
    producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    scratch_and_accum: (&mut ScratchV2Strip, &mut StreamChannelAccums),
    output: &mut RetainedRows<'_>,
    local: &ComputeSet,
    work: RetainedCoreWork,
) {
    let (scr, acc) = scratch_and_accum;
    let scale = info.scale;
    let fold_v1 = work.fold_v1;
    stream_gather_windows(producer, info, ch, scr);
    let overrides = [
        producer.wide_window(crate::feature_v2_stream::Side::Source, ch, info),
        producer.wide_window(crate::feature_v2_stream::Side::Distorted, ch, info),
    ];
    rev5_blur_into_retained(info, scr, overrides, output);
    let (src, dst) = stream_windows_shared(producer, info, ch, scr);
    let mut bands: [V1BasicSums; V1_BANDS_PER_STRIP] =
        core::array::from_fn(|_| V1BasicSums::at_revision(work.toggles.formula_revision));
    rev5_retained_phase_b(
        info,
        [src, dst],
        output.moment_rows(),
        work,
        acc,
        &mut bands,
    );
    if fold_v1 {
        for band in &bands[..info.strip_h.div_ceil(V1_BAND_ROWS)] {
            acc.v1[scale].merge(band);
        }
    }
    if local.blockiness {
        let (sum_v, sum_h) = &mut acc.block[scale];
        blockiness_sparse_strip_wide(
            src,
            dst,
            info.plane_w,
            info.y0,
            info.strip_h,
            local.formula_revision,
            sum_v,
            sum_h,
        );
    }
}

#[cfg(test)]
mod retained_core_tests {
    use super::*;

    #[test]
    fn one_thread_retention_keeps_blockiness_features() {
        const NAME: &str = "feature_v2::retained_core::retained_core_tests::one_thread_retention_keeps_blockiness_features";
        if std::env::var_os("PERF3_RETAINED_CORE_CHILD").is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture", "--test-threads", "1"])
                .env("ZENSIM_FORMULA_REV", "5")
                .env("PERF3_RETAINED_CORE_CHILD", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("PERF3_RETAINED_CORE complete")
            );
            return;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                for (width, height) in [(64, 64), (97, 151), (257, 129)] {
                    let src = vec![[96_u8; 3]; width * height];
                    let dst: Vec<_> = (0..width * height)
                        .map(|i| {
                            let shade = if ((i % width) / 8 + (i / width) / 8) % 2 == 0 {
                                40_u8
                            } else {
                                160_u8
                            };
                            if width == 64 && height == 64 {
                                [shade; 3]
                            } else {
                                [
                                    shade,
                                    shade.saturating_add(((i % width) % 17) as u8),
                                    shade.saturating_sub(((i / width) % 13) as u8),
                                ]
                            }
                        })
                        .collect();
                    let source = crate::source::RgbSlice::new(&src, width, height);
                    let distorted = crate::source::RgbSlice::new(&dst, width, height);
                    let toggles = V2NewFeatureToggles {
                        formula_revision: FormulaRevision::Rev5,
                        v1_pools: V1PoolsMode::Off,
                        ..Default::default()
                    };
                    assert!(retained_core_admits(
                        &ComputeSet::from_toggles(toggles).at_scale(0)
                    ));
                    let reference = foldapp_streaming_walk(
                        &source,
                        &distorted,
                        true,
                        toggles,
                        crate::feature_v2_stream::FrontEnd::Sdr,
                        &mut V2Scratch::new(),
                        FoldWalkExtras::default(),
                    );
                    assert!((0..crate::NUM_SCALES).any(|scale| (0..3).any(
                        |ch| reference.features()[372
                            + (scale * 3 + ch) * FEATURES_PER_CHANNEL_V2_TOTAL
                            + idx::BLOCKINESS]
                            > 0.0
                    )));
                    let mut retained = FoldRetention::new();
                    let actual = foldapp_streaming_walk(
                        &source,
                        &distorted,
                        true,
                        toggles,
                        crate::feature_v2_stream::FrontEnd::Sdr,
                        &mut V2Scratch::new(),
                        FoldWalkExtras {
                            retention: Some(&mut retained),
                            ..Default::default()
                        },
                    );
                    assert_eq!(reference.features().len(), actual.features().len());
                    for (i, (expected, actual)) in reference
                        .features()
                        .iter()
                        .zip(actual.features())
                        .enumerate()
                    {
                        assert_eq!(
                            expected.to_bits(),
                            actual.to_bits(),
                            "{width}x{height}/feature {i}"
                        );
                    }
                }
            });
        println!("PERF3_RETAINED_CORE complete");
    }
}
