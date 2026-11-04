//! Ordered batch parity with partial strips and retained scratch of varying sizes.
use super::*;
use crate::{PixelFormat, RgbSlice, StridedBytes};

fn vector_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * core::mem::size_of::<T>()
}

fn retained_queue_bytes(jobs: &Vec<Rev5StripJob>) -> usize {
    vector_bytes(jobs)
        + jobs
            .iter()
            .map(|job| {
                let s = &job.scratch;
                let r = &job.result;
                let planes = [
                    &s.src_wide,
                    &s.dst_wide,
                    &s.mu1_h,
                    &s.mu2_h,
                    &s.ssq_h,
                    &s.s12_h,
                    &s.mu1,
                    &s.mu2,
                    &s.ssq,
                    &s.s12,
                    &s.abs_src,
                    &s.activity_tmp,
                    &s.activity,
                    &s.bs2,
                    &s.activity_dst,
                ]
                .iter()
                .map(|p| vector_bytes(p))
                .sum::<usize>();
                let results = vector_bytes(&r.dense)
                    + vector_bytes(&r.grad)
                    + vector_bytes(&r.app)
                    + vector_bytes(&r.v1)
                    + vector_bytes(&r.block)
                    + vector_bytes(&r.csfw)
                    + vector_bytes(&r.rev4)
                    + vector_bytes(&r.pool_scratch);
                for pool in &r.pool_scratch {
                    assert!(
                        [
                            &pool.stable_sd,
                            &pool.mu1_v,
                            &pool.mu2_v,
                            &pool.act_raw,
                            &pool.act,
                            &pool.ssq_v,
                            &pool.s12_v
                        ]
                        .iter()
                        .all(|p| p.capacity() == 0)
                    );
                    assert!(pool.h.iter().all(|p| p.capacity() == 0));
                }
                planes + results
            })
            .sum::<usize>()
}

#[test]
fn rev5_job_budget_handles_exact_fit_and_overflow() {
    for width in [1024, 4096, 8192] {
        let max_n = width * (STRIP_ROWS + 2 * HALO_P);
        let fixed = rev5_job_bytes(0).unwrap();
        let per_job = rev5_job_bytes(max_n).unwrap();
        assert_eq!(per_job, 11 * max_n * core::mem::size_of::<f32>() + fixed);
        println!(
            "REV5_JOB_ACCOUNTING {{\"width\":{width},\"strip_rows\":{STRIP_ROWS},\"halo\":{HALO_P},\"planes\":11,\"float_bytes\":{},\"fixed_bytes\":{fixed},\"per_job_bytes\":{per_job}}}",
            core::mem::size_of::<f32>()
        );
    }
    let bytes = rev5_job_bytes(4096 * (STRIP_ROWS + 2 * HALO_P)).unwrap();
    assert_eq!(
        rev5_job_limit(4096 * (STRIP_ROWS + 2 * HALO_P), 32, bytes - 1),
        0
    );
    // One or two fitting jobs fall back to the original route (three fused
    // channels); the queue starts at exactly REV5_MIN_JOB_SLOTS.
    assert_eq!(REV5_MIN_JOB_SLOTS, 3);
    for budget in [bytes, 2 * bytes, 3 * bytes - 1] {
        assert_eq!(
            rev5_job_limit(4096 * (STRIP_ROWS + 2 * HALO_P), 32, budget),
            0,
            "budget {budget} admits fewer than three jobs"
        );
    }
    assert_eq!(
        rev5_job_limit(4096 * (STRIP_ROWS + 2 * HALO_P), 32, 3 * bytes),
        3
    );
    assert_eq!(
        rev5_job_limit(4096 * (STRIP_ROWS + 2 * HALO_P), 32, 32 * bytes),
        16
    );
    assert_eq!(
        rev5_job_limit(4096 * (STRIP_ROWS + 2 * HALO_P), 8, 32 * bytes),
        8
    );
    assert_eq!(rev5_job_limit(usize::MAX, 32, usize::MAX), 0);
}

#[test]
fn ordered_batches_match_serial_with_reused_strided_scratch() {
    const SENTINEL: &str = "REV5_BATCH_PARITY_OK";
    if !crate::ssim_form::run_at_revision(
        "5",
        "feature_v2::rev5_scaling_tests::ordered_batches_match_serial_with_reused_strided_scratch",
        SENTINEL,
    ) {
        return;
    }
    for pools in [V1PoolsMode::Peaks, V1PoolsMode::Off] {
        for budget in [
            None,
            Some(0),
            Some(1024 * 1024),
            Some(2 * 1024 * 1024),
            Some(8 * 1024 * 1024),
            Some(64 * 1024 * 1024),
            Some(128 * 1024 * 1024),
            Some(256 * 1024 * 1024),
        ] {
            let toggles = V2NewFeatureToggles {
                formula_revision: FormulaRevision::Rev5,
                v1_pools: pools,
                ..Default::default()
            };
            for threads in [8, 16, 32] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut scratch = V2Scratch::new();
                scratch.rev5_job_budget = budget;
                for (w, h, identity, mask) in [
                    (257, 1025, false, ComputeSet::ALL_SCALES),
                    (511, 259, false, ComputeSet::ALL_SCALES),
                    (129, 513, true, ComputeSet::ALL_SCALES),
                    (191, 193, false, ComputeSet::ALL_SCALES), // One pixel below the gate.
                    (192, 192, false, ComputeSet::ALL_SCALES), // Exactly at the gate.
                    (192, 193, false, ComputeSet::ALL_SCALES), // Above the gate.
                    (65, 129, false, ComputeSet::ALL_SCALES),
                    (257, 513, false, 0b0101),
                ] {
                    let src: Vec<[u8; 3]> = (0..w * h)
                        .map(|i| {
                            let x = i % w;
                            let y = i / w;
                            [
                                (x * 17 + y * 31) as u8,
                                ((x * 7) ^ (y * 13)) as u8,
                                (x * y + 91) as u8,
                            ]
                        })
                        .collect();
                    let dst: Vec<[u8; 3]> = src
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| {
                            if identity {
                                p
                            } else {
                                [
                                    p[0].wrapping_add((i % 11) as u8),
                                    p[1] & 0xf0,
                                    p[2].saturating_sub(7),
                                ]
                            }
                        })
                        .collect();
                    let mut compute = ComputeSet::from_toggles(toggles);
                    compute.v2_scales = mask;
                    let expected = compute_folded720_streaming_impl(
                        &RgbSlice::new(&src, w, h),
                        &RgbSlice::new(&dst, w, h),
                        None,
                        false,
                        toggles,
                        &mut V2Scratch::new(),
                        Some(compute),
                    )
                    .unwrap()
                    .into_features();
                    let stride = w * 3 + 17;
                    let mut padded_src = vec![0; stride * h];
                    let mut padded_dst = vec![0; stride * h];
                    for y in 0..h {
                        for x in 0..w {
                            let at = y * stride + x * 3;
                            padded_src[at..at + 3].copy_from_slice(&src[y * w + x]);
                            padded_dst[at..at + 3].copy_from_slice(&dst[y * w + x]);
                        }
                    }
                    let source =
                        StridedBytes::try_new(&padded_src, w, h, stride, PixelFormat::Srgb8Rgb)
                            .unwrap();
                    let distorted =
                        StridedBytes::try_new(&padded_dst, w, h, stride, PixelFormat::Srgb8Rgb)
                            .unwrap();
                    let actual = pool
                        .install(|| {
                            compute_folded720_streaming_impl(
                                &source,
                                &distorted,
                                None,
                                true,
                                toggles,
                                &mut scratch,
                                Some(compute),
                            )
                        })
                        .unwrap()
                        .into_features();
                    if w * h < PARALLEL_MIN_PIXELS {
                        assert!(
                            scratch.rev5_jobs.is_empty(),
                            "small walks use the serial route"
                        );
                        assert_eq!(
                            scratch.rev5_jobs.capacity(),
                            0,
                            "small walks retain no queue"
                        );
                    } else if budget.is_none() {
                        assert!(
                            !scratch.rev5_jobs.is_empty(),
                            "the default batch route must execute"
                        );
                    }
                    if let Some(budget) = budget {
                        let plane_bytes: usize = scratch
                            .rev5_jobs
                            .iter()
                            .map(|job| {
                                let s = &job.scratch;
                                [
                                    &s.src_wide,
                                    &s.dst_wide,
                                    &s.mu1_h,
                                    &s.mu2_h,
                                    &s.ssq_h,
                                    &s.s12_h,
                                    &s.mu1,
                                    &s.mu2,
                                    &s.ssq,
                                    &s.s12,
                                    &s.activity,
                                ]
                                .iter()
                                .map(|plane| plane.capacity() * core::mem::size_of::<f32>())
                                .sum::<usize>()
                            })
                            .sum();
                        assert!(
                            plane_bytes <= budget,
                            "retained planes {plane_bytes} exceed budget {budget}"
                        );
                        let owned = retained_queue_bytes(&scratch.rev5_jobs);
                        assert!(
                            owned <= budget,
                            "owned queue bytes {owned} exceed budget {budget}"
                        );
                        if budget == 0 {
                            assert!(
                                scratch.rev5_jobs.is_empty(),
                                "zero budget must use the old route"
                            );
                        }
                        let per_job = rev5_job_bytes(w * (STRIP_ROWS + 2 * HALO_P)).unwrap();
                        assert_eq!(
                            scratch.rev5_jobs.is_empty(),
                            // Literal, not the constant: the floor is the contract.
                            w * h < PARALLEL_MIN_PIXELS || budget / per_job < 3,
                            "{w}x{h} budget {budget}: queue above the pixel gate iff at least three jobs fit"
                        );
                    }
                    assert_eq!(expected.len(), actual.len());
                    for (i, (a, b)) in expected.iter().zip(actual.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{threads}T {w}x{h} identity={identity} scales={mask:b} feature {i}"
                        );
                    }
                }
            }
        }
    }
    println!("{SENTINEL}");
}
