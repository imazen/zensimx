# Revision selector site index (2026-10-25)

Companion to [the dependency map](REV7_DEPENDENCY_MAP_2026-10-25.md).
Source locations: `bcf26e70`; phase 1 changes no product arithmetic. Main's
later cfg-only integration is described in the dependency map. Generated locations
from `scripts/rev7/inventory.py`, with source-reviewed caller envelopes below.
Each `line: selector` is one executable occurrence or definition (including
registry matches); comments and string literals are excluded. Entries sharing
a function name are grouped, including overloaded impl methods. Envelopes
include evaluation before a refusal and explicit alternate-engine calls; they
do not assert every branch runs. The map distinguishes successful defaults
from raw research and explains the leaf call graph. Macro-generated copies
are not duplicated. Test rows identify test callers, not a product dependency.

The inventory contains **578 occurrences in 246 function/context groups**.

| File / function | Every occurrence (line: selector) | Reaching configurations / envelope |
|---|---|---|
| `attribution.rs` / `build_attribution_into_sink` | 1712: active_revision, stable_forms | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `attribution.rs` / `max_projection_matches_pixel_reference_across_sampling_and_masks` | 2331: active_revision | research-only (test helper/body) |
| `attribution.rs` / `candidate_map_skipping_matches_the_full_legacy_map` | 2795: Rev3 | research-only (test helper/body) |
| `attribution.rs` / `blockiness_map_matches_finite_repairs_through_every_pipeline` | 3924: active_revision, Rev4 | research-only (test helper/body) |
| `attribution.rs` / `planned_features` | 6073: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `attribution.rs` / `fused_basic_into_at_revision` | 6705: stable_forms | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `attribution.rs` / `attribution_from_retention_binned` | 7538: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `box_blur_v_from_copy` | 106: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `box_blur_h` | 801: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `box_blur_h_into_abs_diff` | 1655: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `fused_blur_h_ssim` | 3578: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `fused_blur_h_ssim_at_revision` | 3596: effective_revision; 3597: Rev3 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `fused_blur_h_ssim3` | 3717: effective_revision; 3718: Rev3 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `blur.rs` / `fused_blur_hv_ssim_at_revision` | 7161: effective_revision; 7178: Rev3 | R7, B5, research-only |
| `blur.rs` / `fused_moment_pass_matches_the_two_passes` | 8643: Rev5, Rev7 | research-only (test helper/body) |
| `blur.rs` / `fused_activity_pass_matches_the_two_passes` | 8749: Rev5, Rev7 | research-only (test helper/body) |
| `blur.rs` / `adjoint_probe_ssim_state` | 9462: Rev3; 9487: Rev3 | research-only (test helper/body) |
| `blur.rs` / `rev3_ssim_color_scale_prerequisite_matches_public_mse_model` | 9717: active_revision | research-only (test helper/body) |
| `blur.rs` / `ssim3_matches_ssim4_bitwise` | 10052: active_revision | research-only (test helper/body) |
| `blur.rs` / `activity_from_the_fused_h_plane_is_bit_identical` | 11031: active_revision; 11032: Rev3 | research-only (test helper/body) |
| `blur.rs` / `h_entries_are_bit_exact_at_a_degenerate_last_column_tile` | 11158: active_revision | research-only (test helper/body) |
| `color.rs` / `srgb_to_positive_xyb_planar_into_at_revision` | 193: cbrt_newton | R7, B5, BHdr(4), CD(4), research-only |
| `color.rs` / `linear_to_positive_xyb_planar_into` | 2057: cbrt_newton | R7, B5, BHdr(4), CD(4), research-only |
| `color.rs` / `linear_to_positive_xyb_planar_into_unclamped` | 2719: cbrt_newton | R7, B5, BHdr(4), CD(4), research-only |
| `color.rs` / `unclamped_matches_clamped_scalar_for_in_gamut` | 3365: Rev3; 3366: Rev4 | research-only (test helper/body) |
| `color.rs` / `opsin_leaf_follows_the_passed_revision` | 3428: Rev1; 3429: Rev2; 3430: Rev3; 3439: Rev4; 3444: Rev5; 3450: active_revision | research-only (test helper/body) |
| `color.rs` / `rev4_xyb_body_tail_bits_identical_across_tiers` | 3636: Rev4 | research-only (test helper/body) |
| `color.rs` / `unclamped_out_of_gamut_stays_finite` | 3725: Rev3; 3726: Rev4 | research-only (test helper/body) |
| `corruption_head.rs` / `from_bytes` | 485: Rev1; 486: Rev2; 487: Rev3; 488: Rev4; 489: Rev5 | A, BHdr, B5, CD/research companions; no Rev6/7 ZCTH format |
| `corruption_head.rs` / `explicit_revision_is_validated_and_hash_bound` | 1463: Rev1; 1464: Rev2; 1465: Rev3; 1466: Rev4; 1467: Rev5 | research-only (test helper/body) |
| `corruption_head.rs` / `named_profile_refuses_another_head_revision` | 2162: Rev2; 2163: Rev3 | research-only (test helper/body) |
| `derived_inputs.rs` / `derived_bake` | 259: active_revision | research-only (test helper/body) |
| `det_math.rs` / `at_revision` | 123: effective_revision; 126: active_revision; 129: for_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `det_math.rs` / `for_revision` | 134: for_revision; 136: Rev1; 137: Rev2; 138: Rev3; 139: Rev4; 140: Rev5; 141: Rev6; 142: Rev7; 335: for_revision; 337: Rev1; 338: Rev2; 339: Rev3; 340: Rev4; 341: Rev5; 342: Rev6; 343: Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `det_math.rs` / `active_root_form` | 187: for_revision, active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `det_math.rs` / `active_pow_form` | 376: for_revision, active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `det_math.rs` / `default_is_the_shipped_libm_form` | 868: for_revision, Rev1; 872: for_revision, Rev2; 877: for_revision | research-only (test helper/body) |
| `det_math.rs` / `default_pow_form_is_the_shipped_libm_form` | 1113: for_revision, Rev1; 1117: for_revision, Rev2; 1122: for_revision | research-only (test helper/body) |
| `diffmap.rs` / `compute_with_ref_and_diffmap` | 838: Rev5; 847: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `diffmap.rs` / `compute_with_ref_and_diffmap_linear_planar` | 1035: Rev5; 1063: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `featcanon.rs` / `mode` | 123: Rev4 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `featcanon.rs` / `compute_mode` | 136: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `featcanon.rs` / `canon_blur_axis_for` | 314: Rev5; 328: Rev5 | R7, B5, BHdr(4), CD(4), research-only |
| `featcanon.rs` / `mode_is_a_function_of_the_computation_revision` | 1042: Rev1; 1043: Rev2; 1044: Rev3; 1048: Rev4; 1050: active_revision | research-only (test helper/body) |
| `feature_defs.rs` / `era_tokens` | 549: era_tokens; 551: Self::Rev1; 552: Self::Rev2; 553: Self::Rev3; 563: Self::Rev4; 575: Self::Rev5; 588: Self::Rev6; 602: Self::Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `ring_high_pass` | 622: ring_high_pass; 623: Self::Rev6 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `computes_globaff` | 627: computes_globaff; 628: Self::Rev6, Self::Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `row_lanes` | 633: row_lanes; 634: Self::Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `cbrt_newton` | 640: cbrt_newton; 641: Self::Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `stable_forms` | 648: stable_forms; 649: Self::Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `paired_global_contrast` | 677: paired_global_contrast; 680: Self::Rev2, Self::Rev3, Self::Rev4, Self::Rev5, Self::Rev6, Self::Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_defs.rs` / `moved_slots` | 694: moved_slots; 695: era_tokens | none at scoring ingress; research-only/tests |
| `feature_defs.rs` / `ssim_stable_moves_exactly_the_f4_slots` | 3560: Rev3; 3561: moved_slots; 3564: Rev3, era_tokens | research-only (test helper/body) |
| `feature_defs.rs` / `rev1_moves_no_slots` | 3573: Rev1; 3574: moved_slots; 3577: Rev1, era_tokens | research-only (test helper/body) |
| `feature_defs.rs` / `tiercanon_is_a_registered_arithmetic_era_moving_every_live_slot` | 3668: Rev1; 3669: Rev2; 3670: Rev3; 3672: era_tokens; 3674: Rev4, era_tokens; 3677: Rev4, era_tokens, Rev3, era_tokens; 3678: Rev3, era_tokens; 3686: Rev4, moved_slots; 3688: Rev3, moved_slots | research-only (test helper/body) |
| `feature_defs.rs` / `localwin_is_a_registered_arithmetic_era_moving_every_live_slot` | 3713: Rev1; 3714: Rev2; 3715: Rev3; 3716: Rev4; 3718: era_tokens; 3722: Rev5, era_tokens; 3724: Rev5, era_tokens, Rev4, era_tokens; 3725: Rev4, era_tokens; 3733: Rev5, moved_slots | research-only (test helper/body) |
| `feature_defs.rs` / `rev2_carries_all_eras_and_f5_needs_the_append_block` | 3741: Rev2, era_tokens; 3746: Rev2, moved_slots; 3747: Rev2, moved_slots | research-only (test helper/body) |
| `feature_defs.rs` / `family_slots_match_compute_set_populated_slots` | 4354: active_revision | research-only (test helper/body) |
| `feature_layout.rs` / `formula_revision` | 63: Rev1; 64: Rev2; 65: Rev3; 66: Rev4; 67: Rev5; 68: Rev6; 69: Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_plan.rs` / `rev5_unsupported` | 93: Rev5; 101: computes_globaff | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_plan.rs` / `derive_with_layout` | 176: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_plan.rs` / `for_bake` | 538: computes_globaff; 546: computes_globaff; 587: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_plan.rs` / `v1` | 610: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `feature_plan.rs` / `mixed_revision_profiles_get_no_plan` | 1843: Rev1, Rev2; 1844: Rev2; 1845: Rev3; 1846: Rev4; 1847: Rev5; 1848: Rev6; 1849: Rev7, Rev1 | research-only (test helper/body) |
| `feature_plan.rs` / `planned_pixels_match_canonical_features_at_consumed_ids` | 1927: Rev5 | research-only (test helper/body) |
| `feature_v2/restore_cuts.rs` / `restore_cuts_plane_dump` | 726: active_revision; 729: Rev3 | research-only (test helper/body) |
| `feature_v2/restore_cuts.rs` / `streaming_block_max_matches_naive_lattice` | 941: Rev3; 943: Rev3; 945: Rev3 | research-only (test helper/body) |
| `feature_v2.rs` / `process_ring_high_pass` | 265: process_ring_high_pass; 267: active_revision, ring_high_pass | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `default` | 1088: active_revision, Rev4 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `finish_append` | 2232: paired_global_contrast | CD, research-only |
| `feature_v2.rs` / `meas_f64` | 2359: active_revision | CD, research-only |
| `feature_v2.rs` / `finalize_free` | 2633: paired_global_contrast | CD, research-only |
| `feature_v2.rs` / `fold_v1_one_band` | 2896: Rev5; 3018: Rev3 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `finish_channel_scale_planned` | 3855: active_revision | CD, research-only |
| `feature_v2.rs` / `compute_v2_diffmap_channel_scale` | 4081: active_revision, Rev3; 4082: active_revision, stable_forms; 4083: process_ring_high_pass | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `build_v2_ref_scales` | 4569: active_revision | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `compute_v2_features_impl_with_toggles` | 4641: Rev5 | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `compute_v2_features_with_ref_impl` | 4676: effective_revision, Rev5; 4733: effective_revision; 4734: Rev4 | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `new` | 4933: Rev5; 4934: Rev5; 8447: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `rev5_execute_job` | 5179: row_lanes; 5258: Rev5 | R7, B5, research-only; CD where predicate is evaluated before refusal |
| `feature_v2.rs` / `rev5_merge_job` | 5307: row_lanes | R7, B5, research-only; CD where predicate is evaluated before refusal |
| `feature_v2.rs` / `rev5_block_walk` | 5534: row_lanes | R7, B5, research-only; CD where predicate is evaluated before refusal |
| `feature_v2.rs` / `ensure` | 5994: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `stream_fold_v1` | 6677: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `blockiness_sparse_strip_wide` | 6746: row_lanes | R7, B5, research-only; CD where predicate is evaluated before refusal |
| `feature_v2.rs` / `validate_wide_revision` | 7354: refuse_rev4_mix; 7355: active_revision | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `validate_raw_rev5_request` | 7373: Rev5 | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `compute_folded720_streaming_extras` | 7438: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `ref_feed_admits` | 7797: Rev5 | R7, B5, research-only; CD where predicate is evaluated before refusal |
| `feature_v2.rs` / `retained_ref_feed_admits` | 7842: Rev5 | R7, B5, research-only; CD where predicate is evaluated before refusal |
| `feature_v2.rs` / `compute_folded720_hdr_streaming_extras` | 7946: Rev5; 7957: refuse_rev4_mix | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `at_revision` | 8453: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `foldapp_streaming_walk_impl` | 8724: computes_globaff; 8781: Rev5; 8859: Rev5; 8877: Rev5; 8880: Rev5; 9049: Rev5; 9131: Rev5; 9997: stable_forms | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `compute_v2_features_with_ref_impl_inner` | 10263: active_revision; 10270: active_revision; 10445: stable_forms | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `derive_v2app_coeffs` | 10914: Rev3; 10915: ring_high_pass; 10916: stable_forms | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2.rs` / `attr_pass_b_for_scale` | 12793: active_revision | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2.rs` / `compute_v2_append_attribution_impl` | 13357: stable_forms | research-only (raw feature/map entry); CD wide-map callers where admitted |
| `feature_v2_kernels.rs` / `default` | 1288: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `active_formula_revision` | 1537: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `rev5_scope` | 1665: Rev5; 1684: computes_globaff | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `free_work` | 1720: Rev5; 1727: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `run_blur_pass` | 2325: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `run_blur_pass_strip` | 2374: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `run_blur_pass_strip_cached_ref` | 2414: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `run_blur_pass_inner` | 2656: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `dense_block_kernel_generic` | 2944: active_revision, Rev3; 2946: active_revision, stable_forms | CD(1–3), research-only |
| `feature_v2_kernels.rs` / `dense_block_kernel` | 3549: active_revision | R7, B5, CD, research-only |
| `feature_v2_kernels.rs` / `dense_block_kernel_at_revision` | 3575: Rev5; 3576: row_lanes | R7, B5, CD, research-only |
| `feature_v2_kernels.rs` / `gradient_block_kernel_generic` | 4077: active_revision, stable_forms | CD(1–3), research-only |
| `feature_v2_kernels.rs` / `gradient_block_kernel` | 6206: active_revision | R7, B5, CD, research-only |
| `feature_v2_kernels.rs` / `gradient_block_kernel_at_revision` | 6234: ring_high_pass; 6334: Rev5, row_lanes; 6418: stable_forms | R7, B5, CD, research-only |
| `feature_v2_kernels.rs` / `blockiness_sparse_rows` | 6512: active_revision, row_lanes | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_kernels.rs` / `csfw_block_kernel` | 7763: effective_revision | research-only (unsupported at >=5) |
| `feature_v2_oracle.rs` / `dense_reference` | 282: active_revision; 283: Rev3 | research-only (oracle) |
| `feature_v2_oracle.rs` / `dense_block_kernel_era2_generic` | 812: active_revision, stable_forms; 813: active_revision, Rev3 | CD(1–3), research-only |
| `feature_v2_oracle.rs` / `dense_block_kernel_canon` | 1257: Rev3; 1258: Rev5; 1259: stable_forms; 1265: row_lanes | R7, B5, CD(4), research-only |
| `feature_v2_oracle.rs` / `dense_block_kernel_canon64_vec` | 1667: Rev3; 1669: stable_forms; 1737: row_lanes | R7, B5, CD(4), research-only |
| `feature_v2_oracle.rs` / `dense_block_kernel_exact` | 2314: active_revision, Rev3; 2315: active_revision, Rev5; 2316: active_revision, stable_forms | research-only (oracle) |
| `feature_v2_oracle.rs` / `the_walk_refuses_a_rev4_request_in_a_rev3_process` | 2518: Rev4; 2522: Rev3; 2523: Rev3 | research-only (test helper/body) |
| `feature_v2_oracle.rs` / `the_walk_refuses_an_earlier_request_in_a_rev4_process` | 2540: Rev1; 2541: Rev2; 2542: Rev3; 2546: Rev4; 2547: Rev4 | research-only (test helper/body) |
| `feature_v2_oracle.rs` / `dense_canon64_vec_matches_scalar_on_every_compiled_tier` | 2743: active_revision; 2806: active_revision; 2838: active_revision; 2871: active_revision; 2904: active_revision | research-only (test helper/body) |
| `feature_v2_retained_core.rs` / `retained_core_admits` | 14: Rev5 | R7, B5, research-only |
| `feature_v2_retained_core.rs` / `rev5_blur_into_retained` | 88: Rev5; 125: Rev5 | R7, B5, research-only |
| `feature_v2_retained_core.rs` / `one_thread_retention_keeps_blockiness_features` | 339: Rev5 | research-only (test helper/body) |
| `feature_v2_stream.rs` / `new_with_front_end` | 670: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_stream.rs` / `new_with_advance` | 775: active_revision | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_stream.rs` / `new_inner` | 800: Rev5 | R7, B5, CD; A/BHdr with explicit SDR Fold; research-only |
| `feature_v2_stream.rs` / `rev5_downscaled_chroma_matches_full_conversion` | 1609: Rev5; 1620: Rev5 | research-only (test helper/body) |
| `feature_v2_stream.rs` / `materialize` | 1697: active_revision | research-only (test helper/body) |
| `feature_v2_stream.rs` / `downscale_into_matches_inplace` | 1730: active_revision | research-only (test helper/body) |
| `feature_v2_stream.rs` / `linear_pyramid_producers_match_the_whole_image_form` | 1910: Rev5, Rev7 | research-only (test helper/body) |
| `feature_v2_stream.rs` / `block_producer_windows_match_the_streaming_producer` | 2032: Rev5, Rev7 | research-only (test helper/body) |
| `feature_v2_tests.rs` / `amplitude_split_folds_the_ssim_pools_into_its_mean_pool` | 42: Rev5 | research-only (test helper/body) |
| `feature_v2_tests.rs` / `rev7_blockiness_strip_partials_match_the_whole_range_form` | 82: Rev7; 100: Rev7 | research-only (test helper/body) |
| `feature_v2_tests.rs` / `tailhist_fold_is_off_below_rev4` | 2552: active_revision, Rev4 | research-only (test helper/body) |
| `feature_v2_tests.rs` / `era2_fused_and_two_pass_are_bit_identical` | 3195: active_revision; 3201: active_revision | research-only (test helper/body) |
| `feature_v2_tests.rs` / `compute_set_matches_legacy_derivation` | 3380: active_revision | research-only (test helper/body) |
| `feature_v2_tests.rs` / `bake_plan_matches_score_pool_mode_on_shipped_b` | 3549: Rev5 | research-only (test helper/body) |
| `feature_v2_tests.rs` / `era2_oracle_bounds_hold_for_every_pool_shape` | 4105: active_revision; 4111: active_revision | research-only (test helper/body) |
| `feature_v2_tests.rs` / `pool_simd_drift_within_policy` | 4354: active_revision; 4360: active_revision | research-only (test helper/body) |
| `fused.rs` / `revision` | 476: effective_revision | R7, A, BHdr, B5, CD; research-only |
| `fused.rs` / `luma_form` | 483: active_revision; 486: for_revision | R7, A, BHdr, B5, CD; research-only |
| `fused.rs` / `fused_vblur_features_ssim` | 898: Rev3 | R7, A, BHdr, B5, CD; research-only |
| `fused.rs` / `fused_vblur_features_edge` | 1145: stable_forms; 1148: effective_revision | R7, A, BHdr, B5, CD; research-only |
| `fused.rs` / `fused_vblur_ssim_canon` | 5444: stable_forms | R7, B5, BHdr(4), CD(4), research-only |
| `fused.rs` / `fused_vblur_ssim_local` | 5619: stable_forms; 5625: row_lanes | R7, B5, research-only |
| `fused.rs` / `fused_vblur_ssim_canon64_vec` | 5825: stable_forms; 5885: row_lanes | R7, B5, BHdr(4), CD(4), research-only |
| `fused.rs` / `fused_vblur_ssim_exact` | 6786: stable_forms | research-only (oracle) |
| `fused.rs` / `vblur_ssim_rev7_lean_loop_matches_every_tier` | 8162: Rev7 | research-only (test helper/body) |
| `globaff.rs` / `walk_matches_two_pass_reference_over_its_own_planes` | 555: Rev6; 561: Rev6; 567: Rev6 | research-only (test helper/body) |
| `globaff.rs` / `rev6_moves_only_ringing_and_adds_the_descriptor` | 604: Rev5; 605: Rev6; 631: Rev6 | research-only (test helper/body) |
| `globaff.rs` / `earlier_revisions_refuse_the_family` | 643: Rev5 | research-only (test helper/body) |
| `gmsbank_calibration_instrument.rs` / `gmsbank_calibration_scale1_y_dump` | 44: active_revision | research-only (test helper/body) |
| `gmsbank_calibration_instrument.rs` / `gmsbank_chroma_calibration_dump` | 163: active_revision | research-only (test helper/body) |
| `hf_gain_form.rs` / `at_revision` | 142: effective_revision; 145: active_revision; 148: for_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `hf_gain_form.rs` / `for_revision` | 153: for_revision; 155: Rev1; 156: Rev2; 157: Rev3; 158: Rev4; 159: Rev5; 160: Rev6; 161: Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `hf_gain_form.rs` / `active_gain_form` | 313: for_revision, active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `hf_gain_form.rs` / `revision_selection_goes_through_the_named_arm` | 782: for_revision, Rev1; 786: for_revision, Rev2 | research-only (test helper/body) |
| `local_refine.rs` / `capture_from_parts` | 524: Rev5 | R7, B5, research-only; older unsupported plans may evaluate admission guard |
| `local_refine.rs` / `deltas_range` | 782: effective_revision; 783: Rev3; 815: Rev5; 878: stable_forms; 884: stable_forms | R7, B5, research-only; older unsupported plans may evaluate admission guard |
| `local_refine.rs` / `recompute_planes` | 1053: Rev5; 1086: Rev5 | R7, B5, research-only; older unsupported plans may evaluate admission guard |
| `local_refine.rs` / `grad_delta` | 1420: ring_high_pass | R7, B5, research-only; older unsupported plans may evaluate admission guard |
| `local_refine.rs` / `prepared_scale0_distorted_planes_match_reconversion_bits` | 1658: Rev5 | research-only (test helper/body) |
| `local_refine.rs` / `take_pair` | 1768: Rev5 | research-only (test helper/body) |
| `local_refine.rs` / `walk` | 1999: active_revision, Rev5 | research-only (test helper/body) |
| `local_refine.rs` / `full_plan` | 2032: active_revision, Rev5 | research-only (test helper/body) |
| `local_refine.rs` / `check_pair` | 2131: active_revision, Rev5 | research-only (test helper/body) |
| `local_refine.rs` / `cone_case` | 2447: effective_revision; 2448: Rev3 | research-only (test helper/body) |
| `local_refine.rs` / `coarse_y_only_plan_coarse_channels_rebuilt` | 2745: active_revision, Rev5 | research-only (test helper/body) |
| `metric/bake.rs` / `prepare_steering_input` | 256: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric/bake.rs` / `check_pixel_revision` | 938: Rev5; 939: refuse_rev4_mix; 951: effective_revision; 959: for_revision; 960: active_revision; 971: Rev4; 972: Rev4; 982: active_revision; 984: for_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric/bake.rs` / `compute` | 1088: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric/bake.rs` / `precompute_reference` | 1285: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric/bake.rs` / `compute_attribution_input` | 1409: Rev5; 1527: Rev5; 1545: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric/bake.rs` / `bake_bytes_over_sampling` | 1998: active_revision | research-only (test helper/body) |
| `metric/bake.rs` / `revisions_two_and_three_select_the_same_luminance_form` | 3959: for_revision, Rev2; 3960: for_revision, Rev3; 3965: for_revision, Rev3 | research-only (test helper/body) |
| `metric/bake.rs` / `rev3_process_refuses_a_rev2_bake_despite_the_shared_clamp_form` | 3987: active_revision, Rev3 | research-only (test helper/body) |
| `metric/bake.rs` / `the_shipped_process_refuses_a_rev3_bake` | 4250: active_revision | research-only (test helper/body) |
| `metric/bake.rs` / `the_diagnostic_bypass_is_inert_without_its_environment_switch` | 4281: active_revision, Rev3 | research-only (test helper/body) |
| `metric/bake.rs` / `the_diagnostic_bypass_serves_a_mismatched_bake_and_says_so` | 4326: active_revision, Rev3 | research-only (test helper/body) |
| `metric/bake.rs` / `prepared_workers_honor_model_revision_and_local_subset` | 4418: active_revision | research-only (test helper/body) |
| `metric/bake.rs` / `undeclared_is_the_shipped_era_and_an_unknown_revision_is_refused` | 4847: Rev4; 4853: Rev5; 4859: Rev6; 4865: Rev7 | research-only (test helper/body) |
| `metric/steer_exact.rs` / `admit` | 32: Rev5 | B5 research policy; R7/A/BHdr/CD attempts refuse |
| `metric.rs` / `precompute_reference_with_scales` | 760: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `refuse_rev5_masked_config` | 766: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_zensim_with_ref_and_config` | 797: check_route | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `scoring_plan` | 1562: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `engine` | 1573: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_extended_features` | 1838: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_v2_features_with_toggles` | 1948: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_v2_features_with_ref_and_scratch` | 2063: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_folded720_features_streaming` | 2163: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_folded720_append_features_streaming` | 2194: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_folded720_features_hdr` | 2231: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_folded720_append_features_hdr` | 2260: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_folded720_append2_features_hdr` | 2314: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_folded720_csfw_features_hdr` | 2369: refuse_rev4_mix | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compare_against_ref_into` | 2435: Rev5; 2447: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_streaming_strips` | 2622: Rev5; 2625: refuse_rev4_strips | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_with_ref_streaming_strips` | 2727: Rev5; 2730: refuse_rev4_strips | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_with_ref_into` | 2782: active_revision, Rev5; 2802: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_pu_linear_extended_features` | 3056: Rev5; 3061: active_revision, Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `pu_fold_route` | 3206: active_revision, Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_all_features` | 3324: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_with_params` | 3404: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_with_config_core` | 4038: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `check_profile_route` | 4170: Rev5; 4175: check_route | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `config_from_params` | 4184: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `metric.rs` / `compute_zensim_with_config` | 5571: check_route | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `research.rs` / `everything` | 183: active_revision, computes_globaff | research-only (R7 and older requests) |
| `research.rs` / `plan` | 371: active_revision, computes_globaff | research-only (R7 and older requests) |
| `research.rs` / `manifest_json` | 632: active_revision; 643: process_ring_high_pass | research-only (R7 and older requests) |
| `research.rs` / `active_era_tokens` | 821: active_revision, era_tokens | research-only (R7 and older requests) |
| `research.rs` / `everything_covers_the_whole_registry` | 1238: active_revision, computes_globaff | research-only (test helper/body) |
| `research.rs` / `the_request_selector_follows_the_arithmetic_selector` | 1334: active_revision; 1348: Rev1 | research-only (test helper/body) |
| `research.rs` / `the_refusal_names_exactly_the_registrys_moved_slots` | 1380: active_revision, Rev1; 1387: active_revision | research-only (test helper/body) |
| `rev5_scaling_tests.rs` / `ordered_batches_match_serial_with_reused_strided_scratch` | 129: Rev5 | research-only (test helper/body) |
| `sampling.rs` / `resize` | 135: Rev5 | R7 sampled bake; research-only / caller-supplied sampled older bake |
| `sampling.rs` / `direct_scales_match_quantized_separable_f64_tap_algebra` | 371: active_revision | research-only (test helper/body) |
| `sampling.rs` / `parallel_pyramids_match_serial_bits` | 471: active_revision; 475: active_revision; 492: active_revision; 494: active_revision | research-only (test helper/body) |
| `source_rows_tests.rs` / `<item>` | 13: Rev1; 14: Rev2; 15: Rev3; 16: Rev4; 17: Rev5 | research-only (test helper/body) |
| `source_rows_tests.rs` / `padded_rows_subset_and_strip_producer` | 296: Rev3, Rev5 | research-only (test helper/body) |
| `ssim_form.rs` / `for_revision` | 925: for_revision; 927: Rev1; 928: Rev2; 929: Rev3; 930: Rev4; 931: Rev5; 932: Rev6; 933: Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `<item>` | 1000: Rev1 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `active_revision` | 1017: active_revision; 1021: Rev1; 1022: Rev2; 1023: Rev3; 1024: Rev4; 1025: Rev5; 1026: Rev6; 1027: Rev7 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `effective_revision` | 1034: effective_revision; 1037: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `check_route` | 1189: check_route; 1190: refuse_rev4_mix; 1191: Rev3 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `refuse_rev4_mix` | 1215: refuse_rev4_mix; 1216: active_revision; 1218: Rev4, Rev4 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `refuse_rev4_strips` | 1234: refuse_rev4_strips; 1235: effective_revision, Rev4 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `warn_pinned_revision_scoring_once` | 1256: active_revision; 1268: active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `warn_cross_revision_once` | 1300: active_revision | research-only (compiled diagnostic) |
| `ssim_form.rs` / `active_luma_form` | 1326: for_revision, active_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `ssim_form.rs` / `probe_rev3_ssim_partials` | 1695: for_revision, Rev3 | research-only (oracle) |
| `ssim_form.rs` / `selecting_the_shipped_revision_is_a_no_op` | 2296: for_revision; 2297: for_revision, active_revision | research-only (test helper/body) |
| `streaming.rs` / `active_channels` | 971: effective_revision | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `streaming.rs` / `convert_source_to_xyb_into_slices_chunked` | 1582: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `streaming.rs` / `process_strip_channel` | 2021: effective_revision; 2025: Rev3; 2295: stable_forms | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `streaming.rs` / `rev5_retained_mean_offset_does_not_read_pixels` | 4831: Rev5 | research-only (test helper/body) |
| `streaming.rs` / `compute_xyb_mean_offset_at_revision` | 4852: Rev5 | R7, A, BHdr, B5, CD; research-only (including refused routes) |
| `streaming.rs` / `convert` | 5703: Rev5 | research-only (test helper/body) |
| `streaming.rs` / `convert_chunk_rows_is_semantics_not_a_knob` | 5759: active_revision, Rev3; 5811: Rev4 | research-only (test helper/body) |
| `streaming.rs` / `locality_fixture_reproduces_out_of_support_movement_on_the_shipped_revision` | 8047: active_revision, Rev1 | research-only (test helper/body) |
| `streaming.rs` / `rev3_retained_signal_is_local_under_a_reference_replacement` | 8081: active_revision; 8082: Rev3 | research-only (test helper/body) |
| `streaming.rs` / `rev3_moves_exactly_the_registered_slots` | 8462: active_revision; 8511: Rev3, moved_slots | research-only (test helper/body) |
| `streaming.rs` / `rev3_moves_exactly_the_registered_slots_on_the_944_layout` | 8598: active_revision; 8629: Rev3, moved_slots | research-only (test helper/body) |
| `streaming.rs` / `dump_ssim_precision_from_coherence` | 8811: active_revision; 8813: Rev1; 8814: Rev2; 8815: Rev3; 8816: Rev4; 8817: Rev5; 8818: Rev6; 8819: Rev7; 8831: for_revision | research-only (test helper/body) |
