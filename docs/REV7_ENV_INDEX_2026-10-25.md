# ZENSIM environment read index (2026-10-25)

Companion to [the dependency map](REV7_DEPENDENCY_MAP_2026-10-25.md).
Library locations are pinned to `bcf26e70`; later cfg-only main integration
is described in the dependency map. Validation
locations include the phase-1 timing harness extension. Every literal read and
the two constant-name child-sentinel reads are listed. The source inventory
contains 100 read sites across 67 variable names; defaults below are from code,
not the environment used on this machine. All product switches were unset for
the oracle captures. A child/test site does not create a product requirement.

| Variable | Every read site | Selection and default | Kept-configuration need |
|---|---|---|---|
| `ZENSIM_ALLOW_MISSING_PAIRS` | `validate/main.rs:1751` | TID loader allows incomplete input only for `1`; default refuses missing pairs. | validation admission only; not needed by kept inference |
| `ZENSIM_ATTRIBUTION_REVISION_CHILD` | `src/metric/bake.rs:4689` | Child-process test sentinel: absent runs parent/spawns child; explicit marker selects isolated body. | test-only |
| `ZENSIM_ATTR_PERF` | `src/attribution.rs:6457`; `src/attribution.rs:7089`; `src/attribution.rs:7342`; `src/feature_v2.rs:13089` | `1` enables map pass timing output, default off; multiple map owners | none |
| `ZENSIM_BYPASS_CONTAMINATION_GUARD_FOR_AUDIT_I_REALLY_MEAN_IT` | `validate/contamination_guard.rs:87` | Exact `1` explicit audit bypass; default contamination checks enabled. | validation admission only; not needed by kept inference |
| `ZENSIM_CORRUPTION_GRID` | `validate/bin/bake_verdict.rs:1039`; `validate/bin/bake_verdict.rs:1090` | Verdict corruption-grid path; default committed constant in `bake_verdict` lines1039–1045; explicit flag wins. | validation input only |
| `ZENSIM_CROSS_REVISION_DIAGNOSTIC` | `src/metric/bake.rs:4276`; `src/metric/bake.rs:4306`; `src/ssim_form.rs:1284` | compiled only with feature of same name; exact `1` bypasses serving revision agreement with warning | none; not a serving contract |
| `ZENSIM_DECAY_DEBUG` | `validate/mlp_train/mod.rs:1295` | Exact `1` enables weight-decay debug print; default off. | training diagnostic only |
| `ZENSIM_DIAG_SPLIT` | `src/metric/bake.rs:5211` | Ignored map split diagnostic: exact only for `exact`, otherwise repair. | test-only |
| `ZENSIM_DIAL_GRID` | `validate/bin/bake_verdict.rs:1073`; `validate/bin/bake_verdict.rs:1089` | Verdict dial-grid path; default CANONICAL_DIAL_GRID in `bake_verdict.rs:176` (quarantined-v2 table); explicit flag wins. | validation input only |
| `ZENSIM_DIAL_PRED_OUT` | `validate/bin/bake_verdict.rs:2843` | Optional prediction TSV destination; absent no TSV. | validation output only |
| `ZENSIM_DIFFMAP_UPSAMPLE` | `src/streaming.rs:400` | custom-profiles: nearest default; bilinear/catmull-rom/bspline/lanczos3 with optional `-mass`; invalid panic | default proxy-map output retained; alternate policies research |
| `ZENSIM_DUMP_IMG` | `src/streaming.rs:7651` | Comma-separated image paths, required in ignored image diagnostic unless another mode selected; no default. | test-only |
| `ZENSIM_ERA2_DENSE` | `src/feature_v2_kernels.rs:3429` | era2 enabled unless `0`; only noncanonical dense dispatch consumes it | CD/research may reach default; A/R7/B5/BHdr do not need selector |
| `ZENSIM_EVAL_ANNOTATIONS` | `validate/bin/freeze_check.rs:2315` | Optional annotation path; default benchmarks/eval_annotations.json. | validation metadata only |
| `ZENSIM_FEATCANON` | `src/featcanon.rs:154`; `src/featcanon.rs:1038`; `validate/bin/tier_audit_features.rs:198`; `validate/bin/tier_audit_features.rs:281`; `validate/bin/tier_audit_features.rs:302` | `featcanon::override_cell`, oracle only: `exact,c32,c64,neum,off`; absent/unknown no override | none; exact ruler should remain test/oracle-only |
| `ZENSIM_FEATCANON_BLUR` | `src/featcanon.rs:172`; `validate/bin/tier_audit_features.rs:310` | oracle only `rec,rec64,local,fresh`; default exact→Fresh, >=5→Local, c64 Rev4→Rec64 | none; diagnostic axis |
| `ZENSIM_FEATCANON_DUMP` | `validate/bin/tier_audit_features.rs:301` | Optional tier-audit dump directory; absent no raw dumps. | validation ruler output only |
| `ZENSIM_FEATCANON_MOMENT_CHECK` | `src/feature_v2_kernels.rs:3450` | oracle only, `1` enables independent moment assertion, otherwise off | none |
| `ZENSIM_FOLD_CENSUS_ASSERT` | `src/fold_timing.rs:225` | presence asserts work census when telemetry dumps, default absent | none |
| `ZENSIM_FOLD_TIMING` | `src/fold_timing.rs:147` | positive parsed interval enables fold telemetry, default 0 | none |
| `ZENSIM_FORMULA_REV` | `src/metric/mapperf_audit.rs:116`; `src/metric/steerfix_packet.rs:135`; `src/metric/steerfix_precision.rs:368`; `src/ssim_form.rs:1020`; `src/ssim_form.rs:1100`; `src/ssim_form.rs:1133`; `src/streaming.rs:8453`; `src/streaming.rs:8589`; `validate/bin/tier_audit_features.rs:329` | `ssim_form::active_revision`: exact strings1–7, unset/other→Rev1, OnceLock | none for A/R7/B5/BHdr1; needed for existing raw research and BHdr2–4 replays |
| `ZENSIM_FREE_DIAG` | `src/feature_v2_tests.rs:5236` | `1` prints free-extras deltas; default off. | test-only |
| `ZENSIM_FREE_TOL` | `src/feature_v2_tests.rs:5169` | Existing free-extras parity tolerance, parsed f64 default 2e-5. Not changed or set by this lane. | test-only |
| `ZENSIM_GADDR_LADDER_TRACE` | `validate/dial_addressability.rs:1026` | Optional detailed G-ADDR trace destination; absent no trace. | validation output only |
| `ZENSIM_HF_GAIN` | `src/hf_gain_form.rs:307` | `active_gain_form`: `ratio`, `bexcess`, `log1p`, `satexcess`, `cap`; default revision mapping | none; explicit research arms |
| `ZENSIM_H_TILE` | `src/blur.rs:3440` | `blur::h_blur_tile_width`, parsed usize or 1024 | legacy default bits depend on the tiling; remove override only with default fixed at same value |
| `ZENSIM_IDENTITY_BAKE` | `src/metric/bake.rs:1807` | Required external bake path for ignored identity forward diagnostic; no default. | test-only |
| `ZENSIM_IDENTITY_RGB8_HEIGHT` | `src/metric/bake.rs:1799` | Required parsed fixture height for ignored diagnostic; no default. | test-only |
| `ZENSIM_IDENTITY_RGB8_PATH` | `src/metric/bake.rs:1803` | Required RGB8 fixture file for ignored diagnostic; no default. | test-only |
| `ZENSIM_IDENTITY_RGB8_WIDTH` | `src/metric/bake.rs:1795` | Required parsed fixture width for ignored diagnostic; no default. | test-only |
| `ZENSIM_IW_REF_DIR` | `src/iw_pool.rs:1063`; `src/streaming.rs:7525` | Required reference-image directory for ignored IW diagnostic; no default. | test-only |
| `ZENSIM_JBU_GUIDE_STATS` | `src/streaming.rs:4718` | `1` prints guide statistics only when a guide exists; default off | none |
| `ZENSIM_L2_PROBE_BIN` | `src/attribution.rs:3630` | Ignored map-cost test bin size; parse usize, default 8. | test-only |
| `ZENSIM_MAP_GUIDE` | `src/map_paint.rs:206` | `sq` default / `abs` | none; `with_map_guide` supplies explicit API |
| `ZENSIM_MAP_PAINT` | `src/map_paint.rs:202` | `nearest` default / `guided` | same |
| `ZENSIM_MAP_SMOOTH` | `src/map_paint.rs:205` | sigma parsed 0..16, default 0, invalid panic | none; `with_map_smooth` supplies explicit API |
| `ZENSIM_MAP_SPLIT` | `src/map_paint.rs:203` | defaults to `MapPaint::served().split` = **exact**; accepts repair/exact/amplitude | keep exact default for bake maps; older fused APIs deliberately use removal default separately |
| `ZENSIM_MAP_WINDOW` | `src/map_paint.rs:201` | `MapPaint::from_env`: `box` default, `center`, `guided` | default map geometry must persist; selector unnecessary |
| `ZENSIM_MAX_PROJECTION_MODELS` | `src/attribution.rs:2197` | Comma-separated model files; absent uses generated deterministic linear fixture. | test-only |
| `ZENSIM_MAX_TIER` | `validate/tier_cap.rs:44` | Absent no cap; only accepted value `v3` disables v4/v4x, invalid panics. | reaches zensim dispatch policy; no override needed for kept inference |
| `ZENSIM_NARROW_REVISION_CHILD` | `src/metric/bake.rs:4514` | Child-process test sentinel: absent runs parent/spawns child; explicit marker selects isolated body. | test-only |
| `ZENSIM_NEIGHBOUR_EXACT` | `src/attribution.rs:4757`; `src/local_refine.rs:3008`; `src/metric/bake.rs:1524`; `src/metric/steerfix_packet.rs:144` | unset→on for revision>=5; `1`→on, any other explicit value→off | preserve default-on R7/B5 local refinement when removing switch |
| `ZENSIM_NEIGHSTEER_GATE_CHILD` | `src/attribution.rs:4729` | Child-process test sentinel: absent runs parent/spawns child; explicit marker selects isolated body. | test-only |
| `ZENSIM_PERF_TRACE` | `validate/perf_trace.rs:32` | `1` or case-insensitive true enables telemetry; default off. | validation telemetry only |
| `ZENSIM_PLAN_AUDIT_BAKES` | `src/metric/bake.rs:1839`; `src/metric/bake.rs:1878` | Required colon-separated bake paths for ignored audit; no default. | test-only |
| `ZENSIM_PLAN_AUDIT_HEAD` | `src/metric/bake.rs:1859` | Required corruption-head path for ignored audit; no default. | test-only |
| `ZENSIM_PLAN_AUDIT_IDS_JSON` | `src/metric/bake.rs:1831`; `src/metric/bake.rs:1866` | Required candidate-ID JSON for ignored plan audit; no default. | test-only |
| `ZENSIM_PLAN_AUDIT_REGIME` | `src/metric/bake.rs:1864` | `by-v2fy-420` selects explicit candidate-ID regime, otherwise ordinary dense audit branch. | test-only |
| `ZENSIM_PLAN_CENSUS_TSV` | `src/feature_plan.rs:1876` | Optional serve census TSV; absent uses shipped roster; a supplied TSV must add >=400 served bakes. | test-only |
| `ZENSIM_POW_FORM` | `src/det_math.rs:373` | `active_pow_form`: `libm` or `pure`; default revision mapping (Rev1 libm, >=2 pure Rust) | none; affects scoring and generic post transforms, so preserve defaults while removing selector |
| `ZENSIM_PYRAMID` | `src/feature_v2_stream.rs:55`; `src/feature_v2_stream.rs:1844` | `feature_v2_stream::linear_pyramid`: XYB default (`xyb`) / `linear`; invalid panic; materialized builder refuses linear | none; research changes coarse pixels |
| `ZENSIM_REV4_CORPUS_ROOT` | `src/feature_v2_tests.rs:2769` | Required explicit corpus root for ignored ladder test; no default. | test-only |
| `ZENSIM_REV4_DIAG` | `src/feature_v2.rs:1029` | presence enables extra Rev4 tail-histogram bins, default off | none |
| `ZENSIM_RING_FORM` | `src/feature_v2.rs:254`; `validate/bin/byc_score_timing.rs:322` | `active_ring_form`: raw default / `hpm`; invalid values panic; Rev6 independently selects HPM | none; R7 raw is required, HPM is research-only |
| `ZENSIM_ROOT_FORM` | `src/det_math.rs:184` | `det_math::active_root_form`: `libm` or `sqrt`; default revision mapping (Rev1 libm, >=2 nested sqrt) | none; A needs the default libm behavior preserved |
| `ZENSIM_SAMPLE_DIGEST` | `validate/mlp_train/mod.rs:3414`; `validate/mlp_train/mod.rs:4830`; `validate/mlp_train/mod.rs:5709`; `validate/mlp_train/mod.rs:9267` | Exact `1` enables sampler digest in four training paths; default off. | training diagnostic only |
| `ZENSIM_SIGMA_MSE` | `validate/parquet_loader.rs:909` | Exact `1` loads metric-disagreement sigma columns; default off. | training input policy only |
| `ZENSIM_SSIM_KERNEL_PERF` | `src/streaming.rs:7643` | Optional ignored SSIM diagnostic JSON path; absent falls through to precision/image mode. | test-only |
| `ZENSIM_SSIM_LUMA` | `src/ssim_form.rs:1321` | `active_luma_form`: `ssim2`, `c1`, `lorentz`, `clamp`; default `for_revision` | none; some per-request legacy gates consult it only when revision equals process |
| `ZENSIM_SSIM_PRECISION_PROBE` | `src/streaming.rs:7647` | Optional ignored precision JSON path; absent requires ZENSIM_DUMP_IMG. | test-only |
| `ZENSIM_STEERING_CONTRACT_CHILD` | `src/metric/bake.rs:4348` | Child-process test sentinel: absent runs parent/spawns child; explicit marker selects isolated body. | test-only |
| `ZENSIM_STEER_EXACT` | `src/metric/steer_exact.rs:8`; `validate/bin/steer_exact_smoke.rs:34` | absent/`off` false, `all-blocks-v1` true; other values refuse; Rev5 isolated RGB8/bin1 only | none for R7; research diagnostic |
| `ZENSIM_THREADS` | `validate/parallel.rs:30`; `validate/parallel.rs:75` | Positive integer thread count when RAYON_NUM_THREADS absent; default available_parallelism minus4, minimum1. | reaches thread policy; no arithmetic selector |
| `ZENSIM_TID_IWSSIM_CSV` | `validate/bin/zensim_mlp_train.rs:5916` | Explicit CSV path required by ignored real-loader equivalence test; no default. | test-only |
| `ZENSIM_TIER_TOGGLER_CHILD` | `src/ssim_form.rs:1059` | Child-process test sentinel: absent runs parent/spawns child; explicit marker selects isolated body. | test-only |
| `ZENSIM_TRAIN_CACHE` | `validate/train_cache.rs:58` | Trainer cache directory; default HOME/.cache/zensim-train (source HOME fallback /tmp when HOME absent). Not used by this lane. | training I/O only |
| `ZENSIM_XYB_TIER_CHILD` | `src/color.rs:3609` | Child-process test sentinel: absent runs parent/spawns child; explicit marker selects isolated body. | test-only |

One dynamic read exists outside the literal table: `bake_verdict.rs:4884`
`std::env::vars()` captures all ZENSIM-prefixed keys except ZENSIM_M3*,
ZENSIM_FULLEVAL* and ZENSIM_EVAL_STAGE into a reproducibility manifest.
It records selectors; it does not add a numerical selector or default.

The scanner deliberately does not treat documentation strings, JSON/HTML text
or environment writes as reads. In particular, ZENSIM_DUMP_IMG2 appears only
in an old comment and is not a read site. Main-map tables distinguish defaults
from explicit old-revision replay and research-only arithmetic overrides.
