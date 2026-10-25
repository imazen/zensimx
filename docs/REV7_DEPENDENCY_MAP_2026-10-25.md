# Revision 7 dependency map (2026-10-25)

Source audited: `rev7-unify-base`, `bcf26e70` (revision-7 merge `6d146e15`).
Phase 1 changes only tooling and shared test fixtures; `zensim/src`, model
bytes, formulas, supported API and existing test expectations are unchanged.
The controlling decisions are [D1–D3 in the plan](REV7_UNIFICATION_PLAN_2026-10-25.md#owner-decisions-my-recommendation-first).

After capturing that baseline, the coordinator requested integration with main.
The lane rebased onto `ead38550` (main's revision-7 integration includes
`e223fad3`). Product-source differences from the audit baseline are cfg/allow
attributes on five map/globaff items, not arithmetic. The validate integration
test now recognizes the revision-7 partial producer; the fetched main also
contains dataset-tooling/docs work, retained unchanged. Site-index locations
remain explicitly pinned to `bcf26e70`; the rebase did not change their selector
inventory or the function-body counts in this map.

## Limits before deletion

- This is a source call-graph audit plus runnable bit oracles, not exhaustive
  execution coverage. Arbitrary custom profiles, sampling, stop tokens,
  corruption companions and research toggles create more paths than the five
  captured configurations. A table entry means the selector can be evaluated;
  it does not mean every arm runs. Conditional reach is explicit below.
- Keeping A does **not** permit deleting the buffered walk. Keeping BHdr
  additionally preserves the PU buffered front end. Retaining raw research
  revision selection preserves old arithmetic even after D1 and D3.
- `feature_v2_oracle.rs` is included unconditionally by `feature_v2.rs` and
  contains live production dense kernels, including revision 7's scalar,
  NEON, wasm and x86 implementations. Its filename is not a compilation gate.
- D3 does not mention `LegacyLinearV0_2`, `Custom`, the `training` functions,
  raw v2 APIs or corruption-head format revisions. Removing their dependencies
  requires preserving/moving those callers or an additional API decision.
- No aarch64, wasm or i686 execution was performed here. Their fallback
  bodies are traced in source; x86 tier selection is checked at runtime.
  No replacement model or HDR qualification is claimed.

## Configuration keys and revision ownership

`R7` means any valid revision-7 bake; `A` means named A at its legacy revision
1; `BHdr` includes the existing revision-1 HDR serving route and explicit
revision-2–4 research replays; `B5` means today's revision-5 B; `CD` groups C,
CHdr, D and the previews. `research-only` includes explicit older-revision
requests, measurement switches and raw feature APIs, even if invoked on a
`Zensim` value whose profile is A or B. `none` means no ordinary scoring
caller, not necessarily no unit-test caller.

`ssim_form::SHIPPED_REVISION` is **Rev1**, while the default **model B declares
Rev5**. `config_from_params` copies bake revisions >=5 into the request; it
leaves earlier named profiles process-bound. `BakeScorer::plan` binds all
active member/companion reads and checks revision agreement.
`check_pixel_revision` permits model-bound >=4 plans (Rev4 also checks process
agreement); basic/peak plans have a narrower explicit-revision exemption.
Older wide bakes require process/luma agreement. These are different contracts.

| Public profile / aliases | SDR walk, arithmetic | HDR / PU walk | Embedded weights |
|---|---|---|---|
| A (`deprecated-profiles`) | buffered Rev1; optional explicit Fold engine remains public | buffered PU with A's same weights | `v47_strict_qat_native_byid_2026-09-06.bin` |
| PreviewV0_1 | buffered Rev1, weight-derived channels | same profile, PU buffered | `WEIGHTS_PREVIEW_V0_1` |
| PreviewV0_2 | buffered Rev1, weight-derived channels | same profile, PU buffered | `WEIGHTS_PREVIEW_V0_2` |
| B; `Default`, `codec_target()`, `latest_preview()`, deprecated `latest()` | planned fold Rev5, regardless of default Buffered selector | routes to BHdr's buffered PU weights | `b_sdr_diff_frag_820_rev5_2026-10-19.bin` |
| BHdr | buffered Rev1 (SDR callable, outside HDR model domain) | buffered PU; pinned 2–4 replay uses those kernels | `bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin` |
| C (`candidate-profiles`) | ID-planned wide fold Rev1 | routes to CHdr's PU fold | `c_sdr_purity944_byid_2026-09-07.bin` |
| CHdr (`candidate-profiles`) | SDR path uses `params()` and CHdr weights; **does not** swap to C | planned PU fold Rev1 | `c_hdr_l1t1944_byid_2026-09-07.bin` |
| D (`candidate-profiles`) | fold Rev1 by constructor; basic reads / Peaks scratch policy | narrow PU buffered with D weights | `d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin` |
| LegacyLinearV0_2 (`training` or tests) | buffered Rev1, optional fold | same weights, PU buffered | `WEIGHTS_PREVIEW_V0_2` via separate params |
| Custom (`custom-profiles`) | params/bake-derived, buffered or fold subject to validation | supplied params, plan-dependent PU route | caller's static params / bake functions |
| R7 via `BakeScorer` | planned fold Rev7 | planned PU fold Rev7; no qualified R7 HDR named model | caller's bake; no embedded R7 model |

The profile routes above come from executable `profile.rs::params`,
`params_pu_linear`, `metric.rs::config_from_params`, `scoring_plan`,
`compute_with_config_core` and `pu_fold_route`. In particular, CHdr's SDR
route differs from its profile documentation's stated C routing; source and
`chdr_pu::c_routes_to_chdr_through_every_pu_entry_bit_identically` are the
authority for this audit. No existing documentation was rewritten.

## Public entry inventory

Leaf keys below refer to the next table. Named calls use the table above's
weights, including B→BHdr and C→CHdr at PU ingress. Raw extraction has no bake
forward; caller-gradient maps do not embed a replacement scoring model.

| Entries (names are explicit, grouped only when they share an owner) | Walk / revision / leaves / weights |
|---|---|
| `Zensim::compute`, `compute_with_codec_hint` | `compute_with_config_inner/core` → F for B5/R7; L for A/BHdr/previews; wide CD requires F. Stop token forces older requests to L; >=5 uses stoppable F. Named scoring composition follows extraction. Descriptor HDR dispatches to `compute_pu_linear`. |
| `precompute_reference`, `compute_with_ref`, `compute_with_ref_into` | `compare_against_ref_into` chooses F for >=5 or opt-in compatible fold, otherwise cached L. Same profile/bake; cache revision checked. `precompute_reference` runs conversion/downscale/moments, not a model forward. |
| `compute_extended_features`, `compute_pu_linear_extended_features` | force masked family, L below5; refuse >=5 before extraction. Raw feature result, no final named MLP rescore. |
| `compute_all_features` (`training`) | L below5; F >=5 with full basic+peaks+v2 plan, fixed width720. It does **not** request globaff at R7; use `research::extract` or the full R7 stub for that. No final MLP forward. |
| `classify` | calls `compute`, then pixel delta classification; named score preserved. |
| `compute_with_params` | explicit caller params, derives >=5 serving plan then common core; applies supplied model/scoring metadata. |
| `compute_streaming_strips`, `compute_streaming_strips_default`, `compute_with_ref_streaming_strips`, `compute_with_ref_streaming_strips_default` | >=5 routes to ordinary canonical F entries. Rev4 is refused by `refuse_rev4_strips`; <4 uses buffered 256-row-strip accumulator merges (L), same named weights. |
| `precompute_reference_linear_planar`, `compute_pu_linear`, `compute_pu_linear_planar` | H conversion; >=5 or compatible wide CD → PU F; otherwise PU L. `params_pu_linear` selects HDR twin. Interleaved/planar stride is honored. |
| `compute_v2_features`, `compute_v2_features_with_toggles`, `prepare_v2_reference`, `prepare_v2_reference_with_moments`, `compute_v2_features_with_ref`, `compute_v2_features_with_ref_and_scratch` | raw materialized v2 walk / cached planes; explicit toggles or process revision, no profile model. `compute_v2_features_with_ref_impl` routes >=5 to F and refuses Rev4's noncanonical buffered bounded walk. Older paths reach L + D/G and research append families. |
| `compute_folded720_features`, `compute_folded720_append_features`, `compute_folded720_append2_features`, `compute_folded720_csfw_features` | F through `compute_folded720_streaming_extras` / raw wrappers; process/toggle revision, no profile model; >=5 refuses unsupported append/append2/CSFW requests rather than computing them. |
| `compute_folded720_features_streaming`, `compute_folded720_append_features_streaming` | F, explicit toggle/process agreement; same restrictions, no profile model. |
| `compute_folded720_features_hdr`, `compute_folded720_append_features_hdr`, `compute_folded720_append2_features_hdr`, `compute_folded720_csfw_features_hdr` | H + PU F; encoding/display contract explicit, toggle/process agreement required; same family restrictions, no profile model. |
| `compute_v2_diffmap` | materialized v2 map terms, process revision and caller `s_v2`, D/G formulas plus map leaves; no learned model forward. |
| `compute_with_diffmap`, `compute_with_ref_and_diffmap`, `compute_with_ref_and_diffmap_linear_planar` | L proxy planes/map. At >=5 scalar result is replaced by `compute_with_ref`'s canonical F score. This is two walks, not an R7-only map implementation. Planar variant retains its linear input contract. |
| `compute_attribution_density`, `compute_attribution_density_with_ref`, `compute_attribution_density_binned`, `compute_attribution_density_with_ref_binned`, `compute_attribution_density_full`, `compute_attribution_density_full_binned` | buffered attribution planes + caller gradient; full variants add v2/append pass. Process-selected math for legacy entry; no embedded learned forward. |
| `compute_with_ref_score_and_attribution`, `compute_with_ref_score_and_attribution_binned`, `compute_with_ref_score_and_attribution_stale`, `compute_with_ref_score_and_attribution_stale_binned` | `Fused944Session` caches/retains F features, scores named profile and runs pass B; stale forms deliberately reuse sensitivities. Basic pass has revision-aware forms; v2/append pass derives coefficients from retained plan. |
| `compute_folded944_score_and_attribution`, `compute_folded944_score_and_attribution_binned` | raw wide F retention + named profile forward + map pass B; research surface, does not make unsupported R7 append families computable. |
| `BakeScorer::compute`, `precompute_reference`, `compute_with_ref` | validated model plan; >=5 F, older narrow L / older wide F / sampled F. Explicit bake(s), derived inputs, output calibration and optional corruption companion; no returned BHdr scoring even though BHdr params provide intermediate front-end defaults. |
| `BakeScorer::compute_hdr` | H + PU F, explicit bake plan and caller encoding; returns scalar. No implicit SDR→HDR weight substitution for an arbitrary bake. |
| `BakeScorer::compute_with_ref_and_attribution`, `prepare_steering`, `prepare_steering_hdr`, `SteeringSession::compute` | retained F, complete scorer gradient, pass B, binned SAT/max/moment removals; >=5 defaults to local-refinement capture N. Older HDR steering accepts basic/peak reads only. `ZENSIM_STEER_EXACT` selects a **Rev5-only** research path. |
| `BakeScorer::score_features`, `score_features_with_identity`, `score_features_fd_gradient`, `score_network_output`; `score_features_with_profile`, `score_features_with_profile_and_codec`, `score_features_fd_gradient_with_profile`, `try_score_from_features` | cached feature/model math only; no pixel kernel. Declared layout/derived inputs, transforms, network/head, spline/postcalibration and score disposition. Caller owns extraction revision provenance. Identity method preserves explicit identity proof. |
| `compute_zensim_with_config`, `compute_zensim_with_ref_and_config`, `precompute_reference_with_scales` | training-config buffered/cached L, default legacy linear weights, raw revision from config/process; routes validate unsupported >=5 masked requests. No named bake is implied. |
| `research::extract`, `research::extract_hdr` | `Request::plan` → F / PU F then layout gather; requested family/ID set, process revision, no model forward. `at_revision(RevisionRef)` is provenance/admission, **not** a mutable formula selector. Palette is a separate training-only extraction after F. |
| `cvvdp_features::extract_cvvdp_features`, `xyb_lms_features::extract_xyb_lms_features`, `compute_iw_weights` (`training`) | separate research extractors; no named scoring bake; not made unreachable by D1–D3. |

## Leaf arithmetic and callers

`incant!` expands the dispatch ladder; source-body counts below count written
bodies, not macro-generated machine-code copies. On canonical dense/gradient
and fused kernels, NEON/wasm wrappers call scalar canon; x86 v3/v4/v4x use the
vector forms. Shared scalar/generic bodies remain necessary for R7.

| Key / form | Files and executable owners / callers | Reach |
|---|---|---|
| F, planned fold | `fold_engine::{compute_fold_backed_stoppable,compute_fold_backed_with_ref_stoppable_plan}` → `feature_v2::{compute_folded_v1_372_streaming_impl,foldapp_streaming_walk_impl,rev5_block_walk}`; producer in `feature_v2_stream::{StripPlaneProducer,BlockPlaneProducer}`; retained route in `feature_v2_retained_core` | R7, B5, wide CD; A/BHdr SDR only with explicit alternate-engine choice. PU wide CD and R7 use H. |
| L, buffered Rev1–3 | `streaming::{compute_zensim_streaming_stoppable,process_strip_channel}` → `blur::{fused_blur_h_ssim_inner*,fused_blur_h_mu_inner*,box_blur_*}` and `fused::{fused_vblur_ssim_inner*,fused_vblur_edge_inner*}`, `simd_ops` for remaining extended/IW passes | A, BHdr1–3, CD and raw research; Rev3 direct error moments and fused extension are gates within these shared bodies, not separate removable files. |
| Canon f32 | `featcanon::Mode::Canon32`; `fused_vblur_ssim_canon<LanesF32>`, dense/gradient generic canon, color generic canon | explicit oracle measurement only; generic bodies also serve c64 scalar fallback and cannot all be removed. |
| Canon64 / rec64 | `featcanon::mode` and `canon_blur_axis_for`; `blur::{fused_blur_h_rec64_row,rec64_run512,rec64_run256,fused_blur_h_ssim_rec64_rows_*}`; `fused_vblur_ssim_canon64_vec` with LanesF64; D/G c64 | BHdr4, CD4, research Rev4. Generic/vector SSIM/D/G bodies also serve R7/B5 with other pools/blur axis. |
| Local windows | `blur::{fused_blur_h_ssim_local,fused_blur_h_local11{,_wide},fused_blur_hv_local11{,_wide},activity_local11{,_wide},box_blur_{h,v}_local*}` and `blur_local_rows::box_blur_v_local_rows`; `fused_vblur_ssim_local` scalar fallbacks | R7 and B5. **Keep**, even though introduced at revision5. |
| R7 row lanes | `featcanon::{Lanes16F64,Lanes16F32,ROW_LANES_FOLD}`; `fused_vblur_ssim_canon64_vec` / `fused_vblur_ssim_local`; `feature_v2_oracle::{dense_block_kernel_canon64_vec,dense_block_kernel_canon}`; `feature_v2_kernels::{gradient_block_kernel_canon64_vec,gradient_block_kernel_canon}`; `blockiness_sparse_strip_wide` and retained-row path | R7 (and research Rev7); false specializations remain B5/BHdr4/CD4 dependencies. |
| D, dense | `feature_v2_kernels::dense_block_kernel_at_revision` → era1 `dense_block_kernel_generic`, era2 `feature_v2_oracle::dense_block_kernel_era2_generic`, canonical D forms, exact D | CD1–3 and raw research use era1/2; R7/B5 and Rev4 callers use canonical. A/BHdr normal narrow score does not run v2 dense. |
| G, gradient | `gradient_block_kernel_at_revision` → generic six entry instantiations, canonical/vector, exact; bandvis/dst-activity/gmsbank const axes remain in these bodies | CD and research; R7/B5 use narrowed supported axes. R7 raw ringing; Rev6/HPM is research-only. |
| C, color/downscale | `color::{srgb_to_positive_xyb_planar_into_at_revision,linear_to_positive_xyb_planar_into,linear_to_positive_xyb_planar_into_unclamped}` → tier-specific old conversion or `srgb_xyb_canon`, `linear_xyb_canon` / vector siblings; `blur::downscale_2x*` | all; `cbrt_newton` only runs under active canonical mode and is true at R7. Legacy downscale is shared, not deletable with old SSIM bodies. |
| H, HDR | `transfer::{decode_pq_row*,decode_hlg_row*}` → `pu21::{pu21_encode*,pu21_encode_canon}` → `color::{pu_xyb_canon,pu_xyb_rows_inner}`; `streaming::compute_multiscale_stats_pu_linear*` or PU F | BHdr, CHdr; A/D explicit PU calls; arbitrary R7 HDR bakes. D2 cannot remove all HDR conversion if R7 HDR API remains. |
| N, local refinement | `local_refine::LocalRefineSnapshot::{capture_from_parts,deltas_range,recompute_planes,grad_delta}`; local blur rows + dense terms + revised G/HF forms; maps read exact local deltas through `ScoredAttribution::refinement_gain` | R7/B5; older/sampled/unsupported plans refuse or do not capture. High-pass ringing refuses capture. |
| Exact f64 | `opsin_px_exact`, `pu_xyb_px_exact`, `fused_vblur_ssim_exact`, `dense_block_kernel_exact`, `gradient_block_kernel_exact`, `append_block_kernel_exact`, `gridblk_strip_wide_exact`; `feature_v2_oracle::oracle::{Neumaier,Exact,dense_reference}` and `ssim_form::stable_ssim_plane` are additional reference contracts | oracle/test/research only. The direct f64 oracle still consumes the walk's rounded planes; this is distinct from an exact end-to-end color oracle. |
| Canonical edge-only | `fused::{vblur_edge_elem_canon,fused_vblur_edge_canon,fused_vblur_edge_canon64_vec{,_scalar,_wasm128,_neon},fused_vblur_edge_exact}` | **none** from ordinary score ingress: sole runtime caller `streaming::process_strip_channel` takes edge-only only when `need_ssim` is false; `active_channels` forces it true for every active canonical channel. Unit tests call these bodies directly. |
| Other banks | append/append2, CSFW, gridblk/ringbasis/tailhist/arttype, gmsbank/mapdev/z1max/gmsnative, palette, CVVDP/LMS; callers in raw extraction and `research::extract` | CD uses declared append families; remaining banks research-only. `ComputeSet::rev5_scope` clears unsupported families for both B5 and R7; palette is not accepted by `Plan::for_bake`. |

## Deletion boundaries and measured source-body sizes

These are **physical function-body lines** including internal whitespace and
comments, excluding surrounding attributes and types (profile declarations
are explicitly marked). They are reproducible
from `scripts/rev7/inventory.py`'s ranges; they are not predicted patch sizes
or a claim that a whole named group can be removed. Counts can overlap other
conceptual groups; do not add the rows to estimate a total deletion.

| Group | Lines / bodies | Configurations needing it | D1 / D2 / D3 consequence |
|---|---:|---|---|
| Canonical edge-only bodies | 437 / 7 | none at score ingress; direct leaf tests | Independent of D1–D3: remove score-dispatch arms and keep/move test ruler if needed. Confirm guard invariant before deletion. |
| Centered stable SSIM reference (`stable_ssim_plane`) | 106 / 1 | tests only | Independent of D1–D3: can compile only in tests/oracle; keep reference tests. `StableSsimScratch` is extra, not included in count. |
| Rec64 H helpers | 1,101 / 22 | BHdr4, CD4, old research | D1 alone changes nothing; D2=drop plus D3 removes named need, but old research must move first. D2=keep requires a deliberate BHdr revision contract (Rev1 only vs replay1–4). |
| Per-tier fused SSIM/edge legacy bodies | 3,177 / 8 | A, BHdr1–3, CD, research | A always retains them (possibly a narrower subset after a separate oracle-backed isolation). D2=drop and D3 reduce callers, not wholesale deletability. |
| Local-window H/V and local rows | 1,599 / 27 | R7 and B5 | Keep under all D1–D3 outcomes. D1 must not delete these as “Rev5 code.” |
| Shared canonical dense bodies | 812 / 5 | R7, B5, Rev4 | Keep; specialize away old pool instantiations only after D1 and old research/HDR replay decisions. |
| Shared canonical gradient bodies | 874 / 5 | R7, B5, Rev4 | Keep; Rev6 HPM/gmsbank/old pool instantiations are narrower branches, not this whole group. |
| Exact leaf bodies listed above (excluding edge-only exact) | 842 / 7 | oracle research | Keep one ruler per needed leaf; move obsolete-family rulers with research. No product score dependency. |
| Standalone CVVDP/LMS/palette function bodies | 666 / 19 | research-only, already `training`-gated | Independent of named-profile decisions: move these with their research callers; deleting public `training` extractors is a separate API action. They are already excluded from default builds. |
| Preview/C/CHdr/D params + C/CHdr/D loaders | 109 declaration/body lines (100 params + 9 loaders) | CD | D3 permits removing those named bindings, not `LegacyLinearV0_2` or Custom. C/CHdr removal frees named append-family demand; raw public research still retains it. Variant arms, docs and tests are additional uncounted edit sites. |
| V0_1 coefficient declaration | 232 declaration lines | PreviewV0_1; public training alias | D3's weights removal can remove it with its aliases through the API review. V0_2's separate 232-line table stays: A, B, BHdr and Custom defaults still use it. |
| BHdr params + loader | 22 declaration/body lines | BHdr **and generic BakeScorer front-end defaults** | D2=drop requires replacing internal BHdr-name references with private front-end defaults before removing the named binding; this 22-line group is not presently BHdr-only. |
| B5-only false row-lane/old-form instantiations | shared bodies above; no separable body count | B5, possibly research | D1 permits retiring the default B5 identity only after all B5 controls migrate. It does not authorize changing arbitrary baked revision5 models. |
| BHdr PU buffered frontend | shared streaming/color/transfer bodies; no separable body count | BHdr (also A/D explicit PU) | D2=keep retains it; D2=drop alone cannot delete shared H helpers or A's explicit PU entry. |

There is **no evidence for deleting all Rev2–6 arithmetic solely from D1–D3**.
The plan's migration/gating of old research is an additional prerequisite.
The unconditional opportunities above are product exclusion or removal of
unreachable dispatch arms; direct reference tests remain load-bearing.

## Environment switches

No captured kept configuration needs a process override: the harness clears
all exported `ZENSIM_*`, sets only Rayon thread count and disables higher SIMD
tokens explicitly. Model-bound R7/B5 choose arithmetic from their bakes; A and
BHdr use the unset process default Rev1.

| Variable | Owner / selection / default | Kept configuration need |
|---|---|---|
| `ZENSIM_FORMULA_REV` | `ssim_form::active_revision`: exact strings1–7, unset/other→Rev1, OnceLock | none for A/R7/B5/BHdr1; needed for existing raw research and BHdr2–4 replays |
| `ZENSIM_SSIM_LUMA` | `active_luma_form`: `ssim2`, `c1`, `lorentz`, `clamp`; default `for_revision` | none; some per-request legacy gates consult it only when revision equals process |
| `ZENSIM_ROOT_FORM` | `det_math::active_root_form`: `libm` or `sqrt`; default revision mapping (Rev1 libm, >=2 nested sqrt) | none; A needs the default libm behavior preserved |
| `ZENSIM_POW_FORM` | `active_pow_form`: `libm` or `pure`; default revision mapping (Rev1 libm, >=2 pure Rust) | none; affects scoring and generic post transforms, so preserve defaults while removing selector |
| `ZENSIM_HF_GAIN` | `active_gain_form`: `ratio`, `bexcess`, `log1p`, `satexcess`, `cap`; default revision mapping | none; explicit research arms |
| `ZENSIM_RING_FORM` | `active_ring_form`: raw default / `hpm`; invalid values panic; Rev6 independently selects HPM | none; R7 raw is required, HPM is research-only |
| `ZENSIM_PYRAMID` | `feature_v2_stream::linear_pyramid`: XYB default (`xyb`) / `linear`; invalid panic; materialized builder refuses linear | none; research changes coarse pixels |
| `ZENSIM_FEATCANON` | `featcanon::override_cell`, oracle only: `exact,c32,c64,neum,off`; absent/unknown no override | none; exact ruler should remain test/oracle-only |
| `ZENSIM_FEATCANON_BLUR` | oracle only `rec,rec64,local,fresh`; default exact→Fresh, >=5→Local, c64 Rev4→Rec64 | none; diagnostic axis |
| `ZENSIM_FEATCANON_MOMENT_CHECK` | oracle only, `1` enables independent moment assertion, otherwise off | none |
| `ZENSIM_ERA2_DENSE` | era2 enabled unless `0`; only noncanonical dense dispatch consumes it | CD/research may reach default; A/R7/B5/BHdr do not need selector |
| `ZENSIM_H_TILE` | `blur::h_blur_tile_width`, parsed usize or 1024 | legacy default bits depend on the tiling; remove override only with default fixed at same value |
| `ZENSIM_REV4_DIAG` | presence enables extra Rev4 tail-histogram bins, default off | none |
| `ZENSIM_FOLD_TIMING` | positive parsed interval enables fold telemetry, default 0 | none |
| `ZENSIM_FOLD_CENSUS_ASSERT` | presence asserts work census when telemetry dumps, default absent | none |
| `ZENSIM_ATTR_PERF` | `1` enables map pass timing output, default off; multiple map owners | none |
| `ZENSIM_MAP_WINDOW` | `MapPaint::from_env`: `box` default, `center`, `guided` | default map geometry must persist; selector unnecessary |
| `ZENSIM_MAP_PAINT` | `nearest` default / `guided` | same |
| `ZENSIM_MAP_SPLIT` | defaults to `MapPaint::served().split` = **exact**; accepts repair/exact/amplitude | keep exact default for bake maps; older fused APIs deliberately use removal default separately |
| `ZENSIM_MAP_SMOOTH` | sigma parsed 0..16, default 0, invalid panic | none; `with_map_smooth` supplies explicit API |
| `ZENSIM_MAP_GUIDE` | `sq` default / `abs` | none; `with_map_guide` supplies explicit API |
| `ZENSIM_NEIGHBOUR_EXACT` | unset→on for revision>=5; `1`→on, any other explicit value→off | preserve default-on R7/B5 local refinement when removing switch |
| `ZENSIM_STEER_EXACT` | absent/`off` false, `all-blocks-v1` true; other values refuse; Rev5 isolated RGB8/bin1 only | none for R7; research diagnostic |
| `ZENSIM_DIFFMAP_UPSAMPLE` | custom-profiles: nearest default; bilinear/catmull-rom/bspline/lanczos3 with optional `-mass`; invalid panic | default proxy-map output retained; alternate policies research |
| `ZENSIM_JBU_GUIDE_STATS` | `1` prints guide statistics only when a guide exists; default off | none |
| `ZENSIM_CROSS_REVISION_DIAGNOSTIC` | compiled only with feature of same name; exact `1` bypasses serving revision agreement with warning | none; not a serving contract |

The exhaustive read-site appendix includes test-only variables and validate
transport variables. In validation, `ZENSIM_MAX_TIER=v3` disables AVX-512;
`ZENSIM_THREADS` chooses Rayon pool size only when `RAYON_NUM_THREADS` is
absent (otherwise default available threads minus4, minimum1). These two
reach pixel execution policy. `tier_audit_features` reads formula/canon/blur
variables to describe/select the same ruler and `ZENSIM_FEATCANON_DUMP` for
its dump prefix. `byc_score_timing` reads `ZENSIM_RING_FORM` only for reporting.
Other validate variables in the appendix control file destinations, data
admission, training or diagnostics; they do not add a product arithmetic gate.

Every occurrence is listed in the [revision selector site index](REV7_SITE_INDEX_2026-10-25.md)
(578 occurrences, 246 function/context groups), and every environment read in
the [environment index](REV7_ENV_INDEX_2026-10-25.md) (100 direct/constant-name
reads across 67 variable names, plus the validate manifest's dynamic capture).

## Existing regression owners

| Test owner | What it actually pins |
|---|---|
| `v1_golden_bytes` | v1 config and fold fixture features, real/nontight/synthetic; finite numerical golden tolerance plus same-class bit determinism, not an A-model score golden |
| `imageflow_checksums` | named A downloaded checksum-image diagnostic; **ignored** in ordinary suite, not executed here |
| `cross_platform` | hardcoded PreviewV0_2 and PreviewV0_1 scores, A pixel formats/determinism/mean offset/validation; numerical tolerance is not cross-target bit identity |
| `cross_tier` | A score tier comparisons with existing ULP/tolerance policy, not every feature's identical bits |
| `featcanon_tier_parity` | full Rev4 research vector tier bits, Rev3 negative control and revision movement |
| `featcanon_rev5_parity` | B5-scope vector tier bits and Rev4-vs5 change guard |
| `featcanon_rev7_parity` | full R7 basic+peaks+v2+globaff tier bits (includes width4133 fold boundary), movement guard against Rev5/6 |
| `featcanon_rev4_contract`, `rev4serve_gate` | named/profile/candidate entry contracts, revision mixing refusal, PU and maps at Rev4; explicit-corpus gate remains ignored |
| `fold_engine_parity`, `v1_feature_width_pure_function` | buffered/fold, cached/scratch, thread-count and stride behavior for legacy profiles/features |
| `native_hdr_color`, `chdr_pu`, `pu_entry`, `pu_linear_extended` | native primaries, PQ/HLG/linear, HDR route, PU layout/cache/stride parity; shared fixture relocation preserves generator tokens |
| `metric::bake::revision_contract_tests`, `attribution_cross_tier`, `bake_surface` | plan/companion and revision binding, canonical cached maps, prepared steering, local refinement contracts; several explicit artifact diagnostics remain ignored |

The new dumps complement these tests: they freeze each legacy tier separately
and every emitted slot of the two R7 bakes. They do not replace spatial-map,
boundary4133, cancellation, source-format or cross-target tests.

## Reproduction and evidence

Run `just rev7-inventory <json>` for lexical occurrences and body ranges;
`just rev7-tool-tests` has negative controls for comments/quoted false sites,
array-type semicolons, empty captures, missing tiers, extra dumps and bit drift.
This scanner does not resolve Rust types or macro expansion; source call
edges and guards above establish the important reachability boundaries.

`just rev7-probe-ssim <new-dir>` was run under the resource wrapper. Temporarily
compiling `StableSsimScratch` and `stable_ssim_plane` only with `cfg(test)`
passed `cargo check --locked -p zensim --lib --all-features` (rc0); the probe
restored `ssim_form.rs` byte for byte, SHA256
`e12de82465dbab9c1a76945d749fe6f776d2dcd26b5822c8fcdfe2321a75717b`.
This proves the centered reference can be excluded from the product build;
it does not assert that all other SSIM reference helpers are unused.

Build with `just rev7-oracle-build`, then run
`scripts/rev7/oracle_dumps.sh <new-dir>` under the resource wrapper. It extends
`perf_bits.sh`, `perf_bits_compare.py`, `byc_score_timing` and `byc_stub_bake`.
Named-profile mode uses `Zensim`, not a parallel scoring implementation.
HDR fixtures are shared with their original integration tests. The dump is
score first, then all `ZensimResult::features()` f64 bits, including structural
zero slots; `full_rev7.ids` names the 600 declared R7 feature IDs.

Start-point verification on r7900x: `cargo test --release -p zensim
--all-features` (with `--locked`, offline dependency resolution) **942 passed,
0 failed, 29 ignored** across 44 result summaries. CI-exact workspace/all-targets/
all-features clippy excluding `zensim-wasm-tests` exited0. Cargo additionally
reports a pre-existing future-incompatibility notice for `proc-macro-error2`;
there are no Clippy diagnostic failures. Baseline wrapper evidence:

```text
tests: run-heavy: done rc=0 455s | peak-RSS 2.50GiB | min-avail 10027MiB | peak-load 16.08
clippy: run-heavy: done rc=0 45s | peak-RSS 1.06GiB | min-avail 15793MiB | peak-load 7.95
```

These resource observations describe those commands, not a product speed or
memory benchmark. Local evidence lives on r7900x under
`~/tmp/rev7unify/`; the coordinator must preserve the baseline/tools before
removing this unlanded workspace. Local-only commits are intentional for this
lane; no network fetch, push, GitHub write or release was performed for the
initial baseline. The coordinator subsequently requested a fetch/rebase.

### Frozen numerical baseline

The final tier-availability-checking harness captured `baseline/capture3` and
`baseline/capture4`. All 305 dumps and the complete 309-file SHA256SUMS manifests
are identical; manifest SHA256 is
`0a5ebc6da2f806f2d74087189136f714e85b13d87389f8c8b87ed4db370513fc`.
Earlier captures1/2 are retained and their 305 numerical dumps also equal these.

| Configuration | Inputs | Tiers / threads | Dumps | Emitted features per dump |
|---|---|---|---:|---:|
| B weights stamped Rev7 | 6 synthetic + 5 photos | v4x/v4/v3/scalar at 1; v4x at 8 | 55 | 720 |
| Full Rev7 stub (600 declared IDs) | same 11 | same 5 policies | 55 | 1,891 |
| Named A | same 11 | same 5 policies | 55 | 372 |
| B5 control | same 11 | same 5 policies | 55 | 720 |
| Named BHdr through PU | same 11 converted to nits + 6 shared HDR fixtures | same 5 policies | 85 | 372 |

All 235,590 dumped words (scores included) are finite. R7 B, full R7 and B5
each have 0 tier mismatches across 55 dumps. A and BHdr intentionally retain
their own tier baselines: comparing every other policy to scalar gives 44/44
and 68/68 differing dumps respectively. This is not a failure of repeated
capture determinism and must not be “fixed” by changing legacy expectations.

The B revision-7 stamp SHA256 is
`5fe0a96230badfa4228e314f12b962b3c18408612d8da9a6390d6259c5a9bcd4`;
full R7 stub is
`cdd9612010b0d1e45f0fde1d02c9547991984dd157d167d904941413a94fdfd4`.
The two final captures each took 22s under the resource wrapper; capture4:

```text
run-heavy: done rc=0 22s | peak-RSS 0.15GiB | min-avail 18805MiB | peak-load 0.56
```

Shared HDR generators are token-identical to their original test bodies;
all 9 tests in `chdr_pu` and `native_hdr_color` pass after relocation. The
strict comparison/scanner tools pass 6 negative-control tests; script lint
and the modified Rust workspace clippy pass. No public API change or relaxed
test expectation is part of phase 1.

### Verification after integration with main

The full release suite was repeated on the rebased stack: **942 passed,
0 failed, 29 ignored**. CI-exact `just clippy` also passed. The focused
`--no-default-features --features custom-profiles` Clippy check passed, covering
the feature combination fixed by `e223fad3`.

Rebuilt tools wrote `~/tmp/rev7unify/post-main/capture1`. Its **305/305 dumps
are byte-identical to baseline/capture4**; input hashes and matrix coverage
also match. The only changed product-source fingerprints are `attribution.rs`,
`globaff.rs` and `map_paint.rs`, exactly the incoming cfg/allow edits. The
original baseline/tools and all original captures remain preserved.

```text
main tests: run-heavy: done rc=0 441s | peak-RSS 2.50GiB | min-avail 10298MiB | peak-load 16.06
main clippy: run-heavy: done rc=0 5s | peak-RSS 0.75GiB | min-avail 17915MiB | peak-load 1.64
post-main capture: run-heavy: done rc=0 22s | peak-RSS 0.15GiB | min-avail 18644MiB | peak-load 0.90
```

Repeat comparisons with `just rev7-bits-compare <base-dumps> <new-dumps>`;
`just rev7-bits-tiers <dumps> <model>` requires all five policies for every
input. These strict comparers refuse empty captures, missing policies,
missing/extra dumps and changed bits. Run the legacy-tier comparison only as
a diagnostic: its differences against scalar are expected and retained.
