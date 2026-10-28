# Migration plan: `ZensimProfile::A` → `ZensimProfile::B`

> **⚠ BANNER 2026-07-18: SHIPPED 2026-07-12** — `codec_target()`/`latest_preview()`/`latest()` all return B (the header's 'PLANNING ONLY' is obsolete). See `docs/PROFILE_B_ROUTING_DESIGN_2026-07-05.md` + README.md. Kept for its file:line survey rigor.

**Status:** PLANNING ONLY. No consumer code has been touched. This document is
a survey + phased plan; execution is future work, tracked per-phase below.

**Author's note on rigor:** every claim below is either (a) traced to a
specific file:line in this repo or a named sibling repo, (b) quoted from a
committed benchmark/changelog doc, or (c) marked **OPEN** with the exact
command to verify it. Nothing here is guessed. Sibling repos
(`~/work/zen/*`, `~/work/coefficient`, `~/work/imageflow`) were surveyed
**read-only** — zero files outside `zensim` were modified to produce this
plan.

---

## 1. Executive summary

`ZensimProfile::B` (`zensim-b`) landed on `zensim` main 2026-07-04 (commits
`fe8b00aa`, `e3438a71`, `6c81f67e`) as an 823-byte deterministic linear
ensemble (`ens-Pline-cid80-anchored`, sha256 `7b326ac56a05c240`) — the
2026-07 campaign's SDR pick, paired with `ZensimProfile::BHdr`
(11.7 KB shaped-feature HDR head, sha256 `373eac56e7a07d6d`). Both are
committed at `zensim/src/profile.rs:747-807` and shipped as `include_bytes!`
constants; nothing about them is experimental or behind a feature flag — they
are exactly as available as `ZensimProfile::A` today.

**B is not a default today.** `ZensimProfile::latest_preview()` and
`ZensimProfile::codec_target()` both still return `Self::A`
(`zensim/src/profile.rs:104-106,176-178`, unchanged by the B landing). Every
consumer that constructs `Zensim::new(ZensimProfile::latest_preview())` /
`::codec_target()` / `::latest()` (deprecated alias) is, right now, still
scoring with `A`. This plan is about *whether and how* to change that — it
recommends **not flipping the shared aliases yet** (§5, Phase 3) pending two
concrete gaps: an unmeasured SDR25 holdout number for B, and zero GPU-side
support for B's architecture.

### B vs. A evidence table (sourced)

| Axis | A | B | Winner | Source |
|---|--:|--:|---|---|
| CID22 (SROCC) | 0.8657 | 0.8733 | **B** | `benchmarks/provenance_best_results_2026-07-04.md` L9,13 |
| KonJND (SROCC) | 0.4185 | 0.5439 | **B** | same, + `zensim/src/profile.rs:46` |
| AIC-3 (SROCC) | 0.7680 | 0.7775 | **B** | `benchmarks/provenance_best_results_2026-07-04.md` L13; A's 0.7680 from `README.md:116` |
| AIC-4 (SROCC) | 0.8854 | 0.8906 | **B** | same sources |
| KADID (SROCC) | 0.7933 | 0.8017 | **B** | `README.md:116` (A); OPEN — B's KADID number not found verbatim in the provenance table (table has no KADID column); verify: `bake_verdict --bake zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin --corpora kadid` |
| TID (SROCC) | 0.7927 | 0.7998 | **B** | same caveat as KADID — OPEN, same verify command with `--corpora tid` |
| Dial monotonicity | q-sweep 94.33% monotone / 0.33% tied (`profile.rs:687-689`) | 0.9711 monotone, 0 dead zones | **B** | `benchmarks/provenance_best_results_2026-07-04.md` L13 |
| UPIQ (SROCC) | 0.6933 | 0.6846 | **A** | `zensim/src/profile.rs:59` (cites A's UPIQ number while documenting BHdr) + provenance table L13 |
| SDR25 | not in table (untested against this axis historically at this bake) | **unmeasured** | — | provenance table L13, column is empty for the B row; see Gate G1 §4 |

7 of 9 axes favor B, 1 favors A (UPIQ, by 0.0087 — noise-adjacent but real),
1 is an open gate (SDR25). This matches the brief's "beats A on 7 of 9 axes"
framing exactly.

**Two structural properties, not in the 9-axis table, found directly in
source and worth weighing independently:**

- **B is collapse-immune by construction.** It's a closed-form Gram-matrix
  least-squares fit (`scripts/v_next/linear_projections_2026-07-03.py`) —
  "no training seed exists" (`profile.rs:47-48`), so the MLP family's
  documented collapse mode (`benchmarks/provenance_best_results_2026-07-04.md`
  §"Falsified this campaign": "MLP stabilization via target choice…via
  selection guard…via seed selection-value ranking") structurally cannot
  recur. A's QAT-native MLP has a known collapse basin
  (CHANGELOG.md "Fixed — trainer: restore Profile-A reproducibility" entry:
  seed 17 at h=64 collapsed AIC-4 0.885→0.546 before the #40 gate fix).
- **B is 33x smaller** (823 B vs. A's 27,316 B / `v47_strict_qat_native_2026-05-27.bin`).
- **Self-identity differs**: a bit-identical pixel pair scores exactly
  **100.0** under B (`zensim/src/profile.rs:1076-1103`,
  `profile_b_loads_scores_and_holds_identity` test, tolerance `< 1e-9`), but
  only **97.69** under A — A has no bit-identical short-circuit; its
  masked-monotone MLP + spline computes through to 97.69 even for literally
  identical bytes (`zensim-experimental/tests/metric_invariants.rs:140-141`:
  *"A's self-identity is the spline's top knot (~97.7), NOT exactly 100 — the
  axiom is that identity is the unique maximum, not a fixed constant"*). This
  is a genuine behavioral difference a "score 100 = lossless" mental model
  will trip over if ported naively. Mechanism for B's exact 100 at d=0 is
  **OPEN** (plausibly an explicit anchor point in
  `scripts/v_next/shared_anchor_refit.py` — not read in this survey; verify:
  `grep -n "100" scripts/v_next/shared_anchor_refit.py`).

---

## 2. Consumer inventory

Surveyed: this repo (zensim workspace — `zensim`, `zensim-validate`,
`zensim-regress`, `zensim-bench`, `zensim-target`, `zensim-experimental`,
`zensim-wasm-tests`, `scripts/`) plus, read-only, `~/work/zen/zenmetrics`,
`zenjpeg`, `zenwebp`, `zenjxl`, `zenavif`, `zenpng`, `zenpipe` (incl. its
`zenfilters`/`zencodecs` members), `zencodec`, `~/work/coefficient`, and
`~/work/imageflow` / `~/work/zen/imageflow` (confirmed identical: same
`origin` remote `imazen/imageflow`, same HEAD `0ba1c9ea8e14`, so treated as
one repo, not two).

**Dependency mechanism matters as much as call-site count** — it determines
who is even *capable* of seeing `B` today, independent of anything this plan
decides:

| Repo | zensim dependency | Resolved / pinned to | Sees `B` today? |
|---|---|---|---|
| zensim (this repo) | n/a | main | yes — it's the source |
| zenmetrics | `path = "../zensim/zensim"` (`Cargo.toml:247`) | live disk state | **yes, already** |
| coefficient | `path = "../zen/zensim/zensim"` (`Cargo.toml:285`, `113`) | live disk state | **yes, already** |
| zenwebp-recompress (sub-crate only) | `path = "../../zensim/zensim"` (`Cargo.toml:21`) | live disk state | **yes, already** |
| zenpipe (workspace incl. zenfilters, zencodecs) | `git = "…/zensim"`, **no rev pinned** (`Cargo.toml:258`) | `Cargo.lock` currently locked to `062bb3ce` (2026-06-12, **158 commits behind** current zensim HEAD) | no — but the *next* `cargo update -p zensim` picks up main, including `B`, with **no version-number signal** (git dep floats; `0.3.0` in Cargo.toml doesn't change) |
| zenjpeg | `git = "…/zensim", rev = "9d8f73a5"` (`Cargo.toml:58`, pinned 2026-05-30) | frozen, predates B | no — needs an explicit, reviewed rev bump |
| zenwebp (main crate) | `zensim = "0.2"` (registry) | `Cargo.lock`: **0.2.7** | no |
| zenavif | `zensim = "0.2.4"` (registry) | `Cargo.lock`: **0.2.7** | no |
| imageflow | `zensim = "0.2.4"` (registry) | `Cargo.lock`: **0.2.7** | no |
| zenjxl, zenpng, zencodec | — (no dependency) | — | n/a, zero consumers |

**Load-bearing finding: `zensim` has never published `0.3.0`.** The staged
`version = "0.3.0"` in `zensim/zensim/Cargo.toml:3` has been sitting under
`## [Unreleased]` in CHANGELOG.md since at least the `0.3.0-unpublished`
section header (`CHANGELOG.md:2252`, dated 2026-05-13) — i.e. **all** of
Profile A's rotations (Tuner v10→v11→v47-QAT) and now all of Profile B have
shipped only to `main`, never to crates.io. `gh issue list --repo
imazen/zensim` shows issue **#46, OPEN, "Release blocker: zensim 0.3.0 cannot
be published while zenpredict is a git dependency"** (2026-06-12). This means
the three registry-pinned consumers (imageflow, zenwebp-main, zenavif) are
blocked from reaching **even current A**, let alone B, by something entirely
outside this migration's scope. Re-verify: `gh issue view 46 --repo
imazen/zensim`.

### 2.1 Inside this repo (zensim workspace)

| Class | Files (count of `ZensimProfile::A` / `latest()` / `codec_target()` call sites) | Representative site | Risk to migrate |
|---|---|---|---|
| (a) Metric scoring, test/bench fixtures | `zensim/tests/*.rs` (~40 sites across `streaming_strips.rs`, `classification.rs`, `size_invariance.rs`, `nan_repro.rs`, `cross_platform.rs`, `cross_tier.rs`, `corpus_icc.rs`, `imageflow_checksums.rs`, `zenpixels_compat.rs`, `medium_hardening.rs`), `zensim/src/streaming.rs:3926`, `zensim/src/diffmap.rs` (8 sites, all `#[cfg(test)]`), `zensim/examples/{zensim_score.rs,downsample_invariance.rs}`, `zensim-bench/{examples,benches}/*.rs` (~14 sites), `zensim-wasm-tests/tests/wasm_regression.rs` (9 sites), `zensim-experimental/{tests,examples,src/lib.rs}` (~10 sites) | `zensim/tests/streaming_strips.rs:140` | **Low.** Internal correctness fixtures; adding a parallel `ZensimProfile::B` test (or parametrizing the existing ones over `[A, B]`) is mechanical. None of these encode an external scale contract. |
| (b) Codec dial / quality targeting | none *inside this repo* — the codec-target consumers are all in sibling repos (§2.2) | — | n/a here |
| (c) Regression gating (scale-coupled) | `zensim-regress/src/checksums.rs` (baseline `.checksums` files + `RegressionTolerance`/`MaxZdsim` thresholds), `zensim-regress/src/profile.rs:23-26` (`legacy_linear()` — a `ZensimProfile::Custom` V0_2 baseline, **unrelated to A/B**, feature-gated behind `custom-profiles`) | `zensim-regress/src/checksums.rs:355` (baseline semantics doc) | **High if touched, zero if left alone.** The tolerance system itself is profile-agnostic (it stores content hashes + a caller-supplied threshold); but every *caller's* hardcoded threshold (e.g. `MaxZdsim(0.05)`) was tuned against whichever profile scored it at authoring time. Switching the scoring profile under an unchanged threshold is a silent behavior change in the consumer, not in this crate. |
| (d) Feature extraction | `Zensim::compute_extended_features` (`zensim/src/metric.rs:1185-1206`) | `metric.rs:1199` (`let params = self.profile.params();`) | **None — verified profile-independent for A vs. B specifically.** See §2.4. |
| Tool gaps (Phase-1 completeness, not "migration" per se) | `zensim-validate/src/bin/rescore_parquet.rs:41-49` (`parse_profile` only recognizes `"a"`/`"zensim-a"`), `zensim-validate/src/bin/upiq_pu_score.rs:73` (hardcoded single-entry `[("zensim_a", …)]` array), `zensim-target/src/bin/zensim_target.rs:72-104` (`parse_profile` has no `"b"`/`"zensim-b"` arm) | `rescore_parquet.rs:43` | **Low, mechanical.** These are exactly the tools Phase 2 (§5) needs; each needs one new match arm. `rescore_parquet` is the rescoring mechanism itself (see §2.4) — its gap is the most consequential of the three. |
| Internal placeholder tags (not real consumers) | `zensim/src/metric.rs:727` (`ZensimResult::nan()`), `zensim/src/metric.rs:3911` (`compute_score`'s internal tail) | both | **None.** Both are immediately overwritten via `.with_profile(self.profile)` by every real call site (comment at `metric.rs:3909-3910`: *"Placeholder profile tag — every `Zensim::compute*` caller overrides it"*). Flagging only so nobody mistakes these greps for a hidden default. |
| Archival / reproduction tooling (not migration targets) | `scripts/reproduce_v47.sh` (reproduces the specific historical A bake, `v47_strict_qat_native_2026-05-27.bin`, by design) | — | **None.** This script's entire purpose is pinning to a named historical bake; it is not meant to track "whatever A is now." |
| Documentation debt | `docs/CODEC_TARGET_METRIC.md` | mapping table, lines 11-19 | **Not a code consumer, but read by every future codec author.** The table still lists `A`'s backing bake as `v39_v32plus_spline_seed17_2026-05-25.bin` (superseded 2026-05-27 by `v47_strict_qat_native_2026-05-27.bin`, confirmed current via `profile.rs:710-712`) and does not mention `B`/`BHdr` at all. Per `docs/NAMING_CONVENTION.md`'s own rule ("Update it in the same commit as any bake rotation"), this table is two rotations stale. Flagged, not fixed, in this planning pass. |

### 2.2 Sibling repos (`~/work/zen/*`, read-only)

**zenmetrics** (`~/work/zen/zenmetrics`) — path-dependency, **already sees B**.

- `crates/zenmetrics-api/src/cpu_dispatch.rs:149,269` — `zensim::Zensim::new(zensim::ZensimProfile::latest_preview())`. Class (b)+(scale-coupled data). **Highest blast radius in the entire survey**: comment at line 147 confirms *"`latest_preview()` matches production sweep workers"* — this is the CPU scoring path behind every fleet/sweep zensim score, including the 372-feature sidecars that back KADIS-700k's `score_zensim` column and the canonical per-codec picker datasets (`s3://zentrain/canonical/2026-06-27/<codec>/`, per this repo's own `CLAUDE.md`). If `latest_preview()` is ever flipped to return `B`, every future fleet run silently rescales without a code change here.
- `crates/zenmetrics-cli/src/metrics/zensim.rs:13,49,68,97,170` — the `zenmetrics score`/`batch --metric zensim` CLI entry point, same `latest_preview()`.
- `crates/zenmetrics-orchestrator/src/cpu_adapter.rs:1232` — orchestrator CPU adapter, same alias.
- `crates/zensim-gpu/src/opaque.rs:116` — `ZensimOpaque`'s **default profile** is `zensim::ZensimProfile::latest()` (deprecated alias, still = A). This is a GPU-side "Default" builder pattern, exactly the kind of implicit-default surface a "recommendation flip" changes automatically.
- `crates/zensim-gpu/src/pipeline.rs:2837,3327` — `ZensimProfile::A` **hardcoded explicitly** in production GPU pipeline code (`DiffmapState::new(ZensimProfile::A)`; `let profile = ZensimProfile::A;`), commented as matched to `RFC_ZENSIM_BUTTLOOP_AUDIT.md` §5. **No GPU kernel for B was found** (`grep -rn "ZensimProfile::B\|PROFILE_B\|linear_bake_b\|ens-Pline\|cid80" crates/zensim-gpu` → zero hits). zensim-gpu's entire design is a from-scratch CUDA reimplementation matched bit-for-bit to A's specific MLP forward pass (`opaque.rs:48`: *"bit-exact equivalent to `zensim::Zensim::new(profile).compute(...)`"*) — this is an open, unstarted port, not a config flip. See Gate G4, §4.
- ~15 more test/example sites in `zensim-gpu/{tests/it,examples}/*.rs`, all `latest()`/`latest_preview()`, parity-test scaffolding.
- `benchmarks/heaptrack/drivers/cpu_profile/{src/main.rs,src/bin/cpu_wall.rs}` (7 sites) — perf-measurement harness (`latest_preview()`); low risk, measures speed not scores.

**zenjpeg** (`~/work/zen/zenjpeg`) — git rev-pinned `9d8f73a5a82a944420ca0e040ecfcea0f4afa263` (2026-05-30 — an ancestor of current zensim HEAD, predates B by over a month; `Cargo.toml:52-58` documents this is a **deliberate** pin, not an oversight: *"so a future zensim rotation of `latest()` cannot silently shift the metric scale out from under the baked GEXP / achieved-quality tables… a committed sibling-path dep breaks every CI job"*).

- `zenjpeg/src/recompress/measure.rs:24` — `const PROFILE: ZensimProfile = ZensimProfile::A;`, explicitly named (not `latest()`/`codec_target()`), with a doc comment explaining exactly why: the recompressor's GEXP calibration tables (`zenjpeg/src/recompress/calibration/per_encoder.rs`) are fit to A's scale. **Safe from any rev bump** — this call site names `A` by identifier, so it will not shift even if the pin is later bumped past a hypothetical future `codec_target()` flip. Migrating it to B is a **deliberate, explicit decision requiring GEXP recalibration**, not a side effect of anything else in this plan.
- `zenjpeg/src/encode/zq.rs:690` — `Zensim::new(ZensimProfile::codec_target())`, the Zq quality-target binary-search encode loop. **This is the one call site in zenjpeg that would shift** if (a) the pin is bumped past a future commit that (b) also flips `codec_target()` to B.
- ~15 more sites across `tests/bundled/*.rs` and `examples/*.rs` (`mozjpeg_parity_tuning.rs`, `mozjpeg_quality_vs_original.rs`, `zq_pareto_calibrate.rs`, `zq_calibrate.rs`, `mozjpeg_parity_regress.rs`), a mix of `codec_target()` and `latest()`/`latest_preview()` — calibration/parity tooling, several with `zensim_regress` tolerance checks (class c).

**zenwebp** (`~/work/zen/zenwebp`) — **split dependency status**: main crate registry-pinned (`Cargo.toml:111`, `zensim = "0.2"` → `Cargo.lock` resolves **0.2.7**, stale); `zenwebp-recompress` sub-crate path-pinned (`Cargo.toml:21`, live).

- `src/encoder/zensim_target.rs:777,1197,1234,1249,1287` — `ZensimTarget` / `EncodeConfig::target_zensim`, **the canonical Pattern-A reference implementation** cited by `docs/CODEC_TARGET_METRIC.md:106-107` (*"Reference implementation: `~/work/zen/zenwebp/src/encoder/api.rs`, `EncodeConfig::target_zensim`"*). Uses `ZensimProfile::latest()`. Running against the stale registry 0.2.7 dependency — **OPEN**: unclear whether 0.2.7 even reflects current-A's v47-QAT bake or an older one; verify with `cargo tree -p zensim --manifest-path ~/work/zen/zenwebp/Cargo.toml` then diff against `zensim/CHANGELOG.md`'s dated entries around whatever date 0.2.7 was tagged.
- `zenwebp-recompress/src/measure.rs:60` — `ZensimProfile::A`, explicit, path-pinned (live).
- ~15 more sites in `tests/*.rs` and `dev/*.rs` (`zensim_regression_matrix.rs`, `webpx_regression.rs`, `cross_format_equivalence.rs`, `vs_libwebp_matrix.rs`, `zensim_calibrate.rs`, `zenwebp_pareto.rs`, `zensim_ceiling_probe.rs`) — `latest()`, mostly regression/sweep tooling with `zensim_regress` tolerance checks (class c).

**zenavif** (`~/work/zen/zenavif`) — registry-pinned (`Cargo.toml:36,71`, `"0.2.4"` → `Cargo.lock` resolves **0.2.7**, stale). Has a committed `[patch.crates-io]` section, but it patches `zenavif-serialize`, not `zensim` (`Cargo.toml:250-253`).

- `src/target_quality.rs:409` — production quality-targeting path, `ZensimProfile::latest()`.
- ~6 sites in `tests/linku_corpus.rs`, `examples/{phase2_oat,predictor_sweep,encode_sweep,sweep_validate}.rs` — `latest()`, regression-sweep tooling with `zensim_regress` tolerance checks (class c).

**zenpipe** (`~/work/zen/zenpipe`, workspace incl. `zenfilters`, `zencodecs` members) — git dependency with **no rev pinned** at the workspace level (`Cargo.toml:258`, `zensim = { git = "https://github.com/imazen/zensimx" }`); `Cargo.lock` currently resolves commit `062bb3cedfbb6e723235044fb0818efaaa493d93` (2026-06-12, **158 commits behind** current zensim HEAD, predates B). Unlike zenjpeg's disciplined explicit-rev pin, this floats — the next `cargo update -p zensim` in this repo silently adopts whatever is on zensim `main`, with zero version-number signal (Cargo.toml still just says "the git repo", `zensim`'s own `Cargo.toml` version stays `0.3.0` either way).

- `zenpipe/zencodecs/src/transcode.rs:463-467` — **production code** (`#[cfg(all(feature = "transcode-iqa", feature = "jxl-decode"))]`, not test-gated), the JPEG→JXL coefficient-domain recompression path: `let metric = Zensim::new(ZensimProfile::A);` feeds this scorer into `zenjxl::jpeg_lossy::recompress_jpeg_lossy_target`. Explicit `A`, no stated rationale in the doc comment (unlike zenjpeg's measure.rs) — worth asking zenpipe's owner whether this pin is deliberate or incidental.
- `zenpipe/zenfilters/tests/{parameter_calibration,quality_validation,reference_validation,imagemagick_comparison}.rs` — `ZensimProfile::codec_target()`, filter-parameter **calibration** tests (class c — these very likely assert against specific expected score values tuned to A).
- `zenpipe/zenfilters/examples/*.rs` (`compare_autotune`, `mobile_parity`, `train_autotune`, `clipart_flatten_demo`, `darktable_parity`, `whitebg_corpus`) — 9 sites, `latest()`, dev calibration tools.

**zenjxl, zenpng, zencodec** — **zero zensim dependency found** (`grep -rn "zensim" Cargo.toml` across all three: no hits; `zenjxl`/`zenpng` have no `zensim` line in any workspace member's `Cargo.toml`). Consistent with `docs/CODEC_TARGET_METRIC.md:198-207`'s per-codec table showing zenjxl/zenavif/zengif's "Pattern A" as merely "tracked"/unimplemented for zenjxl specifically. **No migration action needed for these three.**

### 2.3 Outside `~/work/zen`

**coefficient** (`~/work/coefficient`, NOT under `~/work/zen`) — path-pinned (`Cargo.toml:113,285`, `path = "../zen/zensim/zensim"`), **already sees B**. This is the single largest concentration of *indirect* (auto-shifting) call sites found in the entire survey:

- `src/metric/zensim_metric.rs:65`, `src/metric/zensim_cached.rs:107` — core metric wrappers, `ZensimProfile::latest()`.
- 9 example RD-sweep/selector scripts, **all** `ZensimProfile::latest()`: `boundary_rd_zensim_check.rs:123`, `bdrd_montage_pair.rs:471`, `selector_vs_oracle.rs:86`, `spot_zensim_ba_weights.rs:143`, `sweep_bdrd_params.rs:100`, `sweep_bdrd_text_illus.rs:85`, `sweep_v3_v4_v5.rs:72`, `sweep_v3_vs_v4.rs:74`, `sweep_selector_v3.rs:89`.

Every one of these 11 sites would shift the instant `latest()`/`latest_preview()` returns `B` — zero code change needed on coefficient's side, since it's live path-pinned. This makes coefficient the **highest-leverage** (and highest-risk-of-surprise) consumer for Phase 3 (§5).

**imageflow** (`~/work/zen/imageflow` == `~/work/imageflow`, one repo, confirmed via identical `git remote -v` and `git rev-parse HEAD` = `0ba1c9ea8e1431c5709c4231aa6762e8bb08cf8b` in both checkouts) — registry-pinned (`imageflow_core/Cargo.toml:119`, `zensim = "0.2.4"` → `Cargo.lock` resolves **0.2.7**; `zensim-regress = "0.3.0"` → resolves **0.3.1**).

- **Zero production runtime usage.** `imageflow_core`'s actual image-processing pipeline never calls zensim. All 6 hits are in the **visual regression test suite**: `tests/integration/{sync.rs, common/mod.rs, common/macros.rs, common/upload_tracker.rs}`, `tests/integration/visuals/codec.rs`.
- `tests/integration/common/mod.rs:477` — `Zensim::new(ZensimProfile::latest())`, backs `compare_bitmaps_zensim()`, which every `MaxZdsim(threshold)` visual-diff assertion in the test suite routes through.
- `tests/integration/visuals/codec.rs:42` — concrete example of a hardcoded, scale-coupled threshold: `similarity: Similarity::MaxZdsim(0.05), // measured centos zdsim: 0.036`. This comment literally records the *measured* zdsim under whatever profile scored it (currently 0.2.7's version of `latest()`) — switching the scoring profile without re-measuring risks silently exceeding 0.05 on a threshold that was never re-validated for the new profile.
- Blocked from reaching B (or even current A) by the same crates.io-publish gate as zenwebp-main/zenavif (§2, issue #46).

### 2.4 Feature extraction is confirmed profile-independent for A vs. B specifically

Directly verified from source, not inferred: `PROFILE_A` (`zensim/src/profile.rs:716-745`) and `PROFILE_B` (`profile.rs:764-782`) declare **identical** front-end parameters —

```
blur_radius: 5, blur_passes: 1, num_scales: 4,
extended_features: true, compute_iw_features: true
```

`compute_extended_features` (`metric.rs:1185-1206`) reads `self.profile.params()` only for these front-end knobs plus `params.weights`/`mlp_bytes` (used solely in the *scoring* tail, `compute_with_config_inner`, not in feature computation itself). Since A and B agree on every front-end knob, **the 372-dimensional feature vector `.features()` returned for a given `(source, distorted)` pair is bit-identical whether constructed via `Zensim::new(A)` or `Zensim::new(B)`; only `.score()` (the bake-specific forward pass + calibration spline) differs.**

This has one directly actionable consequence: `zensim/src/lib.rs:287-294` exposes `try_score_from_features` (a `training`-feature-gated API, replacing the removed panicking `score_from_features` as of 0.3.0) that scores an **already-extracted** feature vector against a chosen profile. Combined with the feature-identity fact above, **any fleet/sweep pipeline that has already cached 372-feature vectors (KADIS-700k, canonical picker parquets, any `rescore_parquet`-style sidecar) can backfill a `score_zensim_b` column via a pure re-scoring pass — no re-decode, no re-extract, no GPU/CPU pixel work at all.** This is the exact mechanism `zensim-validate/src/bin/rescore_parquet.rs` already implements for `A` (its own doc comment: *"rescore every cell from its stored feature vector — `score_features_with_profile` is bit-exact with a full `compute()` and needs no re-encode"*); it just needs the `"b"` match arm noted in §2.1's tool-gaps row.

### 2.5 Scale-coupled data artifacts (not code, but load-bearing)

- **KADIS-700k canonical parquets** — both the 2026-06-30 (`score_zensim`) and 2026-07-01 GPU-metrics (`score_zensim_gpu` + 6 other perceptual scores) variants carry a zensim score column produced via `Zensim::compute_extended_features` at whatever profile was default at generation time (`A`, since `latest_preview()` hasn't moved). Per this repo's own `CLAUDE.md` §"ML Data Pipeline Discipline" rule 8 ("one canonical R2 path per artifact, no duplicates") and the "never in-place" ethos throughout that section, **any B backfill must land as a new `score_zensim_b` column, never overwrite `score_zensim`.**
- **Canonical per-codec picker datasets** (`s3://zentrain/canonical/2026-06-27/<codec>/`) — `_MANIFEST.json`-documented zensim/ssim2 scores per this repo's `CLAUDE.md`. Same "additive column, never in-place" rule applies.
- **`.checksums` baseline files** — scattered across zenjpeg/zenwebp/zenavif/imageflow's visual-regression suites (§2.2, §2.3). These don't store raw scores (they store content hashes + petnames), but the accompanying hardcoded `MaxZdsim`/`RegressionTolerance` thresholds in each repo's Rust test source are scale-coupled per §2.1 class (c).

---

## 3. Pre-migration gates

These must be resolved (or explicitly waived by the user) before Phase 3
(§5) — flipping any shared alias (`latest_preview()`/`codec_target()`) —
is responsible to execute.

### G1 — SDR25 measurement for B

The provenance table (`benchmarks/provenance_best_results_2026-07-04.md`
L9-19) has an `SDR25` column populated for every MLP candidate
(`w3_t1dro51_s31`: 0.9694; `w7_guard_s101`: 0.9538) but **blank for both B
rows**. SDR25 appears to be a T0-tier holdout per this repo's
`DATA_SPLITS.md` framework (per project memory: *"AIC-3 420k trains, SDR25
becomes new T0"*) — i.e. plausibly the **primary** dial-fidelity holdout for
exactly the codec-quality-targeting use case `codec_target()` exists for.
Shipping B as the codec-target default without this number is shipping the
dial-critical consumer class blind on its most relevant recent holdout.

**Action:** run `bake_verdict --bake
zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin --corpora sdr25`
(or the SDR25-specific harness referenced in `scripts/hdr/upiq_panel.py` /
`scripts/v_next/reconstruct_sdr25_jnd.py` if `sdr25` isn't a `bake_verdict`
corpus name — **OPEN**, verify which harness owns this corpus first: `grep
-rln "sdr25\|SDR25" zensim-validate/src/`).

### G2 — Dial-equivalence study for `codec_target()` consumers

Confirmed from source (§1): A's practical dial ceiling (self-identity) is
**97.69**; B's fitted-knot ceiling is **95.9**
(`benchmarks/linear_projections_2026-07-03.md:645`, *"knot ranges [0,95.9] /
[24.8,88.6] (cid80's >100 wart FIXED)"*) though B's true zero-distance point
extrapolates to exactly **100.0** (§1, mechanism OPEN). Cross-model
disagreement in the mutual dial zone (40-95, where most `target_zensim`
workloads per `docs/CODEC_TARGET_METRIC.md:88-93` actually live) is
**MAE 5.05pt** (same doc, "Shared-anchor dial refit" section). This is not
small relative to the ±1.0 tolerance `docs/CODEC_TARGET_METRIC.md:119` cites
as typical for `EncodeConfig::target_zensim` callers.

**Action before any codec_target() flip:** any q↔score lookup table a codec
keeps (zenjpeg's Zq calibration, zenwebp's `target_zensim` starting-point
table per `CODEC_TARGET_METRIC.md:158-160`) was fit against A's scale and
needs re-derivation against B's scale — or the flip needs to ship alongside
a documented "expect ~5pt absolute disagreement vs. historical A-scored
data" caveat. Neither has been attempted; treat as unstarted work, not a
quick fix.

### G3 — zensim-regress baseline re-derivation policy

Every `.checksums` baseline + hardcoded `MaxZdsim`/`RegressionTolerance`
threshold across zenjpeg/zenwebp/zenavif/imageflow (§2.1 class (c), §2.2,
§2.3) was authored against whichever profile scored it at the time (`A`,
since that's what `latest()`/`codec_target()` have always resolved to).
Flipping the scoring profile under an unchanged threshold is **not caught by
CI** unless the flip pushes some measured zdsim over its threshold — i.e. it
fails *silently* for every pair that happens to stay within tolerance, and
*loudly but confusingly* (looks like a real regression, isn't) for the
pairs that don't.

**Policy recommendation (not yet executed, needs user sign-off per this
repo's "NEVER relax tolerances/thresholds" rule):** before any per-consumer
default flip (§5 Phase 4), re-run that consumer's full visual-regression
suite scoring with **both** A and B, diff the two zdsim distributions per
test, and only then decide whether existing thresholds hold or need
re-baselining. Re-baselining a threshold is a threshold change and must go
through the same "STOP and ask user" gate as any other tolerance edit — this
plan does not pre-authorize it.

### G4 — GPU parity gap (zensim-gpu)

`zensim-gpu` hardcodes `ZensimProfile::A` in production pipeline code
(`crates/zensim-gpu/src/pipeline.rs:2837,3327`) and its entire architecture
is a from-scratch CUDA reimplementation bit-matched to A's specific 372→128→64
MLP forward pass. Zero references to `B`, `PROFILE_B`, `linear_bake_b`, or
`ens-Pline`/`cid80` were found anywhere in `crates/zensim-gpu` (verified via
`grep -rn`, zero hits). B's architecture (a single linear layer / dot
product) is structurally far simpler than A's MLP — this could make a GPU
port trivial or could simply be unstarted; either way it does not exist
today. **Any fleet/sweep work that depends on GPU-accelerated zensim scoring
(the CUDA metrics tier in `zenmetrics`) cannot use B until this is built.**
CPU-side consumers (§2.2 zenmetrics's `cpu_dispatch.rs`, coefficient, etc.)
are unaffected by this gate.

### G5 — the cid80 >100-knot history (resolved, confirmed)

The provenance table (`benchmarks/provenance_best_results_2026-07-04.md`
L14) lists a **pre-anchor twin** of the shipped bake — "its pre-anchor twin
(tau0)", same 823 bytes but sha256 `1cddfe5e14d81128`, with knots exceeding
100 ("superseded"). **Confirmed the SHIPPED bake is the anchored (fixed)
sibling, not the tau0 twin**: `zensim/src/profile.rs:749-751`'s
`linear_bake_b_cid80()` loads `weights/b_sdr_linear_cid80_anchored_2026-07-04.bin`,
documented sha256 prefix `7b326ac5…` — this matches the provenance table's
**"Profile B (SDR)" row** (`7b326ac56a05c240`), not the tau0 row. Separately,
the *evaluation-harness*-side half of this bug class was fixed in commit
`5d4978db` (per this session's git log: *"output-calibration spline upper
extrapolation now caps at 100 — parity with the product runtime"*) — that
commit fixed `zensim-validate`'s scoring path to match the runtime's
already-correct cap, it did not change which bake ships. **No outstanding
action** — both halves (bake selection, harness parity) are confirmed
correct as shipped. Documented here so a future session doesn't re-open it.

### G6 — zensim 0.3.0 publish blocker (prerequisite, not this plan's scope)

Per §2's dependency table: imageflow, zenwebp-main, and zenavif are stuck on
registry `0.2.7` and cannot reach **either** current-A or B without (a)
`imazen/zensimx#46` being resolved, (b) an actual `0.3.0` (or later) publish,
and (c) each consumer bumping its own `Cargo.toml`/lockfile. This is a
precondition for those three repos to participate in *any* phase of this
plan beyond "does not apply yet" — flagged, not solved, here.

---

## 4. Phased rollout

### Phase 1 — Additive (DONE)

`ZensimProfile::B` / `BHdr` exist, are public, documented, and tested
(`profile_b_tests`, `profile_b_routing_tests`,
`descriptor_hdr_routing_tests`, `linearf32_sdr_not_hdr_tests` in
`profile.rs`). Nothing defaults to them. **Completed 2026-07-04**
(`fe8b00aa`, `e3438a71`, `6c81f67e`).

**Two small mechanical gaps remain within this phase** (found in §2.1, not
new work — just finishing what "additive, available" should mean):
- `zensim-validate/src/bin/rescore_parquet.rs:41-49` — add `"b" | "zensim-b"
  => ZensimProfile::B` to `parse_profile`.
- `zensim-target/src/bin/zensim_target.rs:72-104` — add the same arm.
- `zensim-validate/src/bin/upiq_pu_score.rs:73` — add a `("zensim_b",
  Zensim::new(ZensimProfile::B))` entry to the comparison array.

Effort: **~1 hour**, three small diffs, no design decisions.

### Phase 2 — New-data-only (additive columns)

Backfill `score_zensim_b` onto existing cached-feature datasets **without
re-encoding or re-extracting**, using the profile-independent-features fact
(§2.4) + `try_score_from_features`. Concretely:

1. Extend `rescore_parquet` (post-Phase-1 gap fix) to run both A and B over
   the same `feat_0..feat_371` columns, emitting `score_zensim_a` (rename
   from bare `score_zensim` for clarity, or keep `score_zensim` = A for
   back-compat and add only `score_zensim_b`) — **decision needed from
   user**, this plan does not presume which.
2. Apply to the KADIS-700k canonical parquets and the
   `s3://zentrain/canonical/2026-06-27/<codec>/` picker datasets (§2.5) —
   additive columns per this repo's data-pipeline discipline, never
   in-place.
3. `zenmetrics`'s fleet sidecars (driven by `cpu_dispatch.rs`, §2.2) could
   gain a `--also-score-with b` style flag so *future* sweep runs emit both
   columns going forward, without touching `latest_preview()`'s return
   value at all.

Nothing here changes any consumer's *scoring behavior* — it only adds
comparison data future decisions (Phase 3+) can be based on. This is the
safest phase to execute immediately and does not depend on any gate in §3.

Effort: **~1-2 days** (tool changes + one fleet-scale backfill run; the
backfill itself is CPU-only re-scoring of cached features, cheap per §2.4 —
no GPU time, no re-encoding, no corpus re-download).

### Phase 3 — Recommendation flip (`latest_preview()` → B?)

**Recommendation: do not flip yet.** Justification:

- G1 (SDR25 unmeasured) and G2 (dial-equivalence unstudied) are both
  unresolved, and both bear directly on `codec_target()`'s stated purpose
  (`docs/CODEC_TARGET_METRIC.md:1-3`: *"the stable alias… all zen codecs
  train and target to"*). Flipping the alias before either gate closes
  means shipping a codec-facing dial change with an unmeasured holdout and
  no re-derived q-lookup tables.
- The blast radius if flipped today is large and mostly *silent*:
  `zenmetrics` (fleet driver), `coefficient` (11 call sites, all indirect),
  and `zenwebp-recompress` are all **live path-pinned** — they would shift
  the moment the alias changes, with no Cargo.toml edit anywhere to signal
  it happened. That's a feature when the goal is "bake rotations flow
  through for free" (the documented design intent,
  `docs/CODEC_TARGET_METRIC.md:41-42`) and a liability when the rotation in
  question changes the *scale contract* materially (§2.2 G2's 5pt MAE) —
  A's historical rotations (Tuner v10→v11→v47-QAT) were all same-architecture
  bake swaps; A→B is an architecture change (MLP → linear) with a measured
  scale shift, a different category of "rotation."
- B is younger (landed same day this plan was written) with less
  production burn-in than A had at each of its rotations.

**What would justify flipping:** G1 and G2 closed with results that don't
contradict the 7/9-axis case for B, plus explicit user sign-off given the
live-path-pinned blast radius above. This plan surfaces the decision; it
does not make it.

**If/when flipped:** update `zensim/src/profile.rs:104-106` (`latest_preview`)
and/or `:176-178` (`codec_target`) in the same commit as: (a) the
`docs/CODEC_TARGET_METRIC.md` mapping-table fix (already overdue per §2.1),
(b) a CHANGELOG entry stating the scale-shift magnitude (cite G2's 5pt MAE),
(c) notification to zenmetrics/coefficient/zenwebp-recompress owners given
their live-pin exposure (not a code requirement, a courtesy — nothing
technically stops the flip from reaching them silently).

Effort if executed: **~1 day** for the flip + doc updates; **unbounded**
until G1/G2 close (that work is not scoped here — it's HDR/SDR corpus
measurement and codec q-table re-derivation, each plausibly multi-day).

### Phase 4 — Default flip per consumer, with per-phase verification

Independent of Phase 3 (a consumer can adopt `B` explicitly by name without
waiting for the shared alias to move). Per-consumer sequencing, ordered by
dependency-mechanism readiness (live-pinned first, since they need no
Cargo.toml work) and by risk (test-only before production, additive before
threshold-touching):

1. **zenmetrics** — add explicit `--profile b` scoring path alongside
   `latest_preview()` (Phase 2 already covers the data side); flip
   `cpu_dispatch.rs`'s default only after G1-G4.
2. **coefficient** — same shape; 11 sites, all currently indirect via
   `latest()`. Given it's a research/picker-training tool, not a shipped
   product, this is the natural first place to dogfood B as an explicit
   opt-in before touching any shipped codec.
3. **zenwebp-recompress** (path-pinned sub-crate, explicit `A` in
   `measure.rs:60`) — low-risk, single file, no external users depend on
   its exact scale (it's a dev/measurement tool, not shipped in the
   `zenwebp` crate proper).
4. **zenjpeg** — requires (a) a reviewed git-rev bump past B's landing
   commit, (b) a decision on `zq.rs:690`'s `codec_target()` call (moot
   until Phase 3 or an explicit local override), (c) GEXP recalibration
   if `measure.rs:24`'s explicit `A` pin is ever deliberately changed
   (not recommended without a full recalibration pass — this pin exists
   specifically to prevent silent drift).
5. **zenpipe** (`zencodecs`, `zenfilters`) — needs an explicit rev pin
   added first (currently floats, §2.2) purely as a hygiene fix
   independent of B; then the same per-call-site decisions as zenjpeg.
   `zencodecs/transcode.rs:467`'s explicit `A` needs its own
   rationale-or-migrate decision (unlike zenjpeg's measure.rs, no
   documented reason was found for the pin — worth asking the owner
   before assuming it's deliberate).
6. **zenwebp (main), zenavif, imageflow** — blocked on G6 (crates.io
   publish) regardless of any decision here; not actionable until that
   independent prerequisite clears.

Each per-consumer flip needs, at minimum: G3's before/after zdsim diff on
that consumer's regression suite, and (for codec-dial consumers) G2's
re-derived q-lookup table.

Effort: highly variable per consumer; zenmetrics/coefficient/zenwebp-recompress
are **~1-2 days each** (add opt-in path + verification diff); zenjpeg/zenpipe
are **~3-5 days each** (rev-pin hygiene + GEXP/calibration re-derivation);
zenwebp-main/zenavif/imageflow are **blocked, not estimable** until G6
clears.

### Phase 5 — A retained indefinitely

`ZensimProfile::A` is never removed by this plan. Per `docs/NAMING_CONVENTION.md`,
explicit variant names are exactly the mechanism for frozen reproducibility —
anything needing bit-exact historical behavior (papers, `scripts/reproduce_v47.sh`,
zenjpeg's GEXP-pinned `measure.rs`) continues to name `A` directly regardless
of what `latest_preview()`/`codec_target()` point at. No sunset date is
proposed.

---

## 5. Risks + falsified-roads appendix

**Why B is unlikely to collapse (addressing the MLP family's known failure
mode):** B's fit is closed-form least-squares over a fixed feature set — no
gradient descent, no training seed, no local minimum to fall into. The
provenance doc's own falsified-roads section
(`benchmarks/provenance_best_results_2026-07-04.md`, "Falsified this
campaign") lists MLP-family collapse under three different stabilization
attempts (target choice, selection guard, seed ranking) as dead ends for
*that* architecture — none of those failure modes have a linear-fit analog,
since there's no iterative optimization to collapse. The determinism proof
(44/44 refits byte-identical across a fresh pipeline re-run,
`linear-probe/determinism_check.py`) is direct evidence, not an inference.

**Where B is genuinely weaker (don't oversell it):**
- **UPIQ**: A 0.6933 vs. B 0.6846 — B loses this axis, small but real
  (§1 table).
- **SDR25**: entirely unmeasured for B (G1) — this is a gap, not a loss,
  but it must not be silently treated as "presumably fine because B wins
  elsewhere."
- **Self-identity semantics differ** (§1): A's practical ceiling (97.69) is
  *lower* than B's exact 100 at true zero-distance — anyone with hardcoded
  expectations keyed to A's ~97.7 ceiling (e.g. "scores above 98 are
  suspicious/impossible") will get surprised by B.
- **GPU support is zero** (G4) — this is an engineering gap, not a quality
  finding, but it hard-blocks any GPU-accelerated consumer today.
- **Dial ceiling is lower in the fitted region** (95.9 vs 97.69,
  §1/G2) — codecs whose q-lookup tables assume they can reach ~97-98 on
  clean high-q encodes will see a compressed top end under B.

**Falsified / not-attempted roads relevant to this migration** (from the
provenance doc's own record, so a future session doesn't retry them):
per-domain MLP corrections at every tested λ (SDR residual falsified at
every level — "target defect, not architecture", per
`benchmarks/provenance_best_results_2026-07-04.md` w10/w10b sections), CSF
feature engineering (re-confirmed falsified, v39-era finding holds),
2-stage linear cascades. None of these bear directly on "should A's
consumers move to B" but they're the reason B (a single-stage linear fit)
is the shape it is, and why "just retrain a bigger MLP instead of
migrating" was already tried and abandoned this same campaign.

---

## 6. Effort estimates (summary)

| Phase | Scope | Estimate | Blocked on |
|---|---|---|---|
| 1 (finish) | 3 tool match-arms | ~1 hour | nothing |
| 2 | Additive `score_zensim_b` backfill, cached-feature rescoring | ~1-2 days | nothing |
| 3 | Flip `latest_preview()`/`codec_target()` | ~1 day execution, but gated | G1 (SDR25), G2 (dial study) — each independently multi-day, unscoped here |
| 4a | zenmetrics, coefficient, zenwebp-recompress (live-pinned, opt-in) | ~1-2 days each | Phase 2 |
| 4b | zenjpeg, zenpipe (rev-pin hygiene + calibration) | ~3-5 days each | G3 (per-consumer regression diff), GEXP recalibration if measure.rs pins move |
| 4c | zenwebp-main, zenavif, imageflow | blocked, not estimable | G6 (zensim 0.3.0 publish — external prerequisite, tracked as `imazen/zensimx#46`) |
| 5 | Retain A | ongoing, zero effort | n/a |

**No tracking issue for this specific migration was found** (`gh issue list
--repo imazen/zensim` shows #22, #38, #46, #48, #50 open, none of which is
"migrate A→B consumers" — #46 is the adjacent publish-blocker, #50 "zensim
cliffs at the top" may be related dial-ceiling prior art worth reading before
G2 work starts, `gh issue view 50 --repo imazen/zensim`). Filing one is a
reasonable next step but is left to the user's judgment, not done as part of
this planning pass.
