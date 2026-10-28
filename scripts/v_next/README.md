# `scripts/v_next/` — Python helpers for zensim research

Python helpers for corpus preparation, screening, baking via the
ZNPR v3 JSON pipeline, and analysis. All scripts assume the
`/mnt/v` mount + the zensim workspace at `/home/lilith/work/zen/zensim`.

## Where to look first

New models must execute and serve entirely in Rust through a zensim surface
API, and evaluation must use that API (including heads and splines). Python
prototypes and independent references remain useful. See the
[authorized cleanup plan](../../docs/PLAN_CRUFT_PURGE_2026-09-06.md) for current
owners, retired July pipelines and their frozen source revision.

| If you want to… | Use |
|---|---|
| Compute IW-SSIM on the safesyn corpus | [`compute_iwssim_on_safesyn.py`](compute_iwssim_on_safesyn.py) |
| Run IW-SSIM on a fleet | The zenfleet job system (`zenmetrics/scripts/jobsys/`) — CLAUDE.md forbids hand-rolled fleet orchestration. `vastai_iwssim/` was committed in 5ccea813 and later deleted; its deployment plan is gone too. |
| Screen per-feature transforms (V_20 IS pipeline) | [`v0_20_feature_transform_greedy_screen.py`](v0_20_feature_transform_greedy_screen.py) → Rust trainer `--auto-transforms` |
| Distill V_20b (contrastive pre-train + fine-tune) | [`v0_20b/`](v0_20b/) (subdirectory; see its own README) |
| Affine-calibrate a bake (distance → score) | `cargo run --release -p zensim-validate --bin affine_calibrate` — the Rust owner since 2026-06-18. (`affine_calibrate_znpr_v2.py` was deleted; this row pointed at it long after.) |
| Build the interactive comparison-site data | [`build_site_data.py`](build_site_data.py) + [`build_scatter_data.py`](build_scatter_data.py) |
| Export a corpus to parquet for the site | [`export_human_corpora_to_parquet.py`](export_human_corpora_to_parquet.py), `export_aic3_to_parquet.py`, `export_aic4_to_parquet.py` |
| Verify a baked MLP reproduces a known SROCC | `bake_verdict --bake <bin>` (Rust; ~3.5 s for all corpora). `verify_bake_srocc.py` was deleted 2026-07-15 — its target binary had no source anywhere. |
| Train a cross-codec seed bake (any generation) | [`run_cross_codec_seed.sh`](run_cross_codec_seed.sh) `<v2\|v3\|v4\|v4b\|v5\|v6\|v7\|v8\|v9\|v9cons\|v9mono> <seed> [args]` — ONE recipe; per-experiment knobs in [`cross_codec_variants/`](cross_codec_variants/README.md). All eleven historical names still work (thin shims). |
| Prove the consolidated driver still runs each experiment's exact argv | [`tests/test_cross_codec_seed_argv.sh`](tests/test_cross_codec_seed_argv.sh) — no data, no trainer, no network |
| Run the cross-codec eval harness for a generation | [`eval_cross_codec.sh`](eval_cross_codec.sh) `<v4\|v4b\|v5\|v6\|v7\|v8>` |
| Check multi-codec PJND score agreement | [`cross_codec_pjnd_check.py`](cross_codec_pjnd_check.py) `<exp> <dir>` |
| Check multi-band cross-codec parity | [`cross_codec_multi_band_check.py`](cross_codec_multi_band_check.py) `<exp> <dir>` |
| Build a cross-codec summary scorecard | [`summarize_cross_codec.py`](summarize_cross_codec.py) `<exp> <dir>` |
| Sync the unified parquet store to R2 | [`sync_unified_to_r2.sh`](sync_unified_to_r2.sh) |

## Grouped by theme

### IW-SSIM (added 2026-05-15)

| Script | Role |
|---|---|
| `compute_iwssim_on_safesyn.py` | Compute Wang & Li 2011 IW-SSIM per pair via `piq.information_weighted_ssim` (PyTorch, CUDA). Outputs parquet sidecar with `source_path, decoded_path, iwssim`. ~7.3 pairs/sec on a 4090; ~7.5 hr for the 196k-pair safesyn corpus. (A `vastai_iwssim/` subdir once parallelized this; it was deleted. Use the zenfleet job system.) |

### V_20 input shaping (V_20 IS / V_20a / V_20b / V_20 extended)

| Script | Role |
|---|---|
| `v0_20_feature_transform_greedy_screen.py` | For each feature column, try every `FeatureTransform` and report the one with maximum Pearson lift over identity. Output: TSV with `feat_idx, best_transform, params_csv, lift, ...`. Run against any per-pair features CSV. |
| `v0_20_parse_reeval_logs.py` | Parse `dataset_metric_baseline` per-corpus eval logs + collate full Mohammadi panel rows into a consolidated comparison markdown. |
| `v0_20_extract_statistical_panels.py` | Same as parse_reeval_logs but reads training-time validation logs + emits the full panel structure. |
| `v0_20_low_n_band_analysis.py` | For (corpus, band) cells with n < 100, compute the empirical SROCC ceiling and rank bakes by mean SROCC. |
| `v0_20b/` | Contrastive pre-train + fine-tune scripts (PyTorch). Su 2023 distortion manifold spike. |

### Corpus prep + parquet

| Script | Role |
|---|---|
| `build_unified_parquet.py` | Merge per-codec CSVs into a single unified parquet store. Mostly used for the interactive site backend. |
| `export_human_corpora_to_parquet.py` | Export KADID / TID / CID22 / KonJND to parquet with consistent schema. |
| `export_aic3_to_parquet.py` / `export_aic4_to_parquet.py` | AIC-3 / AIC-4 EPFL JPEG-AIC corpus → parquet. |
| `aic3_pairs_csv.py` / `aic3_anchor_pairs_tsv.py` / `aic3_jnd_sanity.py` | AIC-3 specific helpers (pair list extraction, sanity-check JND values). |
| _(deleted 2026-07-29)_ `convert_features_bin.py` | DEAD: its consumer (`train_v_next_mlp.py --human-csv`) no longer exists; the ZSFC reader it reimplemented lives in `zensim-validate/src/main.rs`. |
| `band_balance_safesyn.py` | Audit / rebalance safesyn rows by quality band. |
| `content_class_explore.py` | Cluster safesyn refs by feature embedding to inform content-class training experiments. |

### Baking + calibration

| Script | Role |
|---|---|
| _(deleted 2026-07-29)_ `affine_calibrate_bake.py` | Duplicate of the Rust `affine_calibrate` bin (`zensim-validate/src/bin/affine_calibrate.rs`, exists since 2026-06-18) — use that. |
| _(deleted 2026-07-29)_ `bake_to_znpr.py` | Emitted BANNED ZNPR v2; its trainer (`train_v_next_mlp.py`) no longer exists. Bake construction goes through `zenpredict bake <input.json> <output.bin>`. |

### Eval + analysis

| Script | Role |
|---|---|
| `verify_mohammadi_anchor.py` | Verify our logistic rescale reproduces Mohammadi 2025 Table 2 anchor values (PSNR-Y / IW-SSIM / CVVDP Z-RMSE on AIC-3 CTC). Companion test exists in `dataset_metric_baseline.rs`. |
| `analyze_score_quality.py` | Per-quality-band score-distribution analysis. Inputs: per-pair eval CSV. |
| `apply_butter_filter.py` | Filter pairs by butter-vs-ssim2 concordance. Pre-process step for noise-reduction in the synth corpus. |
| `butter_concordance_audit.py` | Audit how butter and ssim2 disagree across the safesyn corpus. Informed the cycle-7 dssim experiments. |
| `regen_tv_pairs.py` | Regenerate the TV-regularizer pair indices TSV when the training CSV changes. |

### Sweep / chart / site

| Script | Role |
|---|---|
| `gauntlet.py` / `bandwise_dashboard.py --fulleval-dir` | Build the current offline summer-gauntlet board from stored verdicts. Per-codec q-vs-score charts share a score axis spanning at least 0–100 and all visible medians, including negative scores; toggling bakes rescales it. Run `gauntlet_gates.sh <html>` after every regeneration. |
| `build_site_data.py` / `build_scatter_data.py` | Generate the per-corpus data files for the interactive comparison site at `https://imazen.github.io/zensim/`. |
| `make_v02_v18_candlestick.py` | The V_2 → V_18 candlestick chart used in commit / handoff narratives. |
| ~~`generate_v16_chunks.py` / `launch_v16_sweep.sh`~~ | **Deleted 2026-07-15.** Hand-rolled chunk splitters — the thing CLAUDE.md's fleet rule forbids outright. Superseded by the zenfleet job system (`zenmetrics/scripts/jobsys/`); never produced output; marked "reference only" here for 10 weeks, which is the queued-for-removal anti-pattern rather than a reprieve. In git history if needed. |

### Training and retired experiments

| Script | Role |
|---|---|
| the supported MLP trainer | `zensim-validate/src/bin/zensim_mlp_train` (Rust). Python prototypes may invent new recipes; admitting a model requires the full Rust serving/evaluation path described above. The older `train_v_next_mlp.py` was deleted in `34f796f4`. |

On September 7 the obsolete July blend and negative/diverse training families
were retired along with their private report path. Their source, recipes and
instrument definitions remain at [revision `45e1ec9a`](https://github.com/imazen/zensimx/tree/45e1ec9a95f0cbbb58decef46155aa46d6d1b8b6/scripts/v_next).
This is archival retirement, not a claim of same-recipe training parity with
Rust. Stored results and bake inputs remain intact. `bandwise_dashboard.py`
now forwards to `gauntlet.py`; its old NPZ/Matplotlib mode is retired. The board
reads recorded composites and leaves absent values unmeasured. Four historical
spline writers remain pending recipe and runtime-boundary comparisons.

### Planning notes

| File | Role |
|---|---|
| `CYCLE_7_DSSIM_COTRAIN_PLAN.md` | Cycle-7 design doc for the dssim co-training experiments. Falsified — see `benchmarks/cycle_7_dssim_outcomes_2026-05-12.md`. Kept for historical record. |

## Conventions

- **Output paths**: scripts that produce data write to `/mnt/v/output/zensim/<theme>/` or to a script-specific path documented in the script's header. Don't write to repo `benchmarks/` from scripts unless the script is generating a permanent artifact.
- **Parquet preferred for >50 MB**: zstd-3 compression per CLAUDE.md "Parquet vs TSV". CSV/TSV is fine for <50 MB human-readable inputs.
- **JSON pipeline for bakes**: ad-hoc Python wire-format emitters are banned per CLAUDE.md "JSON pipeline mandate". All bake-side serialization goes through `zenpredict-bake <input.json> <output.bin>`.
- **Logs to /tmp** for one-shot runs; commit to `benchmarks/<name>_<date>.log` for runs producing ship-relevant data.

The Rust screen loader now supports `--auto-transforms-max-features N` (stable
top N by lift) and `--auto-transforms-max-feature-idx N`. To replay the retired
V_20 converter, use `--auto-transforms-min-lift 0.05
--auto-transforms-max-feature-idx 228`; a top-N value of 0 keeps all eligible
rows. Non-finite lifts are rejected from selection. Historical recipes and the
converter's source remain in revision `7cbc2458`.

### Calibration owner (September 7, 2026)

Use `bake_dial_refit pack` for quantize-then-calibrate, or `add-spline` for a
pre-calibration surface fit (`--replace-existing` for a refit). Always name the
training anchor. The four May/V47 Python writers and V9 full pipeline are
archived; see [recipe and boundary evidence](../../benchmarks/spline_owner_cleanup_2026-09-07.md).
