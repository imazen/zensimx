# Develop, train and evaluate zensim

## October 27 active owners

Start with [the current goal and run index](../SESSION-RESUME.md#current-work--october-27-2026),
then `just mlic-priority-progress`. These additive owners implement the current
registered Mlic/AIC comparison; the September owner map remains background.
Read frozen plans/sources rather than relaunching live work from moving main.

| Work | Owner | Required evidence |
|---|---|---|
| Study/process supervision | `scripts/byconstruction/{dataset_study_status,run_processes}.py` | Actual outputs, PID plus start ticks, all thread children, real lock holder, failure sentinels |
| Native confidence source screen | `scripts/byconstruction/confidence_admission.py` | Plan-v2, calibrated source match controls, immutable source authority; not response permission |
| Native fitting populations/uncertainty | `scripts/byconstruction/{confidence_population,confidence_audit}.py` | Original observation IDs and source rows, exact memberships, finite native uncertainty; no fitted linking map yet |
| Order matching | `scripts/rev4_featpot/matched_codec.py`, native `zensim_mlp_train/matched_codec.rs` | Paired original Mlic prediction identity; separate omission/full schemas; unchanged draw law |
| AIC fold continuation | `scripts/byconstruction/aic_adaptation.py` | 48 bound initial models, zero-step parity, 96 final continuation cells, role/fold/seed identity |
| Fold statistics | `scripts/byconstruction/{aic_adaptation_panel,aic_adaptation_diagnostics}.py` | Native full statistics, fitted/held-out isolation, source-level uncertainty, canonical mixture cache checks |
| Full recipes/continuation | `scripts/byconstruction/aic_adaptation_full.py` | Full-budget binary control; 12 initial + 24 continued models; fixed seeds and both strengths |
| Full serving-model panels | `scripts/byconstruction/aic_adaptation_full_panel.py` | Final-f32 versus packed/calibrated output, 360 predictions, mixtures, complete inventories; no automatic winner |

Confidence admission/calibration metadata owners are in the separate private
zenpapers repository. Their purpose-bound response adapter and zensim's native
contrast-confidence loss remain unfinished. A metadata declaration, source
screen, completed fit, or TRAIN report alone does not qualify a product.

**Controlling work order (September 15):** follow
[the production priorities](PRODUCTION_PRIORITIES_2026-09-15.md) before older
task lists. It preserves the completed recovery and assigns concrete next
experiments to the owners below. The separately assignable
[Squintly study handoff](https://github.com/imazen/squintly/blob/main/docs/STUDY_READINESS_2026-09-15.md)
owns paid-human readiness, phone pixel zoom and display qualification. Do not
reuse historical study assumptions as current launch evidence.

Current owner map and workflow, updated September 15, 2026. Start from
[SESSION-RESUME](../SESSION-RESUME.md). The product is one target score with
useful quality across codecs/content and affordable encoding loops. Read
[CODEC_TARGET_METRIC](CODEC_TARGET_METRIC.md) and
[MODEL_SELECTION_SCORECARD](MODEL_SELECTION_SCORECARD.md) before selecting
objectives. B/C/D can be replaced; good rank alone is not product readiness.

**September 14 clarification:** [published TEST assessment when no EVAL exists](DATA_SPLITS.md#september-14-clarification-test-evaluation-when-no-eval-split-exists)
supersedes the blanket TEST ban for frozen assessments. Freeze compositions and
gates before reads, record exposure, prevent adaptive tuning, and leave secret
holdouts untouched. Historical admission packets retain their original rules.

**September 13 split override (historical):** train only / eval only / never test
supersedes older terminal-read and checkpoint-dev instructions below. The
feature-screen v1 recipes are historical and now refused. Use the explicit
[v2 segment contract](FULL_EVAL.md#strict-train-eval-feature-screens-september-13),
with training-only checkpoint monitoring and post-fit eval gates. Do not invoke
legacy default corpus/terminal scans to fill missing evidence.

## One owner per task

| Task | Owner | Required evidence |
|---|---|---|
| Feature definitions / read-set planning | `zensim::feature_defs`, `feature_set_id`, `Plan::for_bake` | Actual consumed IDs versus canonical extraction; era, SIMD, geometry and feature-build checks |
| New candidate inference | `zensim::BakeScorer` | Complete heads/splines/composition; cached-row and pixel/HDR parity |
| Native decoder color/depth inspection | `shared/zen_decode::decode_native_bytes` through `verify_bitstream_decode --inspect-list LIST --out FRESH.jsonl` | Explicit file hashes, native active-row hashes, source metadata separate from decoded descriptor/context, legacy RGB8 replay. [TRAIN evidence](../benchmarks/native_color_admission_2026-09-14.md); not a color transform or model qualification. |
| Native SDR input/extraction | Existing `extract_features_372col --input-contract sdr-native-clip-v1 --audit-jsonl ...`, private `shared/score_input` adapter | Exact native u16 codes or linear f32 plus declared primaries; arbitrary ICC through full CMS; public scalar/cache/spatial parity; separate audit-v2/input era; explicit unsupported/HDR refusals. [Contract/results](../benchmarks/native_sdr_contract_2026-09-14.md). Original encoder provenance still required. |
| Complete ensemble pixel/spatial audit | `extract_features_372col --audit-ensemble <ordered paths> --audit-ensemble-weights <weights>` through `BakeScorer::ensemble` | Exact member/hash/weight identity, pixel/cache and attribution score parity; separate density/refinement coverage; max rectangle reconstruction; unsupported terms block native steering |
| Complete canonical input audit | Same extractor audit, `BakeScorer::consumed_feature_ids` | Require `feature_audit_scope: complete-structural-read-set-v1` and measured IDs; compare every active member/companion input against canonical extraction. Older no-head zero maxima do not prove feature parity. [Evidence](../benchmarks/complete_feature_audit_2026-09-14.md). |
| Cached pair identity / corruption screen | `BakeScorer::score_features_with_identity`, optional audit in `extract_features_372col`, `corruption_gate_eval.py --audit-jsonl` | Proven decoded identity, canonical consumed-feature and stored-f32 parity, complete keyed coverage; raw zero features are not identity proof |
| Same-pixel SSIMULACRA2 comparison | `extract_features_372col --audit-jsonl PATH --audit-ssim2` | Existing fast-ssim2 on exact legacy RGB8 or canonical native SDR linear-float buffers; original and transformed hashes, finite scores; no RGB8 projection for native input, no HDR oracle claim |
| Prepared SDR/HDR maps | `BakeScorer::prepare_steering` / `prepare_steering_hdr`, retained basic/peak extraction (SDR also v2 f372-f719, see [STEERAPI](../benchmarks/steerapi_2026-10-02.md)) | Bind input contract once; complete canonical feature/scalar/refinement parity; bin8 finite moments are opt-in; report map cost and memory. [Recovery evidence](../benchmarks/recovery_completion_2026-09-15.md). |
| Corrected native HDR extraction | Existing `hdr944_extract`, `--audit-composition` | Declared PQ cICP, actual JXL codestream metadata, live peak columns, canonical cached/pixel/prepared parity; never relabel old HDR tables |
| Native HDR human assessment | Existing `upiq_pu_score --composition ... --input-contract upiq-exr-bt709-nits-v1`, `panel` | zenexr native float nits through complete `BakeScorer::compute_hdr`; original UPIQ split and frozen composition; matched BHdr control |
| Native common-primary HDR judges | Existing zenmetrics `score-pairs --hdr --hdr-common-primaries` | Versioned PQ-only ingress; BT.709/P3/BT.2020 to unclipped common nits before native metric; CVVDP display contract explicit; no 8-bit shell/sidecar fallback |
| Named-profile inference | `zensim::Zensim` | Profile/bake identity and supported pixel contract |
| Spatial attribution / rectangle queries | `zensim/src/attribution.rs`, retained extraction in `feature_v2` | `BakeScorer::compute_with_ref_and_attribution` binds complete candidate scoring; signed density with L8, separate finite-max `ScoredAttribution::refinement_gain` and coverage, scalar/feature parity and reusable binned sessions |
| MLP training / capability admission | `zensim_mlp_train`, `mlp_train::capabilities` | Explicit recipe, table declarations, source-disjoint selection, reproducible random streams |
| Historical corruption head replay (not an approved new-data default) | `train_corruption_head.py --canonical-manifest`, existing ZCTH exporter and `corrhead_parity` | Complete admission; source/pixel keys; explicit head IDs within a declared regime; separate fit/calibration; exact exported single-fit Rust evaluation; `--prepare-only` never fits and `--training-screen-only` never scores validation |
| Strict train-only integrity head fit | `train_corruption_head.py --strict-train-manifest` | Original v1 Rev1 recipe; explicit v2 Rev3 D228 recipe keeps fit/development/calibration disjoint, audits complete Rust composition and native prepared maps, and blocks advancement on incomplete registered coverage. [Pilot and color admission limits](../benchmarks/integrity_rev3_train_2026-09-14.md). |
| Native SDR integrity head fit | Same strict trainer, `integrity-head-train-v3` | Native diagnostic-admission-v2, Rev3/sqrt/D228, exact presented color identity, complete consumed-feature coverage and canonical f32 row hashes; round-trip CSV parsing. Final audit keeps native pairs-tsv. Unresolved labels refuse before payloads. [Contract and replay](../benchmarks/native_integrity_admission_2026-09-14.md). |
| Tagged SDR corruption fixtures | Existing `m3_fixture_gen corruption --input-contract sdr-native-clip-v1` (features `m3-fixtures,zen-decode`) | Full native interpretation followed by explicit clipped RGB8 delivery, v2 provenance; default v1 unchanged. [Contract and checks](../benchmarks/integrity_color_resolution_2026-09-14.md). Not native HBD/HDR fixtures. |
| Severity-aware integrity assessment | `corruption_gate_eval.py --integrity-admission --audit-jsonl --out-json` | Exact decoded identities, known-label versus unlabelled duplicate handling, honest activations by codec/content, real-bug recall and unresolved disposition inventory |
| Prepared steering with integrity check | `BakeScorer::prepare_steering`, extractor audit with `ZENSIM_AUDIT_PREPARED_STEERING=1` | Active head returns `CorruptionDetected`; inactive map equals perceptual branch; restoration after failed calls; finite queries do not establish native RD |
| Scatter geometry / raw density / envelopes | `zenstats::scatter::diagnose`, `panel --scatter`, `bake_verdict::scatter_assessment` | Complete population before plot sampling, tie-aware quantiles, raw density beside normalized shape, p99/max and robust tails, missing peer/range evidence is INCOMPLETE |
| Native content overlap screen | `check_holdout_overlap --native-png` or `--native-linear` | Explicit hash era, exact counts, source color interpretation and file identities, full close-pair review; unsupported formats leave admission incomplete |
| Transform selection | Trainer `--auto-transforms` | Declared score/method, top-N and parameter bounds; preserve independent screen references |
| Serialize / inspect ZNPR v3 | `zenpredict-bake` | Versioned metadata and final artifact hashes |
| Serialize tree corruption companions | Existing `train_corruption_head.py::emit_zcth`; Rust `CorruptionHead` and `BakeScorer` admission | Legacy ZCTH v1/v2 require native Rev1 features; explicit-revision v3 requires f32 inputs and matching base arithmetic. Opt-in v4 binds all numerical bytes to a verified TRAIN/decoder admission; `bake_verdict` checks its per-table files and scoring compatibility. Legacy heads remain readable but cannot acquire admission from arbitrary metadata. `verify_corrhead_format.py` uses synthetic rows and the Rust parity owner; a format test does not qualify trained weights. |
| Quantize / calibrate / densify | `bake_dial_refit` | Quantize before calibration; spline-coordinate and boundary gates; final served bytes |
| Full evaluation and reuse | `scripts/run_full_eval.sh` | Content-bound input/scorer identity, atomic stages, no stale reuse |
| Training-only base preference screen | `rd_probe_analyze_2026-07-18.py --model-preferences`, optional `--family NAME --advancement noninferior` | Explicit family/rule registration, D plus all three seeds, unchanged source/model/pixel/judge identity guards; legacy nine-model strict mode remains unchanged. Passing scalar preferences does not qualify spatial guidance. |
| Arbitrary metric/dataset diagnostics | [`panel evaluate`](METRIC_EVALUATION_CLI.md), shared panel/scatter/forced-choice owners (`zenstats`) | Explicit pinned CSV/TSV/Parquet inputs, declared direction, stored or executable scores, missing-capability reporting and bound specialized instruments |
| Rank/dial/G-ADDR statistics | `bake_verdict`, `zenstats` | Final Rust-surface scores, complete panels and named floor ruler; `srocc_ci` resamples references (`srocc_ci_method`), never pairs |
| Evaluation statistics: intervals, pair classes, forced choice, DS-AUC, JND maps, Holm | `zenstats` = zenmetrics-stats (zenmetrics `crates/zenmetrics-stats`, pinned in `Cargo.toml`; method guide `docs/EVALUATION_METHODS.md` there): `resample` (`spearman_interval`, `cluster_bootstrap`), `pairs` (`ds_auc`, pair classes), `forced_choice`, `jnd`, `multiple`; reached through `bake_verdict`, `panel`, `panel evaluate`, `bake_compare` | Source-clustered `Design` for any claim about new images, with design, B, seed and the one percentile rule (Hyndman–Fan type 7) recorded; not-sure counts half for each side; Holm over paired clustered differences. Never re-implement one in zensim or a script; Python stays an independent reference, and pre-2026-10-11 figures keep their own labels (pair bootstrap, order statistics, not-sure dropped) |
| Coherence | `scripts/m3a_sweep.sh` through full-eval | All registered cells; partial sweeps remain incomplete |
| Complete candidate finite-pixel diagnosis | `diffmap_block_coherence --ensemble ... --ensemble-weights ... --json NEW`; `--refinement-analysis INPUT --json NEW` reuses saved blocks | Exact model/pixel hashes; actual per-block served scores, M2/M3a/M3f, separate coverage and failed cells. `ZENSIM_ATTR_DIAG=1` saves per-feature deltas and family density contributions; saved analysis substitutes individual families. Oracles are never runtime inputs. |
| SSIM exactness reference and kernel cost diagnostic | `ssim_form::stable_ssim_plane`, `StableSsimScratch`; existing SSIM diagnostic also accepts `ZENSIM_SSIM_KERNEL_PERF` | Direct centered f64 accuracy, exact identity, isolated SIMD permutations, real-plane locality and bounded row-ring scratch. Served Rev3 uses fused direct-error moments; this separate f64 kernel is its exactness reference, and kernel timings do not qualify complete inference. |
| SSIM numerical precision / locality diagnostic | Existing ignored `streaming::tests::dump_ssim_moment_explosion`, `ZENSIM_SSIM_PRECISION_PROBE=<registered-case-json>` | Actual retained production planes and unchanged f32 XYB pyramids; direct centered f64 reference versus independent raw f64 algebra; canonical feature reconstruction and zero reference response outside changed-pixel support. Diagnostic only; new arithmetic needs an explicit feature era. |
| Research selection / product qualification | `freeze_check --select` / `--qualify` | Distinct claims; missing or failed product evidence never passes |
| Board rendering | `scripts/v_next/gauntlet.py` | Stored owner results; shared negative/positive axes, board gates |
| Contribution / real compute cost | `bake_contrib`, existing extraction benchmarks | Declared IDs; actual passes, buffers, time and memory |
| Frozen feature-profile latency matrix | `zensim/benches/extract_paths_bench.rs`, `ZEN_XP_MODELS`; optional `ZEN_XP_PAIRS` + `ZEN_XP_PAIRS_SHA256` | Complete Rust ensembles; reviewed TRAIN-development PNG byte/pixel hashes and native dimensions; strict resource gate, one actual call per round, saved paired samples. Synthetic 1MP/4MP inputs remain separately labelled. `D_current_revision` is current-formula timing context, not frozen historical D qualification. |
| Complete scalar ensemble cost | `zensim-bench/benches/ssim2_speed_bar.rs`, `ZEN_S2_ENSEMBLE` / `ZEN_S2_ENSEMBLE_WEIGHTS` | Canonical calibrated ensemble and member controls; `ZEN_S2_PREPARED=1` adds complete prepared base/head workers; `ZEN_S2_SINGLE_CALL=1` for latency bounds or `ZEN_S2_CALLS=N` for throughput controls; named hardware/threads, dispersion and strict contention admission; `ZENBENCH_RESULT_PATH` retains paired rounds with actual call counts; historical extrema/means are not p95 |
| Rev3 ensembles on retained native interventions | `diffmap_block_coherence --native-interventions MANIFEST --sha256 HASH --json NEW`; existing `gauntlet_spatial.build_native_gallery` | Hash-bound canonical TRAIN admission, fresh decoder era, complete scalar/peer scoring, exact transform unions, explicit partial density, native mechanism ranks separate from repair/RD gates. Frozen Rev1 D is an explicit matched control. |
| Native local TRAIN recipe comparison | Existing `zensim_mlp_train`, `subset_sim`, full944 extractor, `ensemble_score_rows`, native replay and panel/scatter owners; [September14 experiment](../benchmarks/native_local_train_2026-09-14.md) | Preserved product fit/development families, actual sampler digest/share, complete f32 packed API parity, matched scalar/native controls, explicit sparse-statistic nulls and failed advancement; never EVAL-driven recipe tuning |
| Native finite spatial interventions | Existing JXL `zensim_diffmap_rd --native-interventions --intervention-regions coarse4`, root `rd_probe_analyze_2026-07-18.py --interventions` | Native PNG v2 evidence, complete transform unions, actual q/pixel/score/byte effects, independent judges; mechanism evidence alone cannot qualify allocation |
| Native local allocation / exact scalar-state screen | Same JXL instrument, `--intervention-regions coarse-policy`, and the same root analyzer | Rational enumeration of all local raw-field states before the map; bounds hidden from policy; actual measured judge frontiers, zero-map identity, explicit failed screen; ordinary global-scale/distance controls remain separate |
| Actual target loop | `zensim-target::target_search{,_with_bake,_with_backend_and_bake}`, `SeedCurve`, `native_probe`, `demo_matrix` | Per-image witnessed bounds, frozen train calibration, 1/2/3 shots, encoded bytes, error/tails, time and independent judges; native map loops separately |
| Ladder encoding / extraction / training pairs | `canonical_corpus/build_ladder_grid.sh`, `build_dial372_instruments.py`, `build_ladder_tv_pairs.py` | Original source splits, encoder/decoder/feature identities, distinct floors, keyed row order |
| Detached completion | `harvest_bakes.sh`, `await_artifacts.sh` | Final evaluated artifacts, heartbeat, failure state and idempotent endgame |

Python may invent a method or provide an independent reference. Introducing a
model requires the complete Rust serving surface first; evaluation must execute
that surface. Do not implement a second owner to bypass unsupported metadata.
Inspect `../zenpapers` and later summaries before fundamental changes.

The [completed exact-project memory reconciliation](../benchmarks/claude_memory_chronology_2026-09-14.md)
records later corrections to clipping-induced ceilings, adaptively selected HDR
claims and stale codec-loop status. Broader recovery remains incomplete. Before
calling a feature set inadequate, distinguish preprocessing saturation,
optimization failure, calibration and absent information. ZNPR input support
uses `W[input * out_dim + output]`; a nonzero path is structural evidence, not
proof of functional dependence. Use a sparse asymmetric negative control when
checking layouts; dense random weights cannot distinguish the two orientations.

The [tree revision contract](../benchmarks/corruption_revision_contract_2026-09-14.md)
closes the missing arithmetic check for tree companions. Slot coverage alone
does not admit a Rev1 head to a Rev3 base. New heads must be refitted on matching
features; changing a header is not a refit. The strict integrity trainer's
v1 manifest remains a Rev1 recipe, and its historical head is unchanged. The
explicit v2 route serves the registered Rev3 D228 pilot; it cannot bypass
unresolved source-color coverage or source-family separation.
Fractional sampling still has no matching tree-head contract and is refused.

Native controllers stay in their codec repositories. The
[September 8 reuse audit](../benchmarks/diffmap_reuse_audit_2026-09-08.md)
indexes the already implemented JXL attribution loop, JPEG research workspace,
WebP segment loop and AVIF CQ attribution experiment, with later corrections
and negative results. Check that record before proposing another implementation.

Native galleries distinguish own-score map consistency from map association
with SSIM2/Butteraugli quality responses. The latter aggregate recorded Rust
cell correlations and require complete cell coverage. Neither establishes
native RD benefit; state whether the peers also supplied training supervision.
The [constrained-head replay](../benchmarks/native_constraints_2026-09-14.md)
demonstrates that improving own-score consistency can coexist with worse
quality ordering. The architecture was already trained; its native failures
do not justify repeating the same capacity/constraint sweep.

Every completed study must remain discoverable from the gauntlet. Append its
discussion entry to `benchmarks/board_discussion_sets.json`. TRAIN-development
studies use `role: train-development` with a served `report_url` and a concise
measured verdict; the board links their complete comparisons separately from
EVAL model filters. Do not synthesize qualification rows to make TRAIN models
appear in the EVAL table. Regenerate the board and run its browser gates.

`ZENSIM_ATTR_DIAG=1` on native replay retains the baseline sensitivities already
used for complete-model linearization. Their signed products with saved feature
deltas expose signal/scale/channel tradeoffs without another scoring owner;
retain the finite-head residual and do not present this as a feature ablation.
The [native contribution diagnosis](../benchmarks/native_contributions_2026-09-14.md)
checks unchanged ordinary output and exact decomposition on frozen TRAIN cases.

Native replay accepts optional per-case `sampling_phase_offsets` for the
[joint-translation diagnostic](../benchmarks/native_phase_2026-09-14.md).
It preserves all RGB pixels in a fixed black canvas and requires origin and
period-eight controls before pixel reads. Public scalar/feature/sensitivity
and bin8 map paths remain the owners. Padding changes context; sub-eight map
queries change bin alignment. Retain failed controls and separate these effects
from actual codec RD. A local phase response is not a replacement-model verdict.
Native galleries may attach `diagnostic_figures` with a local SVG/PNG filename,
title, caption and SHA-256; bad identities fail before publication.

The [four-contract filter comparison](../benchmarks/native_filter_2026-09-14.md)
uses neutral declared-ID ZNPR instruments to expose features through the same
Rust replay. Their scores are not trained quality predictions. Current direct
zenresize paths include binary16 input quantization; kernel, pyramid and
precision effects must not be conflated. Reduced phase span alone does not
select a sampling contract: the focal L4 span falls while mean span rises.
Refit against matching TRAIN extraction before assessing model quality.

The [quarter-chroma TRAIN refit](../benchmarks/chroma_scale_train_2026-09-14.md)
tests actual nested y60/y70/y80 read sets with matched H32 recipes and complete
panels. Added B improves rank and own-map consistency but fails the tail gate
and worsens Butteraugli map association. Preserve that distinction and the
failed timing admission. For exact historical control reproduction use the
later full-precision exports; the initial product fit TSVs round to six decimals.

## A complete experiment

1. Register the hypothesis, baseline, exact intervention, sources, seeds,
   evaluation instruments, noninferiority bars and stopping rule before results.
   First ask whether a proposed feature ablation removes compute; if it does
   not, give it low priority unless it resolves a consequential scientific issue.
2. Read the shared [provenance index](../../DATA_PROVENANCE.md),
   [DATA_SPLITS](DATA_SPLITS.md), [DATASET_HISTORY](DATASET_HISTORY.md) and actual
   file manifests. Pin SHA-256, row counts/order, IDs, formula and decoder eras,
   train/select/terminal reference identities and near-duplicate checks. Validate
   derived tables with the existing `validate_parquet.py` and split checker.
   Historical replay is explicitly unqualified; it is not permission to use
   overlapping validation as held-out evidence.
3. Build once through `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 ...`.
   Reuse content-pinned binaries through `ZL_*` / `CARGO_TARGET_DIR`; keep inputs
   immutable and use a fresh output directory. Current capped LAN/local work
   supersedes July Hetzner-first commands. Source dimensions and packing must
   match when comparing performance.
4. For stochastic recipes, screen with at least three paired, recorded seeds.
   Confirm finalists with five and the registered full recipe. A deterministic
   HGB fit with no sampling/early stopping uses one recorded seed; repeated
   identical fits do not add evidence. Separate initialization and sampled-data
   randomness; a seed spread is not proof of source/subset coverage. Tune only
   on the permitted selection data, retain failures, then evaluate final packed
   bytes through the surface. The frozen `bestofall_wave.sh` is a historical
   replay owner, not a default recipe for a new qualified model.
   September 14 correction: legacy sampling seeds are raw-stream offsets.
   Different digests do not prove disjoint sampling. For uniform-sampler
   comparisons, record well-separated `--sample-seed` values and preflight
   them together using `subset_sim --require-disjoint-sampler-windows` with
   the actual epoch/draw budget. This checks replay sampler windows, not all
   auxiliary trainer randomness; stratified schedules are explicitly refused.
   Metadata replay now prefers recorded sample seeds to legacy seeds. See the
   [robust native-pair study](../benchmarks/native_robust_train_2026-09-14.md)
   for real-draw overlap proof and unchanged historical explicit replay.
   Checkpoint selection is separate from early stopping: `--early-stop-patience
   0` still exports the best observed checkpoint. If every validation weight is
   zero, the historical trainer averages **all** group reporting scores,
   including zero-train-weight/zero-validation-weight groups. To keep an
   auxiliary TV-only group out of checkpoint selection, give the intended TRAIN
   selection groups explicit positive validation weights and select the intended
   policy (for example, equal weights with `--val-policy mean`). A group called
   `report` in the log is not proof that it cannot influence the fallback.
5. Run evaluation through the owner. A stage is reusable only if its full scorer
   composition, instrument options, inputs and binaries match. Never create
   corpus fixtures silently during evaluation. Completed verdicts survive a
   failed coherence stage; interrupted output does not become a reusable pass.
6. Compare the full rank/dial/tail/coherence panels and all per-codec floors.
   Label historical integrity/memorization guards. Selection cannot qualify a
   model: qualification requires measured G-ADDR plus the scorecard's separate
   product gates and matching provenance. Record failed, incomplete or qualified.
7. Follow the [current steering protocol](TARGET_STEERING_PROTOCOL_2026-09-08.md):
   establish attained bounds before judging steering, fit seeds only on canonical
   imazen-26 training families, and measure 1/2/3-shot encode/decode/score loops.
   Report target errors/undershoot, failure to converge, bytes, passes, latency
   and memory. Use independent judges and matched-quality RD comparisons.
   A model's own scalar or cached q curve cannot prove an encoding improvement.
8. Review, record results and limitations, run relevant checks, push through
   `scripts/safe_push.sh`, and verify. Preserve source/manifests/bakes/evidence.
   A negative experiment may close a hypothesis without removing the component.

## Commands and artifacts

For feature invention, start with the [bounded T2 development screen](FULL_EVAL.md#five-minute-feature-development-screen-september-13-2026):
`scripts/run_full_eval.sh --stage feature-screen <recipe.json> <fresh-output> [--cache <directory>]`.
It reuses the Rust extractor, trainer, BakeScorer audit and panel under a
300-second deadline; it cannot qualify a model. Build its owners once first.
It passes `zensim_mlp_train --no-auto-eval` to keep protected holdouts out of
the repeated development loop.

From the repository root, with explicit paths and a compatible feature root:

```bash
just check-mix path/to/recipe.toml
just check-data path/to/recipe.toml
scripts/run_full_eval.sh --stage verdict path/to/model.bin run-name 372 /path/to/features
scripts/run_full_eval.sh --stage coherence path/to/model.bin run-name 372 /path/to/features
scripts/run_full_eval.sh --stage qualify path/to/model.bin run-name 372 /path/to/features
# --stage all runs verdict + coherence; qualification is an explicit decision.
target/release/freeze_check --select /path/to/run.fulleval.json
target/release/freeze_check --qualify --fulleval /path/to/run.fulleval.json
just compare
```

Use the command's `--help` for supported options. Keep model hashes and final
composition beside the measurement; primary-file identity alone cannot bind an
ensemble or corruption head. `metric-eval` is a quick offline report.
`bandwise_dashboard.py` forwards to the gauntlet renderer. Neither is a second
source of scientific statistics.

For detached work, harvest as each bake lands and attach the bounded endgame
via `await_artifacts.sh --then`. It runs in the driver, not in a notification.
Success needs evaluated artifacts; failed stages leave nonzero exits and
failure sentinels. Check the terminal artifact rather than waiting for a log
line. An endgame may assemble evidence, but does not commit/push without review.
Do useful independent work while jobs run.

## Evidence and chronology

The [archived playbook](history/WAVE_PLAYBOOK-through-2026-09-07.md) preserves
August timing, orchestration incidents and old commands. The
[archived July protocol](history/ITERATION_PROTOCOL-july-2026.md) records the
screen/confirm rationale. Later split and scorecard rulings override their old
recipes. The [cleanup plan](PLAN_CRUFT_PURGE_2026-09-06.md) tracks execution and
links the confidence review, reproducibility gates and negative results.

The three-seed [training reproduction](../benchmarks/cleanup_training_reproduction_2026-09-07.md)
proves preservation of a competitive historical ranker. It also retains known
identity/floor failures. The [paired controls](../benchmarks/cleanup_scientific_controls_2026-09-07.md)
answer separate data/constraint/target-loop questions; none inherit a pass from
the reproduction. Keep measurement claims bounded to the actual experiment.

E34's paired reference × seed comparisons run through `panel e34` and the
approved zenmetrics-stats pin. Its [input and statistical contract](METRIC_EVALUATION_CLI.md#e34-paired-seed-comparisons)
keeps seed pairing, two-way/reference/seed components, directional and Student
p-values, all nine Holm entries and report-only pair classes in the Rust
owner. The summary-only E33 parity precedes the separately ledgered one-time
noise read. D12 still controls any future steering-sensitivity amendment.
