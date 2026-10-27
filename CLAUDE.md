# zensim — current working rules

## Current continuation — October 27, 2026

Read the current section of [SESSION-RESUME.md](SESSION-RESUME.md) before the
historical chronology. It records the active goal, frozen queue identities,
completed and unfinished science, exact status command and recovery order.
`just mlic-priority-progress` checks real processes, locks and artifacts; a
queued stage is not a completed experiment. Continue after negative results.

The owner's AIC refinement compares Mlic/Neff/Vote/Order (Confidence after
admission) with complete panels, including in-sample results, without strict
SROCC rejection or an automatic winner. Historical cutoffs/verdicts remain
unchanged. Preserve source roles, numerical/pixel correctness and provenance.
Production qualification and stable adoption precede the queued seamless HDR
frontend; maintain speed and compare against PU21 rather than assume it wins.

Dev compute is reserved for zensim training/evaluation. No benchmarks or
unrelated builds there; do not use r7900x for this work. Resolve live locks
before work on another eligible host. A stopped agent requires immediate
job inspection and supervision takeover, not an unverified “running” report.

## Controlling user priorities — September 15, 2026

Follow [the production execution plan](docs/PRODUCTION_PRIORITIES_2026-09-15.md)
before any conflicting older task list or paused goal: close evidence gaps;
fix attainable targeting tails; recover local ranking on TRAIN; establish JXL
spatial value; train correct native HDR; qualify the integrity head; finish
runtime and product gates. Preserve the fast Rev3 and richer controls and the
completed recovery work. Use the existing owners and tight local checks.

The [Squintly study handoff](https://github.com/imazen/squintly/blob/main/docs/STUDY_READINESS_2026-09-15.md)
owns the separately assignable paid-human task. Consult `../zenpapers` and
primary sources; qualify phone color and unsmoothed pixel zoom, distinguish
normal and magnified judgments, and require verified display behavior for HDR.
Do not start full paid collection from the historical September 1 protocol.
These are the user's current work priorities; they do not weaken data-split,
mathematical-correctness, provenance or registered qualification requirements.

Start with [SESSION-RESUME.md](SESSION-RESUME.md), then the
[wave playbook](docs/WAVE_PLAYBOOK.md). This file was consolidated on September
7, 2026. The [dated instruction archive](docs/history/CLAUDE-through-2026-09-07.md)
preserves prior findings and recipes; later dated corrections and the current
implementation take precedence over an old status paragraph.

## Product and implementation

The end user controls **one target score**. Quality includes content/codec
consistency, identity and near-lossless behavior, negative tails and distinct
codec floors, target accuracy, bytes, passes, latency and memory. Rank and a
research composite alone cannot establish a useful dial. B is currently
`codec_target()` and, since 2026-10-19, names the frozen diff+frag-820 Rev5
bake (`89cb4861…`), without a corruption head or JND map. See the
[promotion record](docs/PROFILE_B_DIFF_FRAG_820_2026-10-19.md); the [integration guide](docs/CODEC_TARGET_METRIC.md) owns the
profile mapping. No consumers are calibrated to B/C/D; they may be improved
or replaced, with explicit evidence and versioned experiment identities.

A new model must execute and serve entirely in Rust through a zensim surface
API. Evaluation must call that API, including every head, corruption gate,
spline and ensemble/routing step. `BakeScorer` is the dynamic candidate surface;
`Zensim` owns named profiles. Python is useful for invention and independent
references. Model comparisons lead with the registered composite coverage,
full rank/band/within-image panels, scatter geometry and tail/dial/spatial gates.
Raw MAE is an auxiliary calibration diagnostic, never a replacement for that
selection evidence. Full-population scatter diagnostics come from
`zenstats::scatter` through `bake_verdict`/`panel`; plot sampling and rank
normalization must not hide raw density, tails or saturation. Missing evidence
is INCOMPLETE, not a pass. Canonical ownership is a maintenance decision, not proof of
correctness. Keep independent numeric references with real parity gates.

Before adding an implementation, find its existing owner in the playbook and
extend it. Do not create a second scorer, calibrator, statistic, report
pipeline or controller under a new filename. A gated mirror must state its
independent contract, test real implementations and include negative controls.
Feature ablation is low priority unless it removes actual extraction work or
settles a named consequential scientific question. Consult `../zenpapers`,
especially `docs/zensim-720-feature-gaps-2026-07-26.md`, and later summaries
before fundamental feature/model changes.

## Workspace and changes

Follow [the parent AGENTS.md](../AGENTS.md). Work in this existing checkout;
no new worktrees, cleanup of another lane, or reverting unexplained dirty files.
Preserve user changes and immutable experiment evidence. Sibling repositories
are separate projects; do not modify them incidentally. Find instructions in
the target project before any separately authorized sibling work.

**Push only through `scripts/safe_push.sh`**, including explicit revisions:
`scripts/safe_push.sh -r <reviewed-revision>`. It fetches, requires ancestry,
checks outgoing identifiers, pushes and verifies. Never bypass its gates with
bare `jj git push`, bookmark pushes or force. Resolve divergence by inspecting
and preserving both histories. Keep private fleet addresses/identifiers in
private configuration; `just lint-scripts` checks tracked files too.

Before new public items or intentional API changes, preregister the concrete
caller and exact delta. Prefer private items; preserve supported APIs unless
the user authorizes a change. Update changelog and API snapshots together.
Run `just api-doc-check` for public/hidden-public edits; its nightly pin is
shared with CI. Run appropriate semver checks for supported API/release work.
`zensim` and `zensim-regress` have separate semver; validation and target tools
are unpublished. Publishing requires explicit authorization, passing checks
and a matching pushed tag; this cleanup does not authorize a release.

## Scientific data and reproducibility

**September 14 user clarification:** train-only for training, transforms,
calibration, feature/hyperparameter and checkpoint selection. Use EVAL when
available; when no EVAL exists, published TEST may assess frozen candidates
with recorded exposure and guards against adaptive tuning. CID22's oracle
training references and human test references are distinct. Secret holdouts
remain untouched. Preserve original roles rather than renaming TEST to EVAL.
Use explicit split admission; the legacy checker scans terminal tables and is
not an allowed default for this work. See the current DATA_SPLITS override.

Before training, baking, feature extraction or picker work, read
[`../DATA_PROVENANCE.md`](../DATA_PROVENANCE.md),
[DATA_SPLITS](docs/DATA_SPLITS.md), the relevant later entries in
[DATASET_HISTORY](docs/DATASET_HISTORY.md), and the recipe's file manifests.
Pin original bytes, decoder commits, extraction/formula revision, feature-set
identity, input hashes/order, source splits, seed streams and tool binaries.
Never infer semantic compatibility from column count or a familiar path.
The September 6 safesyn AVIF decoder era is incompatible with R6b inputs.

CID22 human scores and its 49-reference gold set are holdout-only. Metric
labels on the separate training references do not authorize human-label use.
AIC-3/AIC-4 and other T0 content stay holdout-only. Use current reference-level
splits, including KonJND/KADID; historical overlapping rows are integrity or
memorization guards, never held-out evidence. TID is train-only under the
later August 29 ruling. The canonical origin split is owned by
`../zenmetrics/scripts/picker/origin_split.py`; derivatives inherit the origin.
Use the existing near-duplicate audit; dHash flags need contextual review,
not automatic quarantine. Historical replay records limitations explicitly and
cannot qualify a new model.

Feature IDs and arithmetic revisions are explicit. `feature_defs`,
`feature_set_id` and `Plan::for_bake` own feature families/layout/planning;
validation's registry describes historical producers. No silent truncation,
zero fill or width-based family inference. Dense/gapped tables are refused by
loaders that do not support them. A bake's known primary leg cannot hide an
unknown companion or training leg. Pixel scoring currently requires the
process luminance revision to match the bake; do not silently mix revisions.

Use ZNPR v3 via `zenpredict-bake` and the Rust pack/refit owner; do not introduce
v2 writers or hand-serialize model layouts. Keep the pinned git/path dependency
for zenpredict; do not substitute the old crates.io v2-only implementation.
Quantize before calibration; evaluate final packed/declared-ID bytes. Removing
an exactly dead input differs from folding a corpus-constant input; preserve
those distinct guarantees. Emit structured manifests, not TOML string surgery.
Use Parquet for large tables; preserve row keys through joins and extraction.

## Compute and verification

Run heavy builds, tests, training, extraction and benchmarks through
`~/work/zen/scripts/run-heavy --mem 16G --jobs 8 ...` with workload-appropriate
explicit caps. Current LAN/local execution supersedes July Hetzner-first
recipes. Reuse verified binaries when only consuming results; record their
hashes. Do not create duplicate builds or leave failed chains marked complete.
Use the existing harvest/await owners for detached chains; final artifacts
and failure sentinels, not an agent notification, determine completion.

Benchmarks need a quiet machine, pinned geometry/threads/build, before/after
runs and dispersion. Record competing processes. Do not claim a speedup from
contended timing, coefficient counts, a narrower caller vector or a smaller
loop bound that still packs the whole image. Preserve buffered/full-feature
paths until actual callers and discriminating parity/performance gates justify
retirement. Planner evidence is in the September 7 feature-plan record.

Run tests appropriate to changed behavior, CI-exact `just clippy`, and
`just lint-scripts`. Product inference changes require feature-build serving
checks, canonical consumed-feature comparisons, candidate/pixel parity and
relevant HDR coverage. Use actual registered datasets and final bakes for
scientific claims. Full evaluation, selection and qualification are distinct
commands in the playbook. Failed, missing, stale and qualified evidence must
remain distinguishable in JSON and reports.

The authorized cleanup checklist and bounded acceptance evidence are in
[PLAN_CRUFT_PURGE](docs/PLAN_CRUFT_PURGE_2026-09-06.md). Record negative results
and unresolved product limitations; do not declare a model qualified merely
because cleanup tests or a historical training reproduction pass.

## Known Bugs

* **2026-10-27 — full-control raw hashes included run provenance.** AIC full run02 completed its control, then the new comparator rejected the embedded paths/build/run metadata. Native `strip --key zentrain.repro` on copies proves exact final-f32 and packed serving bytes; the recorded training curves match excluding elapsed time. The corrected comparator follows the existing E33 control owner and also compares all result fields except output paths/model hashes. Preserve failed run02 and dependent panel/confidence failures; recovery requires a fresh pinned plan and unchanged numerical equality.


* **2026-10-27 — confidence manifest logical width.** Binding run02 declared the 410 selected raw IDs as the model width, omitting the 1,853 raw-input extent plus 410 registered derived products. Caught before candidate fitting. The producer now binds the exact derived declaration and records logical width 2,263 separately; old manifests remain unexecuted and superseded. Fresh manifests and full-budget unweighted parity are required before confidence fits.

* **2026-10-27 — queued admission referenced an absent generic binary name.** The preserved reader is hash-suffixed, while the first confidence plan named `admission-packed` without binding it in `input_sha256`. Native decoding finished but receipt writing refused before a screen or response read. The coordinator now requires the actual packed reader as an existing, hash-bound input before queueing. Failed source-screen/AIC dependency attempts remain preserved; replacement plans retain scientific inputs and thresholds.

* **2026-10-27 — confidence omission source-row namespace. Fixed before admission.** The first metadata adapter treated every collection's `source_row_id` as an index into the KADID/TID source file. KonFiG and CID22 use collection-local row numbers; the reference-name guard refused before any output or response read. The repaired owner joins each collection's pinned bank `pair_key` and reference group, retaining native reference pixel hash/dimensions and observation IDs. Original manifest pins are shared through a passive constant module; their values and existing bank resolution are unchanged. Failed run01 is preserved. Run02 then refused canonical decimal-string family IDs; run03 accepts their exact numeric identity, rejects malformed IDs, and completes all omission metadata. Native family admission and purpose-bound response reads remain separate prerequisites.

* **2026-10-27 — zero-step AIC export selection. Fixed before continuation.** A new caller assumed `last.bin` for a zero-epoch control; the frozen owner selects `best.bin` and records its hash in `result.json`. AIC adaptation run01 was cancelled before fitting. Run02 follows that receipt, requires selected epoch -1 and byte-identical predictions; positive-step fits still require the final checkpoint. The frozen warm-start owner and old results are unchanged.

* **2026-10-27 — native metadata confused pixel identity with observation identity.** The first confidence population adapter used the promoted bank's representative filename for every fitting row sharing its pixel hash. Four distinct KADID observations consequently mapped to one native item. The audit was rejected before weights were used. The queued ordering trial was canceled before fitting because its admission also rejected repeated pixel hashes. Both adapters now preserve `source_row_id` alongside `pair_key`, bind native filenames through the original pinned source metadata, and reject duplicate observation identities. Frozen studies and their training loader are unchanged.

* **2026-10-27 — native-vote wrapper schema dispatch. Fixed before trial launch.** The Python strict group wrapper recognized v1/v2 subjective declarations but sent the admitted v3 native-vote declaration into the ordinary feature-table gate. Its dispatch now includes v3 and retains the independent authority and native-count checks. An invented Parquet/count fixture exercises the actual trainer-command wrapper; missing authority still refuses. No candidate fit used the broken wrapper.

* **2026-10-26 — threaded fit runners hide children from main-thread-only process walks. Fixed for dataset progress.** Linux assigns a child to the thread that spawned it. A walk of only `/proc/PID/task/PID/children` missed the live trainer launched by the cell executor. The shared `run_processes.py` owner now walks every task's children, binds PID plus start ticks, rejects zombies/recycled PIDs, and resolves the real compute-lock inode. Real threaded-child and zombie controls pass. A training log alone is reported as log presence, never proof that a process is running.

* **2026-10-26 — DIBR subset pair ingestion. Fixed.** The native table leaves `p_a` blank and stores row-preferred vote counts. The root reader derives the hard direction from the checked `n_a`, `n_b`, `n_tie`, `n` total; votes do not weight the loss. The partial root and TRAIN-read ledger remain preserved. Count-direction parity and inconsistent-total refusal are checked.

* **2026-10-26 — Lazy zero-moment loads after an Adam shortcut. FIXED.** The fixed-point shortcut must retain identity moment stores: on Intel, leaving initially lazy zero pages read-only took 7.806803397 s for the 20,000-step probe; retaining the stores takes 0.840776747 s with identical moment bits and zero FP assists. A per-call opaque +0 prevents dead-store elimination. See `benchmarks/adam_denormal_2026-10-26.md`.
* **2026-10-26 — Adam retains subnormal moments. OPEN.** A fixed-work probe of the fused v3 owner on Intel Core Ultra 7 265K produces 268,105,485 FP assists in 7.412144902 seconds when moments decay below the smallest normal f64; its normal-state control has zero assists in 0.104186039 seconds. See `benchmarks/adam_denormal_2026-10-26.md`. Any numerical repair is a new trainer version; frozen studies keep their existing binaries.

* **2026-10-25 — rank-cell progress display. Fixed.** The per-epoch rank-marker reader initially omitted its regex import. The corrected display passes against the twelve active mix cells, and undefined-name checks pass across the changed Python owners. Running fits and their statistics were unaffected.

* **2026-10-25 — dataset EVAL tied ranks.** The coordinator's new dataset evaluator used nested argsort, assigning different ranks to tied scores. The MOS report now uses the canonical Rust midrank and reference-bootstrap owners for every model and peer; synthetic tied-score and row-order controls pass. The Mdata assessment recomputes its M7/B7/peer controls with this method. The original coordinator outputs remain preserved.

* **2026-10-25 — Mdata trainer transport compatibility.** Admission initially refused the derived-input read set and width; after those bindings passed, fx1 refused new rows with1825 logical columns against the inherited1853-column trainer transport. New dataset tables now retain28 NaN tail columns, matching the existing root. The420 consumed measurements and original source root are unchanged; failed outputs remain preserved.

* **2026-10-25 — fresh dataset-root receipt serialization.** The first Mdata root assembly wrote all eleven TRAIN legs but refused its final receipt because inherited immutable roots were `Path` objects. The builder now serializes those paths as strings. The incomplete root and label-read ledger remain preserved; no frozen input was modified.

* **2026-10-24 — AVIF decode drift in the zensim-bench extractor. FIXED in the bench workspace; publish pending.** Built today (zencodec pin `d327434b`, the sibling codecs' rev; `4fda4391` no longer compiles with them), `extract_features_372col` decodes different pixels for 33,384 of 34,001 safesyn AVIF files than the 2026-10-23 build (`1152f194`), which reproduced the stored Rev5 bank. JPEG, JXL, WebP, PNG and BMP decode identically. Never mix AVIF rows from the two builds in one table; the linear-pyramid study re-extracts a control instead ([record](benchmarks/byconstruction_2026-10-22.md), "AVIF decode drift"). Against an exact float64 conversion of the same 10-bit YUV (ffmpeg as the YUV oracle; these files are CICP 1/13/6, full range, 4:4:4), the new decode is off by one on 4.54% of values (705,672 of 15,544,662 over 10 files, always ±1, at values a few hundredths of a level from a rounding boundary); the old decode equals the exact conversion bit for bit on 2 of the 10 files, where the new one is off on 0.015% and 0.057%. libavif (Pillow 12.3) is further off (up to 3 levels). The route is zenavif Rgb16 → `zenpixels_convert::RowConverter` → RGB8_SRGB (`zensim-bench/examples/shared/zen_decode.rs`). Cause: the 16→8-bit narrowing, not zenavif. The lock resolves registry zenpixels-convert 0.2.16, whose narrowing is not correctly rounded; rounding the same Rgb16 samples as `round(v · 255 / 65535)` reproduces the 2026-10-23 pixels on 10 of 10 files (`byc_avif_route`). The local, unpublished 0.2.17 rounds correctly (zenpixels `d019ae9`); patched into `zensim-bench/Cargo.toml`, the extractor reproduces the 2026-10-23 extractor bit for bit on 13 of 13 pairs. The stored training root holds the correctly rounded pixels. Remove the patch once zenpixels-convert 0.2.17 is published; any zen_decode build without it decodes 10/12-bit AVIF wrongly.

* **2026-10-24 — the v2 SSIM luminance term cancels in f32 (direct form, Rev3+). OPEN: a fix changes feature bits (new formula revision).** `nb − na = (m1² + m2² + C1) − (2·m1·m2 + C1)` stands for `(m1 − m2)²`; the walk (`feature_v2_kernels.rs`) and the map kernels compute it as the difference. On near-identical pairs the v2 SSIM means of the chroma channels at scales 1–3 are off by up to +336% and −87% against f64 on the walk's own moments; computing `(m1 − m2)²` directly agrees to 1e-5. Diagnostic: `map_split_tests::v2_ssim_luminance_cancellation` (ignored). B carries it. See the [record](benchmarks/byconstruction_2026-10-22.md), "Attribution-map correctness".

* **2026-10-24 — candidate attribution maps read the process revision. FIXED (`ec9289df` on `explore/byconstruction`; on main through `bab88fde` and the revision-7 merge).** `BakeScorer::compute_with_ref_and_attribution`'s pass B chose the SSIM form (direct error moment at Rev3+) and the ringing form from `ZENSIM_FORMULA_REV`, while the retained planes follow the bake's revision. B (Rev5) in a default process read its error-moment plane as a product moment; the map summed to about −1.8e19 on a 192×160 blur pair. Main has the same three `active_revision()` reads in pass B (not measured there). Regression test: `attribution_maps_serve_the_declared_revision_in_every_process`. Same session: the SSIM deviation-pool polynomials cancelled in f32 (up to 2.2% map error; now centred), and a Rev5/6 map omits the v2 features of scales 1–3 by design for steering (research map modes now keep them). See the [record](benchmarks/byconstruction_2026-10-22.md), "Attribution-map correctness".

* **2026-10-19 — strict workspace lint found deprecated SIMD constructors in training tools.** Token-explicit aliases replace their equivalent deprecated forwarders in `zensim-train-core/src/simd_encoder.rs` and `zensim-validate/src/adam_simd.rs`. Strict workspace Clippy passes; no arithmetic, dependency version or test expectation changes.

* **2026-10-19 — JND report identity still named retired linear B after the profile rotation.** Fixed the report owner's B artifact hash and checked current-B/no-map and retired-B refusal explicitly. Its two stale profile-map expectations failed platform CI; the reporting tests now exercise the approved unmapped default without fabricating JND estimates. The historical map's arithmetic tests remain in the library.

* **2026-10-19 — default B retains the frozen diff+frag-820 color-noise inversion.** Owner-directed promotion does not close the existing severe LIVE noise, dial or spatial gates. No corruption head or old-B JND map is attached. See [the B record](docs/PROFILE_B_DIFF_FRAG_820_2026-10-19.md).

* **2026-10-18 — PERF3 archive-helper CI path check (fixed).** Script lint interpreted the literal lane basename in an ownership assertion as a deleted sibling dependency. The helper now checks the supplied basename components with the same exact identity and lock-owner requirements. Keep lane names as input validation rather than embedded sibling paths; verify helper lint with no sibling checkout.


* **2026-10-18 — PERF3 final combined API measurement.** Binary `8fd20366` has no significant score regression in its nine-cell paired grid and improves all nine map cells. This is a separate build from the isolated-parent finding below; retain both measurements. See [the full report](benchmarks/perf3_2026-10-18.md).


* **2026-10-18 — PERF3 direct-retention candidate has a score trade-off (unmerged).** On r7900x, the shared-central-block candidate's 1024²/8T score has a significant +6.885% paired mean cost versus the frozen PERF2 rebuild. Its 256²/8T score and all nine retained-map cells improve; 54 complete output audits remain bit-identical. Keep the additional direct-core work in the cached scoring PR for review. Main's queue/fusion/SAT changes are separate. See [the implementation and scope](docs/PERF3_THREADING_DESIGN.md).


* **2026-10-18 — fixed color/noise repairs retain the LIVE severity inversion.** The registered TRAIN-screened fine-chroma three-seed ensemble scores severe manfishing noise50.0116 versus moderate−9.3503. Coverage and combined arms also retain this ordering failure. No production profile changed. See [the repair comparison](benchmarks/color_noise_repair_2026-10-18.md).

* **2026-10-18 — frozen E33/E34 models overrate severe LIVE color noise.** All three omit finest-scale X/B inputs. Large measured retained errors can also increase quality through signed affine/ReLU cancellation: on manfishing wn/img41, removing f136 lowers diff+frag49.7143→11.0847 and W-C57.9947→24.3266. All three rate this real severe noise above the less-damaged wn/img73. Native baseline replay and independent pixel/matrix checks pass; this is model behavior, not a repair. Keep these exposed examples out of fitting/selection. Evidence and limits: [matrix trace](benchmarks/gauntlet_live_color_trace_2026-10-18.md).

* **2026-10-18 — standalone discovery of a private panel module. FIXED.** A helper placed directly in `zensim-validate/src/bin/` was also discovered as a separate binary, where its module visibility was invalid. It now lives in `panel_rank_interval/mod.rs`; Cargo metadata confirms there is no extra target. Focused module tests passed; the complete CI rebuild validates target discovery.

* **2026-10-18 — Scalar target CI fixture dependency, fixed.** The integrity fixtures import Pillow; their new CI step initially omitted it and failed with five import errors. The step now installs the existing prerequisite in an explicit isolated environment. The exact17tests pass in a fresh environment; no assertion or numerical threshold changed.

* **2026-10-18 — G-TARGET AVIF backend selector.** The adapter passed `svt-rs`; frozen native driver925d832b accepts `svt` or `zenav1-svt`, defaulting other values to Zenravif. Earlier AVIF rows are preserved excluded diagnostics. Physical accepted-alias/420 proof and the corrected six-panel rerun are complete; all18 TRAIN-curve budget gates fail unchanged bars. JPEG/WebP reproduce exactly; numerical bars, models and q grids are unchanged. See `benchmarks/gtarget_2026-10-18.md`.

* **2026-10-18 — reachable-target report referenced an undefined target-only flag. FIXED.** The existing complete-matrix regression exposed a `NameError` in report metadata before any summary could be written. Pass the optional reporting flag explicitly; retain every numeric gate and negative-control assertion.

* **2026-10-17 — E36 second live timer extension also left an old expiry.** Native continuation exited124 exactly17:00Z despite the registered22Z clock; both resource helpers drained0 and11625 selected DONE/attempt1 rows were retained. The same-binary fixture and timer readback did not establish full-process enforcement. Fresh continuationv3 constructs its original timeout directly from22Z and audits24 actual runtimes; no further live timer repair, extra clock credit, matrix/source change or attempt reset. A bootstrap collision in fixed PID receipt names was repaired with per-output-version names before new cells launched. Final native/archive/pool admission remains pending.

* **2026-10-17 — E36 grouped archival interrupted recovery. FIXED in reviewed storage owner; final admission pending.** Prospective restart could bypass owner/capacity admission, require already-evicted array originals, or expose a partial JSONL row. Pack recovery now replays native-DONE membership and immutable owner assignments, verifies existing remote packs, checks fresh cache headroom, and drains the same receive process even when hold persistence fails. Atomic hash-named assignment/relocation segments and durable directory admission precede local unlink. Final freeze requires their source/intent/quota/directory proofs. Existing raw/zstd owners and all numerical/teacher bytes remain unchanged; this is not a completed dataset or fit release.

* **2026-10-17 — E36 live timer extension did not prevent the old expiry.** The native wrapper returned124 at14:30:16Z despite the registered17Z clock. The waiting qualifier then failed on an absent lock-window log directory. All9478selected native publications remain DONE/attempt1; no required failure was recorded. Recovery continues the same native IDs, ledger, CAS, worker and numerical startup under a fresh deadline wrapper. No matrix truncation, attempt reset, fit or extra time credit is authorized. Preserve the original failure/readback records; timer readback alone did not prove enforcement.

* **2026-10-17 — a post-creation STEER reservation race stopped supervision. FIXED.** Other containers can appear after the first capacity check. A refusal at the second check previously left an unreleased candidate behind its owned pause and propagated as a transport failure. The lifecycle owner now waits for that guarded candidate to exit cleanly, preserves its hold receipt, restores only the exact unchanged enrollment pause when no operator pause exists, and continues the other hosts. Invalid envelopes, changed ownership and bad exits still refuse. No capacity floor, scientific cap, claim or retry rule changes.

* **2026-10-17 — SHIPGATES source-class correction. CORRECTED INPUTS ADMITTED; RD PENDING.** The original gb82 subset has nine photographs and three game renders, rather than12 photographs. Preserve all original measurements, classify the three renders as screen/nonphoto and append the next three unused SHA256(path)-ordered photographs. The12-image class minimum is unchanged; the corrected27-source cohort passed native metadata and exact full-RGB admission; no photo-class RD qualification until its complete grid and judges are available. See [the report](benchmarks/shipgates_2026-10-16.md).

* **2026-10-17 — peer tests could share a scratch directory on one clock tick. FIXED.** macOS CI saw a peer hash mismatch followed by a missing-directory cleanup panic because fixtures used only PID and wall-clock time. Test-only paths now include a process-local atomic nonce. A forced same-tick test checks independent peer bytes and cleanup; validation and scientific executables are unchanged.

* **2026-10-17 — E36 provisional selector polarity.** The plain N output is `pin-g`, a higher-is-better raw score. The first private forward adapter named it a distance and reversed its selection subtraction. No E36 models existed and no lambda was selected. The adapter now names raw scores explicitly, requires their convention and compares better minus worse; native invented-row polarity and reversed-arm refusal checks precede final admission. Training already uses the existing score-polarity owner.
* **2026-10-17 — E36 preflight alias costs and base-filter prototype.** V4 cached encoded-byte aliases inherited canonical cell-total timing fields. Native command timing and numerical/artifact outputs remain valid; ladder sums from inherited fields are invalid. The bulk producer records actual alias execution and ladder wall time. The first metadata-only family-mask plan omitted canonical `safesyn:` member prefixes and therefore reported zero removals; it is invalidated, preserved and replaced by the strict version2 identity owner before any fit. No E36 optimizer step has run.

* **2026-10-17 — raw ledger footer counts cannot attribute installs after failed attempts. FIXED in the lifecycle status owner.** Footer rows include retained attempt history. Use `just steer-worker-scheduler-metadata` to read the frozen native compact reduction, with no upload or amnesty, and match its scheduler columns to the canonical installed counts. The live14-cell inventory matched6/4/4 across the three active hosts, with two failed identities retained separately. This is execution telemetry, not scientific assessment.

* **2026-10-17 — STEER could re-enroll after a cycle whose ledger listing preceded exit. FIXED.** A global cycle completion time, or a jobset result timestamp, can be later than worker exit even though the store listing was earlier. The lifecycle owner now requires a canonical harvest cycle whose start is strictly after that allocation exited; jobset metadata must belong to that cycle. Current fits and native job identities are unchanged.
* **2026-10-17 — host capacity refusal stopped the STEER controller. FIXED (`6f7203a4`).** Existing Docker reservations can prevent another allocation even when actual free memory is ample. The controller now keeps reservation, available-memory and fit-pool admission holds local to the affected host; other eligible hosts continue. Enrollment independently refuses before remote staging or control mutation if capacity changes while queued. Disk-floor and invalid-slot defects still stop supervision. All resource thresholds are unchanged.
* **2026-10-17 — interrupted STEER fits retain deterministic coverage scratch. Repaired registered retry admitted.** The frozen coverage writer requires fresh outputs, while its STEER temporary filename is determined by the full argv SHA. Interrupting a fit leaves that name occupied, so its unchanged native retry refuses before training. Preserve the exact manifest-bound temporary table, keys and manifest by rename while all STEER controls are paused and that host has no live STEER worker; never reuse the partial table or reset retry counts. The existing lifecycle owner provides `just steer-worker-preserve-scratch`, using Python in the pinned image. The repaired registered retry exited zero, passed native/archive installation and matched all nine installed files on NAS. The frozen image entrypoint lacks the staged source's four-line loop-top deadline-exit guard. The owner approved binding that exact separately hashed script as an execution amendment; the lifecycle owner requires the amendment and checks the read-only mount and in-container hash before release. Future allocations retain the original 150-minute wrapper budget and unchanged scientific cell/pass caps. The earlier one-minute between-pass workaround applied only to the admitted repair allocation. Image, executor, trainer and scientific bytes are unchanged.

* **2026-10-17 — concurrent MAPPERF waiters raced the activity-file rewrite. FIXED LOCALLY.** A queued diff-410 cycle saw the activity file during another same-owner write and refused before parity/timing. The lane adapter retains the ownership check and publishes activity with atomic replace; failed attempts remain preserved.

* **2026-10-16 — native JXL intervention decoder pixel parity. OPEN FINDING.** The representative primary/libjxl0.12 U8 comparisons in the SHIPGATES TRAIN control panel differ by at most1 channel level. All retained outputs decode, neutral repeats are exact, and control quantizer changes stay inside selected regions; this does not establish bit identity with libjxl. The responsible conversion/decoder stage is unresolved. Preserve both outputs and report the finding alongside RD evidence; no decoder fix or product qualification is claimed.

* **2026-10-17 — MAPPERF worker alias omitted from paired-coordinator inventory. FIXED in b2205962.** The fresh map audits passed, but the coordinator refused the first segment before collecting rounds because the before-worker alias was absent from its explicit declarations. The alias now names the same complete retained-output action in both registries. The failed attempt is preserved; no measurement or admission rule changed.

* **2026-10-16 — source-preparation matte rounding differs from exact IEC sRGB. OPEN.** The SHIPGATES white-matte diagnostic through zenpixels-convert0.2.16 differs on21 RGB channels by1 on the transparent codec-corpus GUI image. Its strict independent oracle fails and the prepared matte is excluded. The sweep instead admits a maximal all-opaque crop with exact byte equality and retained original family; no color-conversion fix is claimed.

* **2026-10-16 — full STEER manifests batch four cells before ledger publication. OPEN during launch; isolated repair proven.** The pinned worker interprets ZEN_CHUNK_WALL_SEC as a packing target, while the scientific program separately enforces its 7,663-second cell cap. At target7663, actual invented-job execution produces ten four-row sidecars; target1 produces forty one-row sidecars. Pause is checked only at pass start; SIGTERM does not flush an unfinished chunk. Stop each old container after its original chunk sidecar publishes, then verify/install/back up those rows before changing claim boundaries. Any newly started incomplete fit needs a full retry. The [runtime repair record](docs/STEER_RUNTIME_AMENDMENTS_2026-10-16.md) binds the one-cell scheduling and rolling claim-window plan without changing scientific identities, budgets or gates.

* **2026-10-17 — addressability report transport assumed a nonserialized count. FIXED.** The floor JSON exposes `represented_frac` and `n_ladders`, not `n_represented`. The first f32-control renderer refused after all six numerical replays; rendering retained owner fractions recovered the report without rescoring. Keep best-epoch and selected-final-epoch controls separate: all three production receipts select epoch 119, while the recorded best curve epoch is 0. Same-epoch f32 preserves all 97 observed inversions, so this failure is not caused by packing.

* **2026-10-16 — STEER pinned-image install crossed Docker bind mounts. FIXED; same-artifact transport replay passed.** The archive/native verifier accepted the first registered cell, but separate staging/result bind mounts made its atomic directory rename return EXDEV. Mount their common parent once; the image, archive, verifier and scientific state machine stay unchanged. The preserved first artifact verifies and installs in the same pinned image after the mount change, without another fit; all 24 launch/harvest tests pass.

* **2026-10-16 — matched batch inputs must supply physical row indices. FIXED in the ADDRFIX transport.** The fixed metric batch tool preserves input columns; it does not synthesize `row_index`. The first KADID scoring pass completed all 7,869 rows but the transport refused the missing index before any target assessment. Its original scores are retained; exact pair keys restore all physical observations only after proving duplicated pairs have identical scores. Subsequent inputs carry explicit indices. Numerical owners still refuse duplicate, foreign or missing indices.

* **2026-10-16 — candidate low-quality dial ordering regression. OPEN.** All three assessed Rev5 candidates miss the unchanged mentor-resolvable standard AVIF/JPEG floors and ladder AVIF-rav1e floor. The exact same-cell CVVDP-FHD and registered SSIMULACRA2 references pass both tiers. Removing only the candidate output curve leaves every represented fraction/order/clamp count unchanged; there are no clamp failures. The ADDRFIX investigation retains exact rung evidence and does not qualify a replacement model.

* **2026-10-16 — palette admission imports need NumPy/PyArrow in script CI. FIXED LOCALLY; remote check pending.** The scripts job at `febb0698` failed importing the real admission owner because NumPy was absent. Its palette step now installs the existing research dependency requirements in an isolated environment before running both unchanged unittest invocations. Both invocations pass in a fresh local environment (`just shipgates-palette-ci`); no data or labels are needed and no assertion changes.

* **2026-10-16 — peer engineering captures require the block size in their join key. FIXED in the PEERGATES adapter.** The first 96 broad cases have unique names, but later fixture names repeat at several block sizes. A name-only join selected a different block capture; the existing count assertion stopped the run at case 97. The adapter now joins `(case, block)`, requires all 135 distinct case/block pairs, proves the first 96 inputs unchanged and completes only the remaining 39. The failed attempt and all completed observations remain preserved; no case or numerical bar changes.

* **2026-10-16 — STEER completeness differed from the inherited E34 rules. FIXED in the versioned adapter.** Nested primary-seed/W2-seed diagnostic refusals no longer veto a valid primary comparison. Dataset/W2 numeric evidence is required on every comparison, including improvements; finite false diagnostic point guards do not veto S eligibility. Canonical invented-data checks cover both report-only refusals, invalid improvement evidence, all twelve Holm slots and the S-only verdict. The E34 evaluator/fixtures and numerical thresholds are unchanged. Q12 remains pending; this correction does not approve the alias policy.

* **2026-10-16 — inherited exact-target tie detection uses `signum`. OPEN for legacy groups.** Rust `f64::signum()` returns ±1 for ±0, so the inherited plain rank loop's `target == 0` check does not exclude exact ties. The separately admitted `subjective_rank` group now checks target equality directly and counts skipped ties; inherited control behavior is preserved. Synthetic actual-trainer tests show zero tie gradient, a nonzero inherited negative control, unchanged non-tied control outputs, no rank-leg MSE term and no between-source scale effect (`just subjective-rank-check`). This does not retrospectively correct earlier fits.

* **2026-10-15 — general benchmark workspace dependency resolution. OPEN.**
  `cargo bench --locked --manifest-path zensim-bench/Cargo.toml --bench
  ssim2_speed_bar --features costcmp --no-run` failed before compilation:
  the optional local `zenwebp` dependency required `zencodec >=0.1.27`, while
  the registry offered only through 0.1.26. Cargo resolves optional dependency
  constraints even when that codec is inactive. E34's assessment-only
  [manifest](zensim-bench/e34-runtime/Cargo.toml) builds the existing benchmark
  source with its required runtime dependencies and committed lock; the fresh
  build and strict clippy pass. The general benchmark manifest and sibling
  repositories were not changed. This does not alter any E34 gate or collector.

* **2026-10-15 — one ledger transport error stopped the entire E34 harvest chain. FIXED in the harvest owner.** The pilot ledger still exists at its original prefix and the same sync succeeds on recheck; the old exception omitted captured stderr, so the historical transport cause is unproven. Bounded sync retries now retain diagnostics. The supervised cycle attempts every released jobset, records each failure and installed count, and retries degraded cycles. Missing or denied ledgers remain errors even when local artifacts are installed; no failed cycle can report completion. The frozen verifier, results, image and worker processes are unchanged.

  A live empty-prefix probe on the installed s5cmd 2.3.0 emitted a JSON error while exiting zero. Ledger sync therefore requires both a zero exit and error-free structured events; an empty successful transfer alone cannot clear an error. Three live supervised passes reached every released jobset and installed225/226/229 of305; all1,494 pre-existing installed files stayed byte-identical. Transport/supervisor negative controls and the existing E33 owner checks pass.

* **2026-10-15 — CubeCL CPU smoke tests pass despite worker-thread atomic compiler panics. OPEN verification gap.** The CI-exact `gpu-cpu` command returns success, but its auxiliary smoke executable prints worker-thread failures for unsupported `atomic<f32>`. The existing assertions check finite parameters and batch counts, so their success does not establish successful atomic training-kernel execution. GPU source and CubeCL 0.10.0 pins are unchanged from `a2012f53`. CIFIX does not change these kernels or tests; this limits the meaning of green CI for GPU training.

* **2026-10-15 — CI integration tests retained retired-slot and finite-secant contracts. FIXED in CIFIX E34 with explicit owner approval.** Research requests now use the comprehensive registered read set clipped to 1825, preserving all emitted-slot bit comparisons. Exact-gradient tests retain the independent served-score jump probes and add closed-form nonlinear ensemble/gate checks. The real-bake attribution matrix preserves its 16 finite cases and explicitly asserts four singular-derivative refusals, both bin sizes, repeated-session refusal, unchanged score/features and identity behavior. Active signed-root derivatives at zero remain unsupported; no finite sensitivity is fabricated. The retired-slot failures reproduce at `a2012f53`; all 17 bake-surface tests pass there, isolating the three derivative-contract failures to E34's `0d63ba5c`. Reproduction: `just cifix-e34-targets`.

* **2026-10-15 — minimal-feature CI compiled v2-only accumulator helpers. FIXED in CIFIX E34.** The no-default-features Clippy invocation reproduced eight dead-code errors on both `a2012f53` and current main. Internal accumulators and the oracle measurement setter now compile with their callers; the legacy reduction also remains available to oracle-only builds. The v2-only derived-input test fixture follows its tests' feature gate. No score arithmetic or test expectation changes. Reproduction: `just cifix-e34-minimal`.

* **2026-10-15 — E34 freeze amendment omitted the approved secondary-seed floor. FIXED in E34 fix 1.** The draft retained revision 4’s control-only seed wording. Revision 5 now explicitly requires seed 0 at 135/135 and seeds 1 and 2 each at `max(paired control count, 133)`, marks the historical wording as superseded, and preserved the then-open D13 state. The owner subsequently chose margin 0.0065 for both families (2026-10-15 08:42 MDT); the frozen record and statistics owner bind that decision. No qualification used the weaker draft rule.

* **2026-10-15 — a lower-only WinsorP99 clamp kept a nonzero exact derivative in its flat region. FIXED in E34 fix 1.** The loaded one-bound metadata reproduced score 2 at −0.01/0/+0.01 but gradient 2 at zero. Activity now uses the actual default/supplied bounds; flat regions and clamp boundaries return zero. Public tests cover zero/one/two bounds and retain stacked WinsorThen transforms’ incomplete-parameter fallback. Forward score arithmetic is unchanged.

* **2026-10-15 — exact-gradient identity size axes indexed past feature roots. FIXED in E34 fix 1.** The public 5-input bake with one supplied feature and four implicit size axes reproduced a panic. Identity inputs now use the forward owner’s exact/prefix/+4 shape rules before any gather; dense/derived gathers check root bounds. Public regressions preserve scores 28/28/30 and verify gradients, zero prefix-tail sensitivity and rejected short inputs.

* **2026-10-15 — D12 dispatch test raced unrelated pixel tests. FIXED locally.** The new storage/dispatch test changed process-wide token overrides while a prepared-session test compared two pixel extractions. The unchanged pixel assertion failed in the parallel suite. The override test now uses the existing separate-process helper; the same prepared-session assertions pass.

* **2026-10-15 — lower-floor recovery could detach a corruption companion. FIXED in D12 preparation.** A reproduced active constant corruption gate gave a zero served derivative while prepared recovery returned the base network derivative. Recovery now remains limited to an isolated bake with no companion; the regression asserts the served score and zero map derivative. No gate threshold or score arithmetic changed.

* **2026-10-15 — D12 supersedes the historical finite-difference sensitivity baseline.** Exact derivatives now drive the served map and instrument. On the unchanged 135 cases, joint passes are A 131, C 130, production seed 0 128 and L smoke 135; nonlinear-model gate failures remain. Score bits are unchanged. See [D12 evidence](benchmarks/e34_part6_2026-10-15.md).

* **2026-10-15 — steering sensitivities are precision-sensitive at the registered FD step. HISTORICAL part-4 report, superseded by D12.** On the frozen 135 G-STEER cases, replacing only gradient evaluation in a diagnostic recovers two A cases, three C cases and the sole part-3 L smoke failure with both f64 FD and analytic gradients; production stays 128/135. At that observation the prepared served map consumed the same native f32 FD sensitivities. A also has a measured ReLU crossing at the unchanged step, so rounding does not explain every discrepancy. Registered results and serving arithmetic remain unchanged. See [E34 precision diagnostic](benchmarks/e34_precision_2026-10-15.md).

* **2026-10-15 — E34 L accepted feature-transform metadata. FIXED locally.** Plain monotone N now refuses explicit or automatic feature transforms before CLI payload loading and in shared library validation. A negative test failed before the fix and passes afterward. The registered L smoke had no transforms, so its existing weights are unchanged. This enforces the E5 identity contract; it does not change the projection or fit arithmetic.

* **2026-10-10 — derived-input (E33 fx1) bakes reported an 820-wide layout. FIXED (`436c41d8`).** `feature_set::bake_feature_set_ref` gave a derived bake `Model::caller_input_width()` (410 direct inputs + 410 products) as its row layout. The bake actually gathers its 420 declared ids from the table row, like any dense bake. `bake_verdict --require-feature-set-match` therefore refused the real E33 C candidate against the 720-wide instrument grids (`LayoutDiffers` 820 > 720). The E33 implementation landed this in `631c4927`. The layout is now the declared read count. The extended `read_set_owners_name_derived_bakes_correctly` fails before the fix and passes after, and the seed-0 `bake_verdict` positive control is unchanged.

* **2026-10-09 — Rev5 steering discards signal beneath the output floor. FIXED for prepared maps; served G-STEER remains OPEN (`40c797fa`).** Prepared steering recovers pre-calibration sensitivities for an isolated single bake when the active lower spline floor erases every served sensitivity. Serving scores and public served-score gradients are unchanged. Rev5 defaults to exact neighbour replay: across the frozen 135 engineering cases, M3f rises in 106, falls in 15 and is unchanged in 14; M2 is unchanged throughout. Largest drops are broad-220-b8 0.847 → 0.765 and broad-220-b16 0.887 → 0.802; none crosses .70. Served G-STEER stays 128/135 with the same seven failures: four tied floor cases and three model linearization limits. Exact feature replay does not guarantee better map ranking for every case. See [STEERFIX](benchmarks/steerfix_2026-10-09.md).

* **2026-10-09 — V40 fleet placement blocked by empty caps. FIXED locally.**
  The packet inherited empty `hosts` maps and the launch gate checked equality
  with live caps without testing placement. The coordinator reports 147 resource
  envelope refusals and 0/40 control cells run before correcting live control
  caps. The packet correction restores the approved standard map for all four
  jobsets and refreshes authorization pins. Launch and freeze now refuse empty
  maps, maps without a registered fleet alias, and invalid slot counts. Training
  archives, cell identities and image remain unchanged; live caps are owned by
  the coordinator. See `scripts/tests/test_v40_launch.py`.

* **2026-10-09 — COSTCMP READY coordinate binding. FIXED (`e95d6622`).** The initial helper accepted a same-input/score/model record with a changed width, tier or requested thread count. The new regression reproduces that acceptance and requires all three mismatches to refuse. READY validation now binds those fields to the declared grid cell; scoring and timing admission rules are unchanged.

* **2026-10-08 — SPEEDQ3 universal speed claim remains unproven. OPEN.** Fresh runtime `c989a2d4` with frozen production model `f803b74c` measures 171 faster, 0 slower and 21 inconclusive of 192 cells. All five SPEEDQ2 slower cells now classify as faster. Inconclusive pointwise paired intervals still prevent the universal gate passing. All 384 Rev4/Rev5 score/420-feature bit checks pass and all 576 frozen comparisons against SPEEDQ2 match. See [SPEEDQ3](benchmarks/rev5_speedq3_2026-10-08.md).

* **2026-10-08 — Frozen production Rev5 misses the universal Rev4 speed gate. SUPERSEDED by SPEEDQ3; the five slower cells were FIXED by REV5PERF2 (`37779f79`, `0bd3b412`) and are faster on `c989a2d4`; the universal claim stays OPEN in the SPEEDQ3 entry.** SPEEDQ2 on runtime `ee9e5b55` and packed seed-0 model `f803b74c` measures all 192 cells: 164 faster, 5 slower, 23 inconclusive using pointwise paired 95% CIs. Slower cells are v4x 1T 64²/128², v4x 2T 64², and v4 1T 64²/128². All 384 Rev4/Rev5 score and 420-consumed-feature bit checks pass. The first-32-clean rule and <2.0 load gate are unchanged. See [the full report](benchmarks/rev5_speedq2_2026-10-08.md) and [the 192-cell before/after comparison](benchmarks/rev5_speedq2_vs_speedq_2026-10-08.md).

* **2026-10-08 — V40 R4 historical population and producer metadata.** The R3
  historical replay path bypassed ordinary-path forbidden declarations on extra
  manifest and auxiliary inputs. Replay now inventories every group, extra input
  and auxiliary declaration, then optional label-free keys, before any payload
  hashing/loading. Forbidden VAL/T0/terminal/test/AIC/holdout identities refuse;
  permitted development identities and recipes without sidecars remain unqualified
  historical compatibility inputs. Packet staging now records the actual producer,
  retained tool hashes and build-log hash in every current binary record, keeping
  prior descriptive records under historical fields. Freeze and package checks
  reconcile these records with source bindings and the actual archive/tool bytes.


* **2026-10-08 — V40 R3 complete native inventory and postfit artifacts.** The
  R2 executable hashed ordinary unregistered manifest payloads and opened four
  auxiliary tables after admitted groups. Strict native preflight now permits
  manifest repeats only of admitted group tables, keys and declarations;
  unregistered provenance and all four unregistered auxiliary routes refuse
  before any group payload open. Explicit unqualified historical recipes retain
  ordinary auxiliary compatibility, with protected ancestry still refused.
  The postfit owner now requires nonempty structured scorer output, exact
  full-budget cell/control pins, finite registered reductions and every hashed
  prediction/result artifact before printing success. Empty successful output
  is a refusal. The synthetic controller tests cover control and all three
  studies through their real shell boundary. Full frozen-image and bounded
  final-scoring receipts are required in the R3 preparation pointer.

* **2026-10-07 — V40 R2 launch/admission/statistics. FIXED locally.** Review found the E30
  prerequisite missing the installed `cells/` level, native `--manifest`
  hashing inputs before group/key admission, and constant seed composites
  receiving a rounded nonzero SE. The corrected actual-binary probes refuse
  before any payload opens; both complete control proofs remain byte-identical.
  Final package/authorization/mirror receipts are required before readiness;
  see [the round-two pointer](benchmarks/v40r2_2026-10-07.pointer.md).
  The controlling four-source E29 amendment and shared fresh control decision
  are bound into the new package. Historical reviewed artifacts remain intact.

* **2026-10-07 — E32 cached assessment planner refusal. FIXED locally in V40.**

  Serving admission correctly refuses the research palette feature IDs. The
  ordinary densify and cached-prediction owners therefore could not assess the
  registered E32 arm. Explicit `--research-palette-cached` validation flags now
  require its qualified Rev5 declaration, fixed 462 IDs, scalar N recipe and
  research-only disposition. They reuse the canonical predictor, gather,
  metadata parser, pin and spline math. The serving predicate remains unchanged;
  an unflagged palette densify still refuses. Cached predictions on twelve
  approved KADID TRAIN+SELECT rows match the Parquet owner bit for bit for
  control, hb4, hc4 and palette; no external/HDR VAL/UPIQ development read occurs.

* **2026-10-07 — E31/E32 native group admission. FIXED locally in V40.**

  The inherited native entry could inspect feature payloads before refusing bad
  UPIQ keys, and palette admission did not bind group weights or complete ordered
  observation identity. The shared trainer preflight now admits every declaration
  and key domain before any feature payload. Actual-binary probes retain zero
  feature opens for late-group failures, development training, VAL keys, false
  order pins, missing pair/ordinal identities and duplicate ordinal indices.
  Historical replay remains explicitly unqualified.


* **2026-10-07 — E29 preparation import/root regressions. FIXED locally.**

  The new E29 adapter initially lacked the existing scripts/ import bootstrap
  when called by the data packer; packing refused on `lib` import. The bootstrap
  is now explicit (`a9b92db3`), and the clean-environment scorer preflight passes.
  An additional tripwire imported the unguarded HDR panel before installing mocks;
  the module opened the legacy 22,860-row HDR VAL default and printed target
  swings before failing on a missing predictor. No prediction or fit followed.
  The CLI is now guarded (`f634504e`); import and absent-control HDR panel checks
  refuse before payload reads. Preserve the incident in `UNINTENDED_EXPOSURE.json`
  and the amendment/exposure ledger; the read cannot be undone.
  Native HDR manifests also use the approved lexical fleet root without a
  local-only admission_root (`9d563474`), so actual image executor extraction and
  FIT_ROOT linkage pass. Evidence and explicit research qualification limits are
  in `benchmarks/E29_WORKLOG.md`. No full E29 assessment was performed.



* **2026-10-07 — Three zensim integration tests failed on main. FIXED (stale test contracts; reviewed patch, no weakening).** Found by the PRODQUAL-A label-free gate
  run: `featcanon_rev4_contract::served_paths_serve_rev4` (1867 vs 1825 features) and
  `research_engine_parity::research_everything_agrees_with_the_production_walk` (production 1825 vs research 1867) most likely
  from the palette landing (`0a8a7ef8`), which was verified with `--lib` and the palette integration tests only; and
  `bake_surface::formula_revision_is_selected_per_bake_and_unknown_or_mixed_revisions_refuse` (revision 5 accepted where the
  test expects unknown). Cause: `2022da95` (palette research slots → research width 1867; production still 1825, palette still
  refused at serving) and `560de601` (Rev5 registered as known). The tests now assert research width and production width
  separately, keep every bit comparison and geometry, and use 999 as the unknown revision plus an explicit known-5 load
  (MAINFIX lane + strictness review; full zensim suite 900 passed). Serving Rev5 bakes needs `ZENSIM_FORMULA_REV=5` by design
  (the Rev5 spec forbids cross-revision mixing in one process). Landings now run the full `cargo test -p zensim --all-features`.

* **2026-10-07 — E28 score chain posted an empty result. FIXED (this commit).** `e24_rev5.py e28-score` imported
  `lib.zen_stats` without putting `scripts/` on `sys.path` (sibling scripts insert it), so the chain's scorer died on
  import and `score_chain.sh` logged `E28 RESULT: |`. The scorer also needs the prepared root (`wide/`) locally, which
  only the fleet data archive held. Fixed the import; re-extracted `wide/` from the pinned archive. Future launchers
  must stage the local prepared root before the chain runs.

* **2026-10-07 — UPIQ draft admission omitted split/key checks. FIXED locally.**
  Five rewired/unbound cases reached a blocked label open before the guard.
  The same tests now refuse before opens; emitted data was correctly bound.
  See `benchmarks/upiq380_ingestion_policy_2026-10-07.md`.

* **2026-10-07 — SPEEDQ's initial stop applied bit parity to Rev3. AMENDED.**
  The coordinator confirmed the scalar/SIMD split is documented in
  `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md` row (d) and DATASET_HISTORY.
  Rev3 is only a timing baseline; compare its features against the existing
  `max(1e-6 abs, 1e-5*scale)` policy and record failures. Rev4/Rev5 retain strict score-bit parity
  across all required tiers and thread counts before timing. Initial receipts
  remain in the SPEEDQ raw archive; no arithmetic or serving policy changed.
  The amended full grid passes all 384 strict Rev4/Rev5 score checks and bit identity for all 420 consumed features in those cells. Rev3
  scalar exceeds the feature policy in 48 of 192 cells (max abs 3.6560853e-6);
  its SIMD tiers pass. Rev3 remains an explicitly flagged timing baseline.

* **2026-10-07 — E28 admission refused the executor's approved root link. FIXED locally (`8283d518`).**
  The approved-inventory path check rejected symlink ancestors above the
  prepared root, including fit-cell-exec's verified FIT_ROOT binding. It now
  requires paths under the prepared root and refuses symlinks below it. The
  linked-ancestor regression is covered by the 17 passing E28 tests. v36
  additionally tests the actual executor CLI, extraction and link: a bounded
  fit returns a verified result, and both registered arms reach their first
  epoch before deliberate stop. Prior direct-owner smokes did not exercise
  that executor link. Evidence: `benchmarks/E28_WORKLOG.md`.

* **2026-10-07 — Palette verification accepted wrong instrument identity/map. FIXED in PALETTE2.**
  Round-one joins ignored the instrument's feature-set identity and column map;
  palette_v1 relabelling and swapped 1825/1826 maps passed. Admission now
  requires a consumer-frozen exact-byte SHA-256 and checks the registered
  identity, ordered integer IDs, exact map, producer commit and false serving
  flag before any feature-table read. The strict negative controls fail before
  and pass after the fix. Original receipts remain `_VERIFIED.round1.json`;
  `_VERIFIED.json` is the new semantic-identity receipt.


- 2026-10-07 SHIPPATH11: strict D1 admission previously hashed freeze coverage and human payloads before rejecting AIC folds/keys. Population checks now finish in a metadata/key phase; open-tripwire regressions cover CLI, lower admission and preparation.

* **2026-10-07 — SHIPPATH7's by_v2fy projection producer broke three registry tests. All three FIXED** (`47a2e1d4` for the two zensim gates; the zensim-validate test now names the projection root and the research-only palette family as known partial producers, so it requires the missing basic coverage to be reported — owner left the call to the coordinator, 2026-10-07).
  `benchmarks/feature_sets_registry.json` registers `basic+v2@w720/rev5_localwin#62adfc93` as a producer whose recorded slots are the
  420 by_v2fy IDs (all other slots NaN). The zensim registry gates derived the full 504-slot `basic+v2` set and failed; the entry now
  declares `slot_selection: by_v2fy_420`, which both gates apply exactly.

* **2026-10-07 — Palette revision declaration lacked its landed commit. FIXED.**
  The full library census rejected the temporary `"-"` marker for palette_v2.
  Its declaration now pins the actual corrective commit `f9a38a33`; the
  existing assertion remains unchanged. Extracted provenance already pinned
  producer `e60a6ad7`, so this metadata correction changes no measured value.

* **2026-10-07 — Initial palette signed-population regression. FIXED at palette_v2.**
  Matched mean populations could report a positive lightness shift for a global
  -0.03 OKLab edit after median-cut repartition (N=3). Signed lightness/chroma
  now subtract each palette's independently weighted means. The full RGB8
  direction regression remains strict for every N=2–8. Initial palette_v1
  sidecars remain superseded evidence and cannot mix with palette_v2.

* **2026-10-07 — Rev1 registry gates fail on the recorded by_v2fy projection. FIXED on main at `47a2e1d4`.**
  Frozen base `1d3bf35a` reproduces both definition-registry and producer-plan
  census failures: `basic+v2@w720/rev5_localwin#62adfc93` records 420 projected
  reads while generic token expansion produces 504 slots. No expectation was
  changed by PALETTE. The registered `by_v2fy_420` selection on main fixes
  both gates; retain the frozen-base historical receipts. The separate
  zensim-validate partial-producer test remains OPEN pending its owner. Saved baseline output is indexed in the PALETTE
  evidence pointer; the palette-only request has its own exact emitted-ID gate.

* **2026-10-05 — E27 launch omitted the actual jobset memory-cap entry. RESOLVED for the running study by the coordinator.**
  Manifest memory hints did not set the fleet runtime cap. The coordinator installed the existing
  `jobset_caps.json` entry with 6g and the registered host caps; observed workers used that envelope.
  Future launches must verify the existing runtime-cap owner before placement. No replacement worker
  or filler was launched by this lane. [E27 launch record](benchmarks/E27_WORKLOG.md) retains the
  omission, runtime correction and immutable publication receipts.

* **2026-10-07 — E28 human-leg/preparation admission gaps (review P2). FIXED locally.**
  Curated human manifests were checksum-only and could be rewired to VAL/dev or another pin;
  preparation recursively copied extra confirmation/HDR payloads before refusal. The new owners
  admit an explicit frozen inventory, expected arm/source/split/role/populations, source bindings
  and label-free row keys before any human table checksum or label decode. All active manifests
  precede table opens, and preparation rejects forbidden directories before copying only approved
  members. Synthetic regressions in `scripts/tests/test_e28_admission.py` have open tripwires.
  The reviewed v32 pack had correct populations; these fixes close reproduced boundary failures.

* **2026-10-07 — E28 opt-in pooled sampling coverage replay omitted pooled endpoint draws. FIXED locally (`5b2cc589`).**
  The original E28 coverage replay still called the legacy pair draw alone. The live trainer already
  pooled endpoints, so its weights were unaffected, but the embedded coverage digest described the
  wrong stream. Replay now calls the live pooled-draw owner; full executor smoke verification compares
  the recorded digest to `ZENSIM_SAMPLE_DIGEST=1` emitted by the actual trainer. The same change refuses
  nonzero pooled objective flags with a GPU runtime before payload reads. Disabled legacy paths keep
  their prior coverage shape. Evidence: `benchmarks/E28_WORKLOG.md`.

* **2026-09-30 — Rev1–Rev3 research tailhist Bin slots keep the sign-bit top-bin defect. OPEN by decision; fixed only at Rev4 (`c3negfold`).**
  `TailEdges::bin` (`feature_v2.rs`) binned `f64::to_bits` directly, so a sign-bit-set value (tiny negative
  `art`/`det` from f32 rounding in `edge_dissim`; NaN) sorted above every positive edge and landed in the TOP bin.
  Effect: phantom top-edge saturation (REVIEW_PARTB 2026-09-25: 2.01–3.36% of KADID/TID `art`/`det` cells) and an
  emitted `p99` above the exact `max` (bank row: p99 1.27 with max 0.998). The `tailhist` family is the landed
  `rev4bank` era and is computed at EVERY formula revision, so the fix is gated on the computation's revision
  (`TailAccum::fold`, true at `FormulaRevision::Rev4` only): Rev1–Rev3 emit the legacy bins byte for byte, so the
  existing Rev3 sidecars stay reproducible from code. Consumers of Rev1–Rev3 tailhist `*_p95`/`*_p99` must treat them
  as defective; `*_max` is exact at every revision. Regression: `rev4_tailhist_bin_folds_sign_bit_values`,
  `tailhist_fold_is_off_below_rev4`. Measurements: `benchmarks/rev4canon_WORKLOG.md`.

* **2026-09-22 — the 372 roots' `ext_sdr25.parquet` is an unidentified table; 63 board cells read it. OPEN.**
  The 2026-05-15, 2026-08-30 (both 372 roots) and 2026-09-05 post-C roots all carry the same `ext_sdr25.parquet`
  (sha256 `4f567646dcc629a6cb930c03fc9ecf8f79ba0992e6d74741404cfb41785eb96b`): 50 rows = 10 references × 5 rows,
  `human_score` ∈ {2, 6, 7, 9, 10} on every reference. No builder or provenance is recorded. It is NOT the
  5-reference × 10-level `q_jnd` table that the `sdr25` DISTORTION declaration in `check_target_orientation.py`
  describes (the 944/924 roots' table), so its label orientation is unverified.
  - The 63 board cells that read it keep their stored `rank.sdr25` per-reference values (62 were repaired and then
    restored exactly). They are annotated `sdr25-372-root-table-orientation-unverified-2026-09-22` in
    `benchmarks/eval_annotations.json`, which names all 63.
  - Since the 2026-09-22 orientation fix, `bake_verdict` pins the declared distortion orientation for every `sdr25`
    read. **A fresh 372-root verdict therefore prints the opposite per-reference sign from those stored cells** until
    the table is identified. Never compare the two, and read neither as evidence.
  - Structural observation (2026-09-26, not an identification): `ref_basename` takes the values 1…10 and
    `human_score` is exactly {2, 6, 7, 9, 10} on every one of them, which are the five AIC-4/SDR25 source numbers
    (`00002`, `00006`, `00007`, `00009`, `00010` in `site/data/parquet/aic4_sample.parquet`). That is the shape a
    5-source × 10-level table would have with the level in the reference column and the source id in the label
    column. If so, the table carries no human label at all. Check this against the features before relying on it.
  - Fix path: identify the table's source and target semantics. If it is distortion-oriented, repair the 63 cells
    with `promote_fulleval --repair-rank-orientation --repair-tag declared-orientation-2026-09-22`; if it is
    quality-oriented, declare it separately from the `q_jnd` sdr25. Do not delete or rewrite the table.
    Record: `benchmarks/board_orientation_fix_2026-09-22.md` §1.4.

* **2026-09-26 — x86 edge-only horizontal-blur tails: FIXED in 7d6d7451.**
  Scalar remainders now accumulate `(sum + add) - remove`, matching the full-feature
  path. The existing bit-exact regression includes narrow and odd-width inputs.
  The Margarine ablation previously differed in 30/168 values on a 17×19 fixture
  (maximum absolute difference 5.960464477539063e-8). The unchanged consumer
  test passes with this repair. Fractional attribution helpers are separately
  gated with their v2 callers in ad18b444; no arithmetic changed there.

* **2026-09-25 — two extraction paths disagree on pixel-identical pairs. OPEN.** `BakeScorer::compute` (the
  `--full-944` extractor route, which built the Rev4 bank's old family) returns the identity short-circuit
  (`metric.rs` `identical_result_at`: score 100 and an all-zero feature vector), while `research::extract`
  (`--full-986`, `--full-rev4`, `--full-gmsbank`, restore-cuts) computes the walk. On the bank's 88 identical keys
  that is 6,971 differing old-family cells (REVIEW_PARTB, 2026-09-25). By the registry, the zero is wrong on 27
  ReferenceOnly slots (`grad_src_mean`, `luma_mean_ref`, `pjnd_fragility`; the computed value equals every
  same-reference sibling, 2,376 cells) and right on 55 Difference slots (the research path emits FP residue).
  Consequences: never mix the two conventions in one table (restore-cuts `mapdev` is nonzero on identity; the
  feature-potential run zero-masks candidate families on identical keys in its adapter); for future Rev4 training
  tables, excluding `pixels_identical` keys is the recommended rule (the runtime never scores them with a model).
  Also open from the same review: `contrast_loss` identity residue reaches 3.64e-3 on real images at Rev3, above the
  2e-3 bar of `feature_invariants::identity_nonzero_slots_are_reference_only_pjnd_or_fp_residue`, which only
  exercises synthetic noise; and `PJND_FRAGILITY`'s F15 note ("should be 0") contradicts its ReferenceOnly Form and
  its formula `1 - saturate(mean grad_src_mag)`.

* **2026-09-25 — bulk sRGB→XYB gives different bits for the same colour in the vector chunks and in the remainder,
  SIMD tiers. OPEN; the fix needs a formula-revision decision.** (The scalar tier / i686 half is FIXED, next entry.)
  - `color::srgb_to_positive_xyb_planar_into` converts each band of `rows × width` pixels in one call. Full
    vector chunks (16- and 8-wide) use magetypes' `cbrt_midp` (seed `0x2a508c2d`, unfused Halley steps). The last
    `rows × width mod 8` pixels use `color::cbrtf_fast` (seed `709_958_130`, fused Halley steps).
  - Measured over all 2^24 sRGB8 colours with the real owner
    (`cargo run --release -p zensim --example xyb_chunk_tail_parity`, one line per tier): the XYB output differs for
    **84.600 %** of colours on x86_64 v4x, v4 and v3 alike (per channel X/Y/B 10,857,190 / 8,584,754 / 9,519,698), by
    up to **196 ULP**. At most 7 pixels per converted band are affected.
  - **wasm128** (wasmtime, simd128) has the same divergence: **85.766 %** (11,023,236 / 8,768,277 / 9,698,716,
    max 196 ULP) — the numbers the scalar tier had before its fix, because magetypes' wasm128 `mul_add` is unfused
    too. aarch64 NEON was not run (no aarch64 linker on the measuring box); it shares the code path, so expect the
    same class of divergence, unmeasured.
  - Consequence: on those tiers a flat image is not flat after conversion when its band pixel count is not a
    multiple of 8. `feature_v2::tests::gmsbank_constant_chroma_shift_is_visible_without_gradients` passes on x86_64
    and aarch64 only because its two test colours shift by the same 1 ULP in Y, so the reference and distorted
    artefacts cancel.
  - Any fix (pad the remainder into one more vector chunk, or one cube-root and `mul_add` form on every path)
    changes feature bits for images whose band pixel count is not a multiple of 8. Every stored feature table was
    produced on one of these tiers, so it is an arithmetic-revision decision, not a silent patch. The scalar-tier fix
    below is the template: same conversion, remainder zero-padded through one more chunk.
* **2026-09-25 — the same chunk-vs-remainder divergence on the scalar tier (i686, and any host with no vector
  token). FIXED** (this commit; user decision 2026-09-25, verbatim: "You can change i686 values fine.").
  - The decision covers the scalar tier only. It does not cover x86_64 v4x/v4/v3, aarch64 NEON or wasm128.
  - Before: **85.766 %** of colours differed between a full chunk and the remainder on i686 (X/Y/B 11,023,236 /
    8,768,277 / 9,698,716, max 196 ULP); the chunk matrix is unfused there because magetypes' scalar `mul_add` is
    `a * b + c`, while the remainder used the fused `cbrtf_fast` form.
  - Now the scalar tier zero-pads the last `n mod 8` pixels into one more 8-pixel array, runs the chunk arithmetic
    (`color::OpsinChunk`, one shared definition for neon, wasm128 and scalar), and copies the first `n mod 8`
    outputs back. Applied to every bulk conversion with the chunk-plus-`cbrtf_fast`-remainder shape:
    `srgb_to_positive_xyb_planar_into`, `srgb_to_xyb_planar_into` and `linear_to_positive_xyb_planar_into`.
    The x86_64 v4x/v4/v3 variants are not touched; neon and wasm128 keep their per-pixel remainder
    (`PaddedTail::PAD_TAIL = false`).
  - Measured after: `xyb_chunk_tail_parity` reports **0 differing colours** on native i686 and on the x86_64
    forced-scalar permutations. `feature_v2::tests::gmsbank_constant_chroma_shift_is_visible_without_gradients`
    now passes on the scalar tier because a flat image stays flat.
  - Scalar-tier values move only for bands whose pixel count is not a multiple of 8, and only in the last
    `n mod 8` pixels of a band. **i686 scalar and wasm128 were bit-identical (160/160,
    `benchmarks/dense_serving_ungate_2026-09-06.md` §2d) and are no longer**, for such bands.
  - The `GamutMapping::Preserve` converter (`linear_to_positive_xyb_planar_into_unclamped`) follows the same
    tail policy: on the scalar tier it runs the same `OpsinChunk` arithmetic without the input clamp (full chunks plus a
    zero-padded remainder chunk, `CLAMP = false`), so it matches the clamped entry bit for bit at every position
    on in-gamut input. On every other tier it keeps its per-pixel form for every pixel, unchanged (dispatch is now
    `incant!` `[v3, neon, wasm128, scalar]`; v4/v4x use the v3 variant). Preserve-mode values on v4x/v4/v3 were
    verified bit-identical before and after over all 2^24 colours plus out-of-gamut linear input.
  - Test changes: `unclamped_matches_clamped_scalar_for_in_gamut` keeps its original body and now holds on every
    tier. `scalar_tier_unclamped_matches_clamped_at_every_position` adds bands of 1–40 pixels asserting unclamped ==
    clamped bit for bit at every position (scalar variants called directly, so it runs on every host).
    `streaming::tests::convert_chunk_rows_is_semantics_not_a_knob` asserts on the scalar tier that no chunk height
    moves a byte (its old second half, "some height must move a byte", is exactly what the fix removes; unchanged on
    every other tier; accepted in coordinator review). `color::tests::scalar_tier_remainder_matches_full_chunk_arithmetic`
    fails on the pre-fix arithmetic (negative control run) and passes now.
* **2026-09-25 — the C8 landing left `main` with an unparseable `zensim-bench/Cargo.toml` and two failing C8
  tests; a C1–C4 identity test was also racing. FIXED** (this commit).
  - The landing rebase kept both the E5A and the C8 `gmsd` entries in `[dependencies]`. TOML rejects a duplicate
    key, so nothing in the standalone `zensim-bench` workspace built, and root CI could not see it. One entry now
    remains, with default features off, so the C8 peer scorer keeps the no_std sqrt that produced the stored
    peer columns; `e5a-render` and `gmsd-arm` enable `gmsd/std` themselves.
  - `research_everything_agrees_with_the_production_walk` compared a 1322-wide production walk against the
    1502-wide `research::full_width()`. It now enables `gmsbank` and also checks the research width.
  - `gmsbank_contrast_reduction_has_exact_zero_gain_in_every_tier` still walked the pre-chroma 12 × 15 layout, so
    it read a `cs_dev` slot as a gain slot. It now follows the landed `GmsbankChroma` layout: scale-0 Y, then
    X/Y/B cells plus 10 `cs_*` slots at scales 1–3.
  - `rev4_synthetic_16_pair_identity` (C1–C4, failing on macOS Intel CI before C8 landed) was a test race, not an
    extraction defect. The permutation tests in the same binary disable SIMD tokens process-wide, so its on and
    off walks could run at different tiers. Measured: 3 failures in 4 parallel runs, a different slot each time;
    0 serially. It and `rev4_corpus_toggle_identity` now hold `archmage::testing::lock_token_testing()`; 0
    failures in 12 parallel runs.
* **2026-09-23 — AVIF RGB16→RGB8 in zenmetrics' sRGB-tagged decode route rounds down at near-half levels. OPEN
  (upstream).**
  - The Opus review (`REVIEW_AVIF_DECODE.md`, recorded in `benchmarks/rev4_avif_decode_diff_2026-09-23.md` after its
    correction) measured it on the same zenavif source.
  - The native RGB16 output is identical on both routes.
  - The **untagged route** equals exact `round(v10·255/1023)` on 60,349,731 of 60,349,731 channel values.
  - The **sRGB-tagged `RowConverter` route** (zenpixels-convert, used by the Sept-14 SafeSyn extraction) is −1 on
    96,160 values (0.16%), at near-half levels such as 169.5015→169.
  - **Consequence:** the stored Sept-14 SafeSyn AVIF features and labels sit on the inexact side. Measured on 31
    pairs: SSIMULACRA2 mean +0.0023 (max ±0.085), zensim B mean −0.0007.
  - Separately, zenmetrics drops the signalled transfer tag (a metadata defect).
  - Rescoring or re-extracting AVIF therefore changes values. Pin the decode route per data era, and never mix routes
    in one table.
  - The fix belongs to zenpixels-convert and zenmetrics, not this repo.

* **2026-09-18 — identity disagrees between scoring paths. FIXED** (this
  commit). On a byte-identical pair `Zensim::compute` returned exactly 100
  (identity short-circuit) while `Zensim::compute_with_diffmap` returned
  96.2017 (Profile B, 900×675), `Zensim::compute_streaming_strips` returned
  96.2368 (129×128) and `compute_folded944_score_and_attribution{,_binned}`
  returned 96.2368. All now delegate to `compute`, which owns the single
  identity check (`metric::images_byte_identical` → `identical_result_at`); the
  diffmap is exactly zero. Pinned by
  `diffmap::tests::identity_agrees_across_every_scoring_entry` and
  `one_lsb_difference_is_not_short_circuited`. **Still open by construction:**
  every `*_with_ref*` entry takes an XYB pyramid, not the source pixels, so it
  cannot reach the identity owner and scores a perfect copy through the model.
  `diffmap::tests::identity_is_undecidable_without_the_source` pins that.
* **2026-09-18 — `BakeScorer` scores a Rev3 bake at the process revision without
  refusing. NOT A BUG — the report was wrong; entry corrected, not deleted.**
  The claim was that with `ZENSIM_FORMULA_REV` unset (a Rev1 process) the narrow
  basic/peak plan serves the frozen R915 Rev3 bakes at the wrong arithmetic.
  Measured 2026-09-18 on all ten `/var/tmp/zensim-validation-2026-09-15/recovery/
  calibrated/R915_*.bin` (each of which DOES declare `zentrain.formula_revision
  = 3`; a raw byte grep misses the key, `zenpredict::Model::metadata` is the
  only reliable reader): every score is **bit-identical** with the variable
  unset, `=1` and `=3`. Reconstructed Y60 and basic228 bakes over the same ids
  show the same invariance, and a Rev1-declaring bake over identical ids scores
  differently in the same process — so the narrow route SELECTS the declared
  revision rather than ignoring it. The `metric/bake.rs` early return (~862-884)
  is that selection; the ~885-902 refusal is the wide-family path, which still
  refuses a mismatch. `ZENSIM_FORMULA_REV` is NOT a serving requirement for a
  declared-revision bake. Pinned by
  `metric::bake::revision_contract_tests::narrow_plans_serve_the_declared_revision_in_every_process`
  (Y60 + basic228 + `compute_hdr` + `prepare_steering`, across Rev1/2/3 child
  processes) alongside the pre-existing
  `prepared_workers_honor_model_revision_and_local_subset`. A bake that declares
  NO revision still resolves to `SHIPPED_REVISION` (Rev1) in every process —
  never to the process revision — and an unknown revision is refused at load.
  If the speed-matrix run really saw two different numbers, the cause is
  somewhere other than revision selection; re-open with the two scores and the
  bake sha256.

## COSTCMP production scoring cost — 2026-10-09

[The measured report](benchmarks/costcmp_2026-10-09.md) compares frozen seed-0
production `f803b74c` at Rev5 on runtime `462f7fe5` with complete serving A/B
at Rev1, fast-ssim2 main `09ec3e7c` (0.9.0), and registry 0.8.2. All 64
configurations / 320 arm timings and 80 fresh-process RSS observations pass
unchanged SPEEDQ gates. All 64 production score/420-feature records match
across tiers/threads and match SPEEDQ3. The report validates 120 parent
batches, retains 2,048 of 2,239 rounds with 191 flags excluded across every
arm, and exactly replays 256 paired analyses.

At 1024²/v4x/1T, production/A/main medians are 22.386163 / 45.486195 /
60.736307 ms. At 32T they are 14.170276 / 9.1165755 / 28.848190 ms: A
is cheaper there. Pointwise production faster/slower/inconclusive counts
are A 20/36/8, B 16/47/1, main 52/10/2 and 0.8.2 55/7/2; no universal
speed claim follows. Tiny 64²/1T also costs more for production than A/main.
At 4096², production fresh-process peak RSS is 191,664 / 199,116 KiB at
1T/32T, versus A 505,364 / 731,980 and main 1,839,488 / 2,595,520 KiB.
Inputs are deterministic synthetic RGB8; this is scoring-cost evidence,
not model-quality or corpus-wide qualification. Forty unconstrained OLS
fits and measured 1024²/4096² ms/MP are in the report; the negative fitted
intercepts are not physical startup costs. Pins, commands and large JSON
are recorded in the adjacent metadata and pointer. Local review only;
production dependencies are unchanged.

## SPEEDQ runtime evidence

SPEEDQ3 uses fresh runtime `c989a2d4` and raw root `/mnt/v/output/zensim/speedq3-2026-10-08/`: 192 timing cells, 112 fresh-process RSS observations, 384 strict Rev4/Rev5 parity checks, and exact replay of every saved paired analysis. All six required timing arms have full coverage; optional Rust-av timing remains absent and all seven arms retain RSS. First-32-clean selection retains 6,144 of 6,474 rounds; 330 flagged rounds are excluded across all arms. All 314 parent batches validate, and 192 lock acquisitions/releases match. Quiet gates and frozen model are unchanged. [The report](benchmarks/rev5_speedq3_2026-10-08.md) and [SPEEDQ2 comparison](benchmarks/rev5_speedq3_vs_speedq2_2026-10-08.md) preserve the pointwise-CI limits; between-run median/RSS changes are descriptive.

New-rule reports also bind the worker gate vector to every saved parent batch: offsets, requested/completed counts, single-call iterations, arm inventory and original-once warmup must agree. A parent/worker flag mismatch refuses the report; raw flags and batches remain untouched.

SPEEDQ2 reruns the full grid on landed runtime `ee9e5b55`, starting from `079042fb`, in fresh `/mnt/v/output/zensim/speedq2-2026-10-08/` evidence. Timing accepts `--lock <path>`: an exclusive flock covers each segment from its first quiet gate through the last round and worker cleanup; it releases before retry backoff and between segments. Quiet and round admission thresholds remain unchanged. Other cooperating heavy-work lanes use the same lock. `just speedq-test` exercises real lock contention, exception cleanup, gate/round coverage and acquisition between segments.

The release timing/RSS qualification uses the frozen seed-0 packed f16 production model, source SHA256 `f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400`. The production bake already declares Rev5 and remains byte-identical; the existing bake owner stamps the Rev3/Rev4 controls without requantizing weights. The production arm uses process revision 5. Model SHA bindings in parity, timing and RSS prevent the earlier research-weight run from qualifying this model. Research attempts remain archived separately.

`SPEEDQ=1 SPEEDQ_ANALYZER=<pinned paired_rounds binary> just bench-speed-matrix <raw-root> <summary-stem>` extends the existing paired speed matrix with isolated formula workers. Correctness covers 8 geometries × 4 dispatch ceilings × 6 thread counts × 3 revisions before timing. Rev4/Rev5 are score-bit strict; the coordinator explicitly admits Rev3 as a timing baseline with its scalar feature-tolerance failure recorded.

Each timing segment must pass the load1 < 2.0 / no foreign build or training gate before setup and again before rounds. All retained zenbench round gates must be clean; excluded attempts remain as `.bak` directories and retries re-enter the quiet gate. Seven explicit worker owners keep per-call timing inside the scorer, with parent IPC rounds stored separately. Peak RSS is a separate fresh-process pass at v4x, 1 and 32 requested threads. `just speedq-freeze` selects artifacts from Cargo JSON instead of guessing a target filename. `just speedq-test` checks parity refusal, evidence preservation, artifact selection and report units/coverage. No stress calibration workloads run.

Contaminated segments move to the end of the pending timing queue after evidence preservation and the ten-second backoff. Each remaining segment gets an attempt before the collector returns to one that failed its round gates; only one segment runs at a time. Admission checks and all 32 round requirements are unchanged. `just speedq-test` reproduces starvation by an immediately retried rejected segment and verifies the queue visits another segment before retrying it.

The October 8 coordinator arm amendment omits only `ssimulacra2_rs` from remaining timing segments via explicit `speedq_run.py timing --arms` worker names. The default remains all seven arms. Earlier seven-arm segments keep their measured values; later six-arm segments retain every revision, B, fast-ssim2 and butteraugli. The reporter records per-arm coverage, uses v4x rows only for the Rust-av peer table, and leaves absent optional timings/fits null. Peak RSS keeps all seven arms; interrupted attempts are preserved as `.interrupted-<ns>.bak`.

The later owner-approved October 8 clean-round amendment replaces whole-segment refusal on any flagged round: retain the first 32 clean rounds in order out of at most 64 total. One shared index vector excludes each flagged round from every arm, preserving pairing. Fewer than 32 clean rounds, foreign build/training during the attempt, or zenbench unreliable status still triggers a fresh TimingNoise retry. Existing all-32-clean segments remain valid; archived attempts are never salvaged. Raw outputs preserve every measured round and flag; COMPLETE.json and the report record the rule, total/excluded counts and indices. Adaptive batches request only the rounds still missing, keep the same persistent workers and run the original explicit warmup once; per-batch untimed estimation calls are excluded. Quiet gates, grid, six-arm inventory, parity and RSS scope are unchanged.

The pinned SPEEDQ Cargo lock resolves fast-ssim2 0.8.2, butteraugli 0.9.3 and Rust-av ssimulacra2 0.5.1 with Rayon enabled for all three. Zensim/fast-ssim2/butteraugli share archmage 0.9.29, so its forced ceiling applies to those arms; Rust-av has no archmage dependency and its tier rows repeat the same compiled implementation. Verify resolved features before assuming a peer is serial.

### Known Bugs: E36 publication phase scoping (2026-10-17)

Fixed: mandatory native publication previously scanned excluded optional-phase
ledger-error logs. Publication now checks its own phase and predecessor logs;
optional-only ledger failures remain diagnostic. Actual mandatory ledger errors,
exact native identities and single-attempt publication remain admission gates.
Fixture: scripts/tests/test_e36_owner_amendment.py.

### Known Bugs: E36 mandatory preference support (2026-10-17)

Mandatory12,300-view native coverage passed; actual pool preparation refused
one empty required stratum: SVT,lineart,1024,selection,band2. All69proposed
edges miss the fixed CVVDP-FHD margin,with zero admitted aliases. The support
guard remains unchanged and no fit pool qualifies. A preview of the entire
predeclared optional phase finds38candidate edges in its three matching views;
full physical dedup/admission has not run. Pool-choice amendment awaits
coordinator direction before fitting. Evidence: e36 generation pointer and
`e36_zero_support_diagnosis.py`; no threshold or source subset was changed.

2026-10-17 pre-fit follow-up: coordinator approved the entire completed12684-view
matrix after this support failure,with no outcome/loss/human-label inspection
and no hand-picked sources. The old choice and failed pool remain immutable;
`e36_full_choice.py` binds the explicit support-triggered amendment. Fresh full
global deduplication/support/semantic replay is required; this approval does not
waive a guard or release scientific fits.

The complete12684-view replay subsequently passed all1884support strata and
both semantic exports; the formerlyempty stratum retains38unique preferences
afterglobaldeduplication. Fullarchive/nativepool/packageadmission remains
required before the authorized bounded diagnostic. No scientificfit released.

### Known Bugs: E36 amendment and diagnostic ingress (2026-10-17)

The actual full-pool native reader refused because its original budget-equality
check did not recognize the approved4096source cut. Exact registered amendment
identity now binds original/effective/runtime hashes; original inputs retain
the old equality. The bootstrap/options/native CLI also rejected the authorized
2:128 diagnostic. Native executor-derived authorization now admits only its
exact full-data tv10seed0 route,with unchanged full120/50000 and output guards.
A finalizer initially attached the outer binding rather than its nested
authorization; this is corrected and tested. A rebuilt feature-only probe
passes with zero optimizer steps and label reads. Numerical sources are
unchanged; preserve old identities and bind the recorded successor before
fitting. Actualsmoke and finalarchive/packageadmission remain pending.


2026-10-17 E36 diagnostic image finding: cached fit parent
`1cc5ab665d53025d320e5d15cd6a833c53afb395775583334f7c8e07c122efd0`
lacks `/usr/bin/time`. The first bounded native diagnostic failed before its
training command. Its ledger and container remain retained; worker exit0 did
not qualify the cell. `docker/Dockerfile.e36fit` now bakes pinned GNU time at
image build and exercises its verbose receipt before installing the program.
No boot installation or scientific-fit retry is authorized by this repair.
