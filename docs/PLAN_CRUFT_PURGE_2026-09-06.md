# The cruft purge — make zensim development obvious

## Current cleanup plan — 2026-09-07

**Authorized for execution by the user on 2026-09-07.** The initial audit used
`0205c45c`; the first plan/chart update is `45e1ec9a`. This expands the existing feature-layout cleanup;
the original September 6 registration is preserved below. Its early “D
(default)”, “no declared bakes” and consumer-not-wired statements are historical:
**B is the library/`codec_target()` default; A/B/BHdr/D already use explicit IDs
and the consumers gather them.** The targeting CLI's earlier `tuner-v4` default
is aligned to `codec-target` in the execution below.
Repeating completed migrations would add work, not clarity.

**Later September 7 ruling, adopted:** a new model must execute and serve
entirely in Rust, be wired to a zensim surface API, and be evaluated through
that API. This includes every head, corruption gate, spline and composition
step. Python remains useful for invention, but a Python-only or evaluator-only
function is not an introduced zensim model. **No consumers have calibrated to
B, C or D; all three may be improved or replaced.** Their current values are
baselines for comparison, not compatibility constraints on future scores.
Feature ablation is low priority unless it removes actual computation or
answers a specific consequential scientific question.

The objective is that a developer can answer five questions from one entry
page: **what behavior matters, which implementation owns it, how to change it,
how to test it, and whether the result is better for users.** The user still
controls one target score. Fewer files alone is not the acceptance criterion.

The initial measured surface was 612 Python/shell files, 46 Rust binary source
files, 76 Rust examples, 13 `just` recipes and 660 tracked Markdown files. Its
structural script checks passed; they cannot tell us whether a script uses the
right science or duplicates another implementation. Detailed candidate inventories are local
at `~/tmp/zensim-science-audit-2026-09-07/`.

### Completion checklist — September 7 continuation

The user has requested **the full remaining list**, followed by evidence that
the cleaned path can still train competitive models. This checklist tracks the
operative September 7 plan, not superseded status statements in the historical
appendix. Conditional scientific removals must earn their evidence; a failed
ablation closes its hypothesis and preserves the useful component.

- [x] Rust surface for dynamically loaded candidate bakes; shared validated
  head/spline decoding; reusable prediction state; pixel/cached-feature parity;
  complete corruption composition through the same API used by evaluation.
- [x] Migrate candidate-evaluation callers and remove alternate runtime dispatch.
- [x] Explicit validated evaluation stages and artifact-identity reuse; harvest
  invokes the owner once; qualification distinguishes failure from missing data.
- [x] Complete transform-screen option migration and remaining spline-writer
  consolidation with historical recipe and boundary evidence.
- [x] Enforce trainer capabilities before CPU/GPU work; enforce feature IDs and
  semantic revisions at table admission; preserve honest historical replay.
- [x] Resolve C/CHdr activity semantics, declared-ID conversion and legacy plan
  derivation retirement, with the serving census and feature-build matrix.
- [x] Consolidate repeated feature/era metadata; retire unused positional
  training surfaces and the stale JS example after caller checks.
- [x] Slim current instructions; consolidate workflow docs; move completed root
  documents and retire obsolete launch tools while retaining their evidence.
- [x] Close the bounded compute/implementation ablation review using existing
  results and any necessary discriminating measurements; run the all-codec
  floor-coverage control, including rav1e, without assuming qualification.
- [x] Reproduce competitive training from recorded recipes on the cleaned path;
  evaluate the emitted bakes through the Rust API and report held-out quality,
  identity/floors, serving cost and actual target-loop behavior separately.
- [x] Run required checks, resolve regressions, push and verify every increment,
  and leave a complete execution/evidence record.

**API delta registered before implementation:** a reusable `BakeScorer<'a>`
surface in `zensim`, borrowing a parsed `zenpredict::Model` owned by the caller.
Construction validates metadata and feature declarations; cached-feature and
image-pair methods use the same inference dispatch. The image path uses the
existing extraction planner and pixel validation. Predictor/gather scratch is
reused, with no leaked bake bytes or mutable global loader. Model-author
configuration can select score disposition and attach Rust corruption heads;
the final composed score is the value returned and evaluated. Existing named
profiles and their signatures remain compatible. Its exact exported methods
will be enumerated with the API snapshot and changelog in the implementation
increment. New public items are limited to concrete evaluator/serving callers.

**Training validation design:** reproduce the latest recorded competitive
`372 S228 H128` / best-of-all `A_plain` recipe with its three recorded seeds
(4004/4005/4006), keeping data, order, packing and instrument identities fixed.
The recorded reference means are approximately CID22 0.8891, KonJND 0.4997,
composite 0.8729. Before launching, pin the exact baseline artifacts and compare
current API scores on the same rows; require before/after prediction or metric
parity for unchanged recipes. Competitive rank is distinct from product
qualification: these controls have known identity and codec-floor failures.
Test the constrained challenger and the registered floor-data control as
separate claims; do not silently waive failed target-dial gates.

### Surface increment — September 7 continuation

The registered candidate surface is implemented as `zensim::BakeScorer`.
Exact public additions are enumerated in `CHANGELOG.md` and the generated
API snapshots, including hidden wire-format views used by the existing
calibration/diagnostic tools. `compute` returns a `ZensimResult` containing the
final composed score and the exact features; `compute_hdr` and cached-row
`score_features` return the same score scalar. The named profile APIs retain
their signatures and pinned outputs.

All candidate evaluation callers now execute this surface. The alternate
`bake_runtime` forward/dispatch and duplicated metadata parsers are removed;
its remaining functions adapt legacy CLI spellings and diagnostic data shapes.
`serve_custom_bake` no longer leaks bytes or mutates a global loader.
`bake_contrib` deliberately perturbs network activations for diagnostics,
uses the surface's diagnostic tail, and gates its baseline against the full
surface. Calibration fitting removes the output calibration explicitly and
then evaluates the emitted artifact with the complete surface.

Instrument corrections: corruption composition affects **all** verdict scores;
the two older comparison/row tools now honor calibration; feature-stream
scoring no longer truncates dense identity rows. Min-max heads now apply a
present output pin before the spline. Unknown/malformed metadata refuses;
dense plans honor per-bake global-contrast revisions. The existing SSIM
kernels select their luminance form per process; candidate pixel scoring
refuses a mismatch and names `ZENSIM_FORMULA_REV` instead of returning a mixed
formula. This is an explicit serving requirement, not a claim that the
per-request kernel migration has landed. Historical results retain their
old instrument identity. No new competitive or product-qualified claim is
made by this migration alone.

Verification for this increment: full workspace tests; all-feature library
421 passed / 6 explicitly ignored; ten candidate-surface tests plus the
score-owner/polarity/nonnegative-distance and contribution gates; CI-exact
`just clippy`; the serving matrix (eight feature arms, zero refusals or score
drift); 486 distinct filesystem bakes served with zero refusals. A stale
checkout path is resolved only by its recorded SHA-256 against preserved
weights; unreadable or unservable entries now fail the census exit status.
The API snapshot check passes; semver-checks against `902aa68f` reports
196 passes, 58 skips and no required semver update.

The fixed `A_plain_s4004_packed.bin` control, rescored on its recorded post-C
root, exactly reproduces **every value** in `rank`, `dial`, `corruption`,
`per_pair`, `gates`, `composite` and `features_root` from its stored verdict.
Its composite remains `0.8727940159416271`. This establishes unchanged
inference/statistics for the competitive control, not training reproduction
or product qualification; those remain in the checklist above.

### Evaluation and training preservation — September 7 continuation

The evaluation-stage checklist item is complete: independently reusable verdict
and coherence stages, content identities, atomic outputs, harvest through one
owner, and explicit product qualification in `freeze_check --qualify`.
The real control ran all 27 coherence cells, then reused both completed stages;
interruption/stale-input tests pass. Attached evidence is retained only for the
same verdict identity and its hashes/composition are rechecked for qualification.
Both gauntlet pages retain the negative score axes and display qualification
separately from research selection.

The three registered training seeds completed on the cleaned path. Every
measured verdict/ladder field exactly reproduces its September 6 reference.
CID22 mean is 0.8890945 (sample SD 0.0023693). All three retain the known
identity and five-codec-floor failures. Historical KonJND is an overlapping
memorization guard, not a holdout. See the [full reproduction record](../benchmarks/cleanup_training_reproduction_2026-09-07.md).
This completes training preservation only; the remaining scientific controls
and product-loop measurements stay explicitly open in the checklist.

### Execution record — first batch, September 7

**This batch executes parts of §§1–3. The API migration in §1a and the
remaining work below are still open.** No new model was trained or admitted.

Fourteen obsolete source files were retired. Their exact source, historical
instrument definitions and invocation recipes remain in
[the preceding revision, `45e1ec9a`](https://github.com/imazen/zensimx/tree/45e1ec9a95f0cbbb58decef46155aa46d6d1b8b6).
Tracked caller searches found no remaining executable consumers after retiring
the obsolete families together. Historical TOML manifests, benchmark records,
bake inputs and data remain intact. A private source backup was also retained.
This is archival retirement, **not** a claim that the current Rust trainer
reproduces every old Torch recipe or that the different statistics agree.

| Retired source | Executed change |
|---|---|
| `scripts/fit_output_spline.py`, `scripts/mohammadi_eval.py` | Retired the holdout-fitting utility and its separate historical statistics CLI. The independent Python panel reference remains; its header now states the actual gate and shared-fit limitations. |
| `scripts/v_next/v0_20b/bake_v3.py` | Removed the incomplete baker; corrected the fine-tuning instructions to the existing historical JSON/baker adapter. |
| `zensim-validate/src/bin/preview_stats_demo.rs` | Removed the unused May report binary and its active caller comments. |
| `scripts/v_next/blend_lib.py`, `blend_search.py` | Retired the obsolete July trainer/search and private statistics path. |
| Eight files in `scripts/v_next/`: `train_mlp_negatives.py`, `train_mlp_diverse.py`, `mlp_piecewise_negatives_probe.py`, `bake_mlp_negatives.py`, `column_audit.py`, `grid_diverse.py`, `cross_eval_diverse.py`, `diagnose_poison.py` | Retired the negative/diverse training family with its analysis/baking consumers. The canonical KADIS builder keeps its dated provenance-gap record. |

`bandwise_dashboard.py` is now a thin entry to `gauntlet.py`; its independent
NPZ/Matplotlib mode is retired. The gauntlet no longer imports a trainer to
recompute missing composites. It displays the recorded verdict, with absent
composites left unmeasured. **All 508 loaded row dictionaries are identical
before and after this change.** Both served boards were regenerated and retain
their score axes below zero.

The targeting CLI and `--profile default` now resolve through `codec-target`,
matching `TargetSpec::default()` (currently B). This deliberately changes
default target scores/search behavior; `--profile tuner-v4` preserves the old
selection. No new codec-performance or convergence claim is made by this
default-alignment change.

The existing `justfile` now requires the actual recipe for `check-mix` and
`check-data`, removing the implicit August campaign input. `full-eval` forwards
explicit bake/name/regime/root arguments to the existing owner; `compare`
renders stored verdicts and runs the board gates. `metric-eval` is labeled a
quick offline report, preserves quoted paths, and honors `CARGO_TARGET_DIR`.
The duplicated join-safety unit-test invocation was removed; its unit tests
and the distinct source gate remain in their existing CI jobs.

**Verification:** `cargo check -p zensim-validate --all-targets`; the CLI/library
default regression test; join source gate and all 18 join unit tests; both
gauntlet gate suites (including 508 strict-valid JSON inputs); real Chromium
checks of negative/positive tails, shared axes, selection changes and both
themes; 36-case panel parity at `1e-9`, conditional on Rust's logistic fit;
argument-boundary checks for the changed `just` recipes; targeted Rust format
checks and `git diff --check`.

The standalone CLI test used `--no-default-features` and a temporary Cargo
path override to the sibling `zenanalyze`: its existing local codec dependency
otherwise requests an unpublished crates.io version. This tests parsing/default
selection, not enabled-codec integration. All 613 remaining script/doc
structural checks pass. **The full script linter still fails on 26 existing
private-identifier matches across eight files**, all byte-identical to the
pushed baseline. They were neither hidden nor exempted; the outgoing-diff
push gate remains mandatory.

**Kept pending evidence:** `v0_20_screen_to_trainer_args.py` has a top-N
`--max-features` cap absent from Rust's `--auto-transforms`; its old README
redundancy claim is corrected. Four spline writers still need recipe and
runtime-boundary comparisons. Active Python hypotheses and independent numeric
references remain. Next is the Rust surface integration, followed by validated
evaluation reuse and the remaining owner/doc consolidation. Feature ablations
without real compute savings stay low priority.

### 0. Establish correctness before choosing what survives

**September 7 user preference:** canonical implementations should generally be
Rust; Python is fastest for invention. Keep that distinction in this plan.
An owner is a maintenance decision, not proof that its result is correct.
There is no language-based deletion rule. Exploratory Python calculations may
test an idea; admitting and evaluating a new zensim model requires the Rust
surface path above, not a promise to port it after selection.

The chronology supports a substantial Rust production/reproduction path by
July 29, but not a completed removal of Python. The Rust trainer existed on
May 10; the older spline scripts date to May and `blend_lib.py` to July 15.
The [July duplication audit](../benchmarks/duplication_audit_2026-07-15.md)
explicitly left the Torch cluster open in its July 29 addendum. These are
mostly unfinished migrations, not evidence of a wholesale move back to Python
in the last month. Check later dated sections before accepting a port's initial
status summary.

| Area | Evidence and confidence | Limit before replacement/deletion |
|---|---|---|
| Specific bake reproduction and packing | Strong, bounded [July 29 byte-identity evidence](../benchmarks/key_bake_repro_verification_2026-07-29.md): BHdr from frozen Gram inputs, packing against fresh Python and the shipped artifact. | B starts from a committed raw bake, so its reproduction is not a fresh original fit. A uses a pinned earlier tree. These records do not prove every calibrator or later checkout equivalent. |
| Product/evaluator score arithmetic | Strong binding evidence from [September 6 consolidation](../benchmarks/score_owner_consolidation_2026-09-06.md): actual adapter comparisons, spline boundaries and mutation-tested controls. | Both implementations were Rust and still diverged before that fix. Sharing arithmetic does not prove the chosen formula's perceptual validity or close separate metadata decoding. |
| Features, serving and training mechanisms | Real goldens, invariants, scalar/exact references, finite-difference gradient checks and selected train/bake/serve tests exist. Confidence applies to those tested branches and tolerances. | C/CHdr activity semantics, ignored trainer options and table/metadata admission remain open. SIMD tolerance checks are not byte-identical whole-training proofs. |
| Statistics | Ordinary finite/tied statistics have useful reference checks. Full panel cross-language gates were reported passing [September 1](../benchmarks/hfhuman_2026-09-01.md) and the 36-case script was rerun in this batch. | Those integration tests are ignored by default and reuse Rust's logistic-rescaled output: they do not independently validate the fit. Retired `mohammadi_eval.py` PWRC/OR definitions differ from the current owner. The parity script's overbroad header is now corrected. |
| Torch/blend training and panel | No complete same-recipe Rust/Python training-and-export equivalence gate found. `blend_lib.panel` has demonstrably different rescaling and band definitions. | Preserve useful experiments; establish the intended method, supported options, data/order and exported-bake behavior before replacement. A matching rank correlation is insufficient. |
| Four remaining spline writers | Only partial migration is established; July 29 `recal_v47_dial.py` strip parity does not validate its fit. | Knot selection, f32 serialization and runtime boundaries need separate comparisons. SciPy polynomial extrapolation and the product's bounded linear tails differ. Monotonic knots alone do not prove diagnostic/runtime agreement. |

This confidence review inspected current source/assertions and dated results;
it did **not** rerun the full parity matrix or retrain the models. One small
spline counterexample was checked: knots `(0,0), (1,1), (2,4)` give SciPy
extrapolated output `10` at input `-2`, versus `0` for the old V9 linear-tail
helper. This establishes a contract difference, not a defect in a shipped bake.

For each proposed substitution, first name the intended method independently
of either implementation. Then compare actual implementations on shared normal
and boundary fixtures, with a relevant negative control. Finally evaluate the
actual exported bake through the zensim surface API and the real codec target
loop. Calling a Rust validation-only scorer is insufficient. Record
intentional corrections as changed behavior; do not force parity with a known
bug or quietly regrade historical results with a new instrument.

Python experiments may invent losses, models, transforms and fit procedures.
Record the recipe and artifacts in the existing experiment record, reuse Rust
owners for unchanged operations where practical, and mark results from a private
instrument accordingly. Implement the full candidate in its Rust owner and
wire the surface before model evaluation/selection. Retire the redundant supported path only after that;
retain a Python implementation when it is a useful independent oracle or a
still-active hypothesis. No second experiment registry is needed.

**Research lookup before fundamental changes:** use the sibling `zenpapers`
repository and its related zensim benchmark summaries. Read dated amendments,
not only opening recommendations. Checked for this revision:

- `zenpapers/docs/zensim-final-metric-plan-2026-07-31.md`, including its
  **August 1** correction that B had not shipped and the separate linear/MLP
  and corruption-head findings. Its old fixed-944/freeze proposals do not
  override this ruling or September's explicit feature-ID contract.
- `zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §0: later July 28–29
  adjudications distinguish data gains, structural corruption failures, useful
  features and rejected additions. An old proposed experiment is not pending
  work if its later falsifier already fired.
- `zenpapers/docs/zensim-bake-subset-plans-2026-07-26.md`: twin/partition
  dependencies, marginal extraction cost, and later execution status. Removing
  a model input can leave the shared feature computation entirely intact.

### 1. Make the supported development path small

- Keep [`../SESSION-RESUME.md`](../SESSION-RESUME.md) as the short current-state
  entry. It names the default B, fast baseline D, current unified challenger,
  remaining blockers, and links the commands. Historical model families stay
  available through the archive, outside the default comparison view. B/C/D
  are replaceable; do not preserve their score values on an assumed consumer
  calibration dependency that the user has explicitly ruled out.
- Reduce `CLAUDE.md` to current instructions, the existing owner table and
  links; target roughly 150 lines instead of 4,000. Move unique historical
  findings to their existing benchmark records before removing copied prose.
- Make [`WAVE_PLAYBOOK.md`](WAVE_PLAYBOOK.md) the single operational sequence.
  Fold in enduring rules from `RESEARCH.md` and `ITERATION_PROTOCOL.md`, then
  turn their active-workflow sections into short historical pointers. Keep
  `REPRODUCIBILITY.md`, `MODEL_SELECTION_SCORECARD.md`, `FEATURE_SET_IDS.md` and
  `DATA_SPLITS.md` as the contracts for their respective subjects.
- Extend **the existing `justfile`**, with short useful `just --list` help:
  check a change → validate a recipe → train → evaluate → compare/qualify.
  The first batch adds `full-eval` and `compare` and removes the hardcoded
  August SOTA recipe from `check-mix`; see its execution record above.
  Remaining train/preflight consolidation must use the existing owners.
  Keep normal library development independent of large research datasets and
  the standalone codec-tool workspaces.
- Align the targeting CLI default with `codec-target`, preserving explicit
  legacy profile selection. Treat this as a documented score-behavior migration:
  test CLI/library default agreement and record the changed default outputs.

**Done when:** a developer can follow this route without reading a campaign log,
guessing a feature width, or choosing among several “default” evaluation tools.
Every command prints its resolved inputs and expected outputs. This cleanup
adds no second supported trainer, statistics library, report generator,
scheduler or registry. Bounded Python prototypes remain available under §0.

### 1a. Make the zensim surface the model evaluation path

Use the existing `ZensimProfile::Custom` / `ProfileParams::builder` entry and
`Zensim` scoring surfaces for candidate bakes. The crate already has Rust
splines and corruption-head support; consolidate and wire those owners rather
than introduce another runtime. Multi-head evaluation must include the same
head routing, feature requirements, thresholds, spline boundaries and final
score composition that a caller gets. Auxiliary diagnostics must identify
their meaning; a separately calculated gated score is not proof that the
user's target score behaves that way.

Route `bake_verdict` and other candidate-evaluation callers through that
surface. The current `bake_runtime` adapter sharing `score_math` is useful
evidence but does not yet satisfy the full surface requirement. Cached-feature
evaluation may use a zensim scoring surface only with matching feature identity
and pixel-to-surface checks, including full head/spline composition. Reuse
expensive extraction; do not regain speed by keeping an alternate scorer.

**Done when:** an actual candidate loads and scores image pairs through the
same Rust API used by evaluation; no Python runtime is needed; identity,
negative tails, spline boundaries and every enabled head are tested through
that API. Record API/bake/feature revisions and measure the complete scoring
and target-loop cost. A missing executable surface blocks model admission,
independently of the statistical scorecard.

### 2. First deletion batch: obsolete independent tools

| Target | Why remove it | Condition before removal |
|---|---|---|
| `scripts/fit_output_spline.py` | Independent spline fitting/serialization against held-out AIC-3; no tracked executable caller found. | Preserve the historical revision/reference. Current calibration belongs in `bake_dial_refit` using admissible training anchors; do not reproduce the leakage as a replacement. |
| `scripts/v_next/v0_20b/bake_v3.py` | Incomplete baking stub: writes an NPZ, reports the remaining baker as pending, and can finish successfully without the requested bake. | Confirm no live caller; update active instructions to the existing complete JSON/`zenpredict bake` path. Keep any unique historical recipe at its recorded revision. |
| `zensim-validate/src/bin/preview_stats_demo.rs` | Autodiscovered 756-line binary embedding seven May bakes and obsolete default labels; no executable/workflow caller found. | Check known external CLI usage and preserve any needed report fixture; use `bake_verdict`/the existing comparison board. Do not delete its historical bake inputs with the binary. |
| `scripts/v_next/v0_20_screen_to_trainer_args.py` | The README already calls it redundant with the trainer's `--auto-transforms`. | Compare top-N, minimum-lift and max-feature-ID behavior; migrate any required semantics and callers before deleting. |
| Repeated join-safety unit-test invocation in `.github/workflows/ci.yml` / `joinsafety.yml` | Same unit suite/dependency setup runs twice. | Keep one test invocation and the separate join gate; preserve required CI status checks. |

Use `lint-scripts`, caller searches and relevant argument/behavior checks for
this batch. A missing prebuilt binary is not evidence that its source is dead.
Current `run_cross_codec_v*` aliases already share one implementation and have
argv parity tests; removing those small compatibility wrappers is low priority.

`scripts/mohammadi_eval.py` was retired after the instrument review in §0;
its differing PWRC/OR definitions and keyed-row/sigma behavior remain at the
pinned historical revision. No mathematical-equivalence claim was made. The
independent panel reference remains with corrected scope claims.

### 3. Separate active invention from redundant supported paths

The retired dependency chain was:

```text
bandwise_dashboard.py legacy mode / blend_search.py
    → blend_lib.py (Torch training, NumPy scoring, private statistics)
gauntlet.py historical composite fallback → blend_lib.py
```

Keep `bandwise_dashboard.py --fulleval-dir` and `gauntlet.py` as the current
board owner. Retire the old NPZ/Matplotlib report mode after preserving its
needed outputs and recipe lineage. Identify which blend-search operations are
active hypotheses and which duplicate a supported operation. For the latter,
prove the intended behavior and migrate to `zensim_mlp_train`,
`bake_dial_refit`, `bake_verdict` and `zenstats`; then delete the proven redundant
implementations and their imports **in the same change**.
Archive abandoned studies instead of porting them merely to preserve a command.

Apply the same treatment to the independent negative/diverse training family
(`train_mlp_negatives.py`, `train_mlp_diverse.py` and their analysis/baking
consumers) and the remaining spline writers (`calibrate_v9_spline.py`,
`calibrate_balanced_v9_spline.py`, `v11_ssim2/calibrate_v11_balanced_spline.py`,
`recal_v47_dial.py`). These live under `scripts/v_next/`. Preserve recipe-specific
anchor selection and postprocessing when migrating a live recipe; use
same-recipe bake/prediction parity, not merely a matching aggregate score.

Remove the board's dependency-sensitive composite recalculation. Current rows
consume the Rust verdict's value. Historical rows retain a verified recorded
value/rule or an explicit unmeasured state. Today the broad exception fallback
can change rejection behavior when an optional Python import fails.

**Done when:** rendering stored verdicts requires no Torch trainer or private
IQA statistics, current scientific fields are unchanged, and both gauntlet
gates/browser checks pass. This removes competing implementations while keeping
the comprehensive board and historical evidence.

Extend the existing script linter only where it can catch accidental production
forks or silent fallbacks at these owner boundaries. Review new private methods
for their declared experimental/reference role; do not reject them merely for
using Torch, SciPy or Python. Keep independent reference tests, measured gated
mirrors and frozen as-run protocols. A prototype need not be a second supported
way to produce the official score.

### 4. Make evaluation complete, reusable and explicit

Consolidate the live path through `harvest_bakes.sh` and `run_full_eval.sh`.
Today harvest may run a standalone verdict and then repeat evaluation through
full-eval; full-eval builds both binaries before choosing which measurements
are needed. Build once, record binary identity, and run only required stages.
Retain `sota944_verdict.sh` as a historical campaign invocation until its callers
have migrated; its frozen preset is not the generic development interface.

Replace “this filename exists, therefore done” with validated result reuse:
bake bytes, feature/data/decoder revision, evaluator binary, instrument and
requested stage must match. A partial, malformed or same-name stale file must
not count as a completed evaluation. Reuse existing provenance fields; extend
the owning schema for missing identity, not a second cache registry.

Add a **qualification mode at the existing verdict/selection owners** that
requires complete passing product evidence. The default board shows B, D and
the current challenger with `qualified`, `fails: …` or `missing: …` before the
research composite. Real codec RD/target results are required evidence; an
offline full-eval alone cannot fill those fields.

**Done when:** candidate scoring uses §1a's API; an interrupted run resumes correctly; changing a bake under the
same filename cannot reuse its old verdict; missing floor/identity/loop evidence
prevents qualification; changing presentation never recomputes science.

### 5. Finish the feature contract, then remove legacy routing

- Make the registered feature IDs and per-bake semantic revision authoritative
  from table admission to runtime output. Reject missing/duplicate required IDs
  and incompatible revisions before fitting/scoring. Preserve registered legacy
  aliases; width and an observed all-zero column are insufficient evidence.
- Fix or replace C/CHdr so training and serving agree on `append2_dst_activity`,
  with before/after evidence; the old numeric output is not a consumer contract.
  Then finish the retained/replacement models' declared-ID
  conversion and retire `ComputeSet::from_block_profile` and
  `fold_engine::wide_bake_v2_read` once every remaining legacy/custom caller has
  an equivalent plan. Require pixel-feature/score parity and a cheap-wide/free-set
  cost control, preserving the existing Off→Peaks working-set policy. Coverage
  tests alone and densifying C/CHdr alone do not prove that census.
- Migrate repeated feature metadata parsing/era labeling to the existing
  feature-set/layout owners. Preserve the information carried by old presets
  and public positional accessors through compatibility adapters.
- Make unsupported trainer option combinations fail before work starts.
  The remaining cases include non-alpha anchor/PJND/KonJND losses in
  `mlp_train/mod.rs` and GPU dispatch before the shared capability checks in
  `bin/zensim_mlp_train.rs`. Resolve one effective configuration for both paths.
  Keep historical reproduction modes explicit; normalize irrelevant defaults
  and reject explicitly requested unsupported behavior. The recent depth,
  skip, EMA and other guards already landed; do not reimplement them.
- Retire `FeatureTier` and its positional training truncations in
  `zensim-validate/src/main.rs` after checking CLI consumers. Preserve its
  still-used extraction mode. Review `examples/mlp_cross_check.rs` with the
  historical site's JS consumer before removing that old 228-wide example.
- Share validated score/head metadata decoding between `metric.rs` and
  `bake_runtime.rs`; distinguish absent metadata from malformed metadata.
  Arithmetic already has the `score_math` owner. Keep the runtime cache and
  evaluator adapters, and prove valid-bake parity before retiring parsers.

**Done when:** serving, training and evaluation agree on the same declared
features and semantics; all supported build configurations pass the existing
serving/golden checks; each removed symbol has no live caller. API removals
remain a versioned compatibility change, not an incidental cleanup.

### 6. Ablate useful complexity, not evidence

**Priority:** model validity, evaluation through the serving API, data coverage,
and complete-loop speed come first. Before a feature-removal experiment,
inspect the feature plan: identify the passes, buffers, pooling or head work
that would actually disappear. Fewer coefficients/columns alone are not a
compute saving; shared accumulations often remain necessary. Defer removals
with no expected compute benefit unless they resolve a named scientific
failure. Measure the realized saving after any retraining and composition.

| Ablation | Method | When removal is justified |
|---|---|---|
| Feature groups with removable extraction work | Low priority otherwise. Use cost/contribution tools to identify actual removable passes/buffers, then retrain with matched data and pack/calibrate identically. For stochastic fits, use at least three paired, recorded initialization/order replicates. Keep the full-feature research path. | Predeclared perceptual noninferiority, complete codec-floor/identity gates, and measured extraction/loop savings. A scientific-only ablation needs a named question; zeroing coefficients is not a speed measurement. |
| Optional losses, transforms and heads in that recipe | One component changed at a time; validate train/serve semantics first. Use the exact served bake for the verdict and high-fidelity checks. | Removal survives held-out content and independent fixed-target quality/cost evaluation. A better composite alone is insufficient. |
| Buffered versus fold execution paths | Use the existing fold-engine retirement plan and caller/parity matrix, including HDR, cancellation, non-four-scale, reference reuse, legacy profiles and minimal-feature builds. | All supported callers have a replacement with demonstrated numerical and cost behavior. This is implementation retirement, not a new model experiment. |

Run the registered **all-codec floor-coverage control, including rav1e**, before
interpreting a floor failure as proof that a model component is unnecessary.
Repeating a deterministic linear fit with different unused seed labels is not
replication; use held-out content and reference-level uncertainty for that arm.
Do not repeat already-exhausted capacity/hinge sweeps. Keep the scientific work
to one justified challenger and a small number of discriminating arms; qualify
on independently judged quality, target error, bytes, passes, latency and memory.

### Archive and storage cleanup come last

Move `regression-report.md`, `zengrid-proposal.md`, the completed
`STREAMING_FOLDAPP_PLAN_2026-07-26.md` and `REPRODUCE_V47.md` out of the root
navigation after updating references. Retire the old `scripts/hetzner/` launch
entry points after caller/reproduction checks; its `rebuild_derived.py` is a data
algorithm requiring separate treatment.

Keep published API compatibility, goldens, retired dense/wide twins, unique
recipes, split/era registries, failed-result evidence and deployed `site/`
assets. The 56 experimental bakes have real compiled consumers. Exact duplicate
bake copies offer only 1.67 MB of savings and need include/URL rewiring first.
Two old local `perf.data` captures offer about 409 MB after a fresh liveness/
evidence check. No blanket `git clean`, dataset deletion or build-directory purge
is part of this plan.

**Execution order:** establish §0; simplify the workflow in 1 and prioritize
surface-based evaluation in 1a. Retire demonstrably obsolete tools in 2 while
that integration proceeds; then 3 → 4;
then the compatible portions of 5, followed
by the bounded ablations in 6. Storage cleanup is optional. Each implementation
change closes its callers, checks and documentation before the next begins.
Success means fewer supported decisions and fewer ways to obtain a plausible
wrong result—not a quota of deleted files.

---

<details>
<summary>Original September 6 feature-layout registration and measured increments</summary>

## Historical plan — retiring positional feature layouts

**Status:** pre-registered 2026-09-06, before any code in this lane. Gates below
are written first and are never edited to match a result. A gate that fails is
reported failed, not re-scoped.

**Design this executes:** [`FEATURE_SYSTEM_DESIGN_2026-09-05.md`](FEATURE_SYSTEM_DESIGN_2026-09-05.md)
(phases 1-5 landed; see [`PLAN_FEATURE_SYSTEM_2026-09-05.md`](PLAN_FEATURE_SYSTEM_2026-09-05.md)).
**Identity layer:** [`FEATURE_SET_IDS.md`](FEATURE_SET_IDS.md).
**Defects:** [`FEATURE_DEFECTS_AUDIT_2026-09-05.md`](FEATURE_DEFECTS_AUDIT_2026-09-05.md).

---

## 0. The ruling, and what it means concretely

**USER RULING (2026-09-06, verbatim):** *"get rid of the cruft and confusion,
the technical debt here is a huge problem for research and production serving
imo. a 372 layout where the bake skips features and features aren't computed is
a bad contract"*

That sentence names a specific artifact, and it exists. **MEASURED 2026-09-06**
(`bake_block_profile` over `zensim/weights/*.bin`; full output at
`/mnt/v/output/zensim/purge-2026-09-06/shipped_bake_block_profiles.txt`):

| shipped bake | profile | declares | layer-0 rows | caller lines READ | positions the runtime carries and nobody reads |
|---|---|--:|--:|--:|--:|
| `d_sdr_add156_id100_negrich_dial_2026-09-05` | **D (default)** | 372 | 372 | **28** | **344 (92.5 %)** |
| `d_sdr_add156_dense_dial_2026-08-31` | D (era-1) | 372 | 372 | 28 | 344 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` | **B** | 372 | 372 | 95 | 277 (74.5 %) |
| `bhdr_linear_shaped_cvvdpmix_2026-07-12` | **BHdr** | 372 | 372 | 133 | 239 |
| `v47_strict_qat_native_2026-05-27` | **A** | 372 | 372 | 285 | 87 |
| `c_sdr_purity944_2026-08-29` | **C** | 944 | 667 | 667 | **277, via `FeatureTransform::Drop`** |
| `c_hdr_l1t1944_2026-08-29` | **CHdr** | 944 | 697 | 697 | **247, via `FeatureTransform::Drop`** |

Two distinct lies ride the same wire, and the ruling names both:

* **"the bake skips features"** — C and CHdr declare a **944** caller width and
  carry 247-277 `drop` transforms. The runtime must emit 944 positions so the
  bake can throw a quarter of them away.
* **"features aren't computed"** — D declares **372** and the runtime, correctly,
  does not compute `f156..371` at all (`V1PoolsMode::Off`). Those 216 positions
  are **written 0.0 because the layout says an id lives there and the plan does
  not populate it**. A consumer cannot tell that zero from a measured zero.

**And nothing declares what it reads.** MEASURED: **0 of 11** shipped bakes carry
a `zentrain.feature_set_id`. The mechanism landed in phase 4
(`feature_layout::declared_layout` + `dense_slots_of`) and no shipped artifact
uses it.

## 0b. THE CONTRACT (five lines)

1. **A bake declares the feature ids it reads.** By id, in its own metadata — not
   by a width, not by inference from which layer-0 rows happen to be nonzero.
2. **A table stores exactly the ids it holds.** An id it does not hold is
   **ABSENT** — no column, no zero.
3. **The runtime computes exactly the declared ids** (at kernel granularity) and
   **emits exactly those, dense, in ascending id order**.
4. **Research and production speak the same contract** — same ids, same layout
   type, same refusal, differing only in plan breadth and provenance output.
5. **A mismatch is a loud, named refusal.** Never a structural zero, never a
   positional prefix, never a silent `Drop`.

## 0c. Non-negotiable: shipped SCORES do not move

Every increment is byte-identical on shipped output. The wire, the API and the
code shape change; the numbers do not. Public API changes ARE authorized by the
ruling but are **batched under `CHANGELOG.md` "QUEUED BREAKING CHANGES"** for
0.3.0. **No crates.io publish in this lane.**

---

## 1. Standing gates — every increment, no exceptions

| # | gate | how |
|---|---|---|
| **P-BYTE** | No shipped byte moves. | `zensim/tests/v1_golden_bytes.rs` (incl. the non-tight fixture) + `fold_engine_parity.rs` + `feature_invariants` + a full `to_bits()` dump over the 20-cell parity matrix (`zensim/tests/common/parity_cells.rs`), diffed before/after. |
| **P-SCORE** | No shipped bake's prediction moves. | `bake_verdict --full-json` on a fixed bake + fixed root, before vs after: the diff is the **wall-time line only**. |
| **P-SERVE** | Servability census stays at **zero refusals**, **in every cargo feature set**. | `serving::tests::every_shipped_profile_is_servable` (no filesystem; **moved out of `feature_plan` on 2026-09-06** — it was gated on `feature-regime-v2`, so it was blind in exactly the builds that were broken, see below) + `serve_custom_bake --census` (filesystem tier, `zensim/weights` + `--fulleval-dir`) + `scripts/serving_matrix.sh` (the cross-BUILD diff). A new refusal is a gate FAILURE, never a known limitation. |
| **P-TEST** | `cargo test --workspace` green; `just clippy` (`-D warnings`) green; `cargo fmt` clean. | |
| **P-API** | Every public-API delta is **ENUMERATED here** and mirrored into `CHANGELOG.md` QUEUED BREAKING CHANGES **in the same commit**, with `cargo public-api` output attached. `cargo semver-checks` is RUN and its verdict recorded. | `just api-doc-check` regenerates `docs/public-api/*.txt`. |
| **P-OWNER** | One owner per task. Any converter is an extension of the canonical tool, never a new script. | grep + a test asserting any wrapper agrees with the owner. |
| **P-APPEND** | Append-only. No id renumbered, no registry entry edited or deleted. | the existing registry test. |

**P-PERF (reported, not gated):** the `zensim_D` bench arm must be unchanged or
better. The box is shared; **measure only when idle**, and check
`~/tmp/fastclass2_w4_deferred.log` before any pinned sweep. If the box is not
idle, report `NOT MEASURED` — never a number taken under contention (the
2026-09-01 own-process-contention finding).

---

## 2. Increments

Each lands as its own commit, each independently gate-clean.

### A. Inventory (measured)

Every legacy concept enumerated with `file:line` and classified **DELETE /
REPLACE-BY-PLAN / DEPRECATE-SHIM / KEEP**. Recorded in
`benchmarks/cruft_inventory_2026-09-06.md`. No code change.

**Gate A.1** — every concept in the ruling's scope appears with a class and a
count: regime enums and `--regime`, width literals (156/228/265/289/300/372/376/
504/720/924/944/956), structural-zero producers, `FeatureTransform::Drop`
skipping, `V1PoolsMode`/`V1FreeExtras`/`skip_unread_pools`, `from_block_profile`,
`wide_bake_v2_read`, `caller_input_width` positional arithmetic,
`prep_bake_input_f32` widening. A concept found later that is not in the
inventory is an inventory failure, recorded as such.

### A-bis. FOUND BY A DOWNSTREAM LANE (2026-09-06): increment 2A's dense flip was correct on a DEFAULT build only

Increment 2A proved "no shipped score moves" three ways, and every one of those
proofs ran on a default-feature build. **`feature-regime-v2` is default-on but
not unconditional**, and the gather that serves a dense declaration lived behind
it — so `A`, `B`, `BHdr` and `D` were served the POSITIONAL PREFIX in every
`default-features = false` build: `B` 48.17 → 13.49 on a blur, `D` 48.43 →
−213.15, all 48 cells wrong, no error. Two in-workspace consumers were exposed,
one of them published (`zensim-regress` 0.4.0).

**Fixed by ungating `feature_layout`** (free: it needs only `feature_set_id` +
`feature_defs` + `mlp::Model`, and every shipped dense bake's highest declared id
is < 372, which the legacy v1 walk already emits — measured cost, same-parent
A/B: `--no-default-features` `.rlib` +4.29 %, default +0.05 %) and by making
`candidate-profiles` require `feature-regime-v2`, so every profile a build can
name it can serve.

**Two things this plan should carry forward.** (1) **P-SERVE and P-BYTE are
per-FEATURE-SET obligations, not per-commit ones** — a proof that ran only under
default features does not cover the contract. (2) **A gate `#[cfg]`-gated on the
same feature as the code it protects is not a gate**; the same shape produced a
second instance in the same lane, where `Cargo.toml`'s `include` allowlist (only
exercised by `cargo package`) was missing six `include_bytes!` targets and the
published crate would not have compiled.

Record: [`../benchmarks/dense_serving_ungate_2026-09-06.md`](../benchmarks/dense_serving_ungate_2026-09-06.md);
ledger `DATASET_HISTORY.md` §3.55.

### B. Bakes declare their ids — `bake_dial_refit densify`

The owner tool gains a mode that rewrites a bake to the dense contract: layer-0
rows, scaler mean/scale, feature transforms and params, bounds and sparse
overrides all **permuted and packed to the read set**; `n_inputs ==
caller_input_width == |read set|`; **zero `Drop` transforms**; the dense
`zentrain.feature_set_id` stamped. Old bakes are **kept on disk** as retired
copies; `zensim/weights/manifests/` updated.

**Gate B.1 (prediction identity)** — for every converted bake, `bake_verdict
--full-json` before vs after differs **only in the wall-time line**. Not a
tolerance: a byte diff.
**Gate B.2 (score identity through the runtime)** — `Zensim::compute` on the
20-cell parity matrix returns bit-identical `score`, `raw_distance` and
`mean_offset` for the dense bake and its wide original.
**Gate B.3 (no Drop survives)** — every converted bake has zero `drop` transforms
and `n_inputs == caller_input_width`. Asserted over `zensim/weights/*.bin`.
**Gate B.4 (the declaration is READ, not decorative)** — a converted bake whose
`zentrain.feature_set_id` is removed must **refuse or differ**, proving the
runtime gathers by the declaration rather than by a positional prefix. A negative
control: an undeclared bake of the same width scores differently.
**Gate B.5 (retired copies)** — the pre-conversion bytes remain on disk with their
sha256 recorded.

### RESULTS — A and B, 2026-09-06

**Increment A: DONE.** `benchmarks/cruft_inventory_2026-09-06.md`. Gate **A.1
PASS** — every concept in scope carries a class and a count. The headline
INVERTS the obvious prior: the production positional layer is thin (**11** width
literals in `zensim/src`, every one already a `const`; **0** inline) while the
debt is in tests (**343**), consumers, and 105 `--regime` call sites.
`wide_bake_v2_read` has **1** production caller and **0** test callers.

**Increment B: the TOOL and the DECLARATION are DONE; the CONVERSION is
BLOCKED for the 944 class, and the consumer side is NOT YET WIRED.**

| gate | result |
|---|---|
| **B.1** prediction identity | **PASS on 9 of 11**, and the 2 exceptions are a NaN class, not a numeric one — 16 of 512 probe rows on `v47_strict_qat_native` and `bhdr_..._anchored2`, every one of them a dropped line whose own value was NaN. `fma(NaN, 0.0, acc)` is NaN, so a zero weight row still poisons the wide bake. Reported with a count, never silently allowed. |
| **B.2** score identity through `Zensim::compute` | **PASS on 8 of 11** (bit-identical served score on real pixels, `serve_custom_bake --census`). **FAIL on the three append2-bearing bakes** — see §5 of `benchmarks/dense_bake_contract_2026-09-06.md`. Cause MEASURED and it is a PRE-EXISTING defect, not densify's: `Plan::for_bake`'s identity-layout branch derives `append2_dst_activity: true` and its id-space branch derives `false`, and the canonical extractor defaults **false**. Shipped C and CHdr therefore serve on a BANDVIS formula their weights never saw, worth **0.87 / 0.31** zensim points. |
| **B.3** no `Drop` survives | **PASS** — every densified output has 0 `drop` transforms and `caller_input_width() == n_inputs()`, asserted in the tool before it writes. |
| **B.4** the declaration is READ | **PASS** — `an_explicit_feature_id_list_resolves_to_that_dense_layout` pins both halves: the declared bake gets the dense layout, and the SAME width with the declaration removed falls back to identity. Plus a strict-parse gate over duplicate / descending / unparseable / empty / out-of-range. |
| **B.5** retired copies | N/A yet — no shipped bake has been REPLACED. |

**Two blockers, both measured, neither hand-waved:**

1. **The `append2_dst_activity` skew** (above) blocks densifying C and CHdr AND
   blocks increment D, because both would adopt the honest `false` and move
   shipped scores. The fix is one line and it is **a user decision**, not a
   lane's. Registered in `CLAUDE.md` "Known Bugs".
2. **The consumer side still slices positionally.**
   `bake_runtime::score_row` — the DEDUP-M canonical dispatch every eval tool
   inherits — copies `row[..n_inputs]` and zero-pads. `zensim`'s runtime gathers
   by declared id; `zensim-validate`'s does not. **So a dense bake would be
   MIS-SCORED by `bake_verdict` and every sibling**, silently, by reading the
   first `|read set|` POSITIONS instead of the declared IDS. Swapping any shipped
   bake to dense before that is wired would be a data-corruption bug, so **no
   shipped bake was replaced.** This is increment B-2 and it is the next step.

The width-floor fix landed in the same pass closes the OTHER half of that hole
(a corpus narrower than the bake is now refused rather than zero-filled).

### C. Tables store exactly the ids they hold

A converter **at the canonical owner** (the `pack_*` / extract tool, never a new
script) rewrites a wide table to dense-by-id: drops absent-id columns, stamps
`feature_set_id` in `_MANIFEST.json`, preserves row order and every kept value
byte-for-byte.

**Gate C.1** — on a converted table, every kept column is **bit-identical**
(`to_bits()`) to its source column, and the row count and row order are equal.
**Gate C.2** — a bake scores **bit-identically** from the dense table and from
its wide source (this is phase 4's G4.1 applied to a real stored artifact rather
than a synthetic one).
**Gate C.3** — the dropped columns were **all-absent**, i.e. every dropped column
is a structural zero for the whole table, proven by a full-column scan, not
sampled. A column with any nonzero value is NEVER dropped; if one is found the
increment stops and reports it.
**Gate C.4** — `_MANIFEST.json` carries `feature_set_id`, `build_commit`, and
per-file sha256; the source root is **not modified and not deleted**.

Scope here: the eval roots, the eval instruments (dial/corruption grids), the
dial anchors, and the fast-class legs. **bigcodec and KADIS are REGISTERED as
fleet jobs** through the existing `JobKind::Feature` executor, **not run in this
lane** — they are millions of rows.

### B-2. The consumers gather by id — **DONE 2026-09-06**

`bake_runtime::score_row` takes a caller-sized `&mut [f32]` scratch and fills it
positionally. It must instead take a per-bake row adapter that knows the bake's
declared layout: identity ⇒ today's copy, byte for byte; dense ⇒ a gather.
`score_row_minmax` has the same coupling (it indexes `transforms[i]` and `row[i]`
at layer-0 positions) and needs the same adapter.

**Gate B-2.1** — for every bake that exists today (all identity layouts) the
scratch fill is byte-identical and `scripts/verify_verdict_identity.sh` reports
**0 mismatches** on a fixed bake + root.
**Gate B-2.2** — a dense bake scored through `bake_verdict` agrees BIT-EXACTLY
with the same bake scored through `Zensim::compute` on the same pixels.
**Gate B-2.3** — a dense bake reaching an un-migrated scorer REFUSES rather than
slicing. No positional fallback survives that could quietly serve the wrong ids.

**RESULTS.** `bake_runtime::CallerGather` (`Positional` | `ByFeatureId`), built
once per bake and threaded through `score_row`, `score_row_minmax`,
`score_with_bake_alloc` and all six scoring bins; resolved through
`zensim::declared_feature_ids`, the ONE owner both sides read.

| gate | result |
|---|---|
| **B-2.1** identity bakes unchanged | **PASS.** `positional_gather_reproduces_the_old_fill_exactly` compares the arm against an explicit reimplementation of the OLD code at three width regimes (`dst` shorter, equal, longer than the row) on `to_bits`. `cargo test --workspace` green. |
| **B-2.2** dense == wide | **PASS.** `a_dense_bake_scores_like_its_wide_twin_through_score_row` — two synthetic bakes that are the same function of the same three ids score to the same bits, with a NEGATIVE CONTROL asserting a positional slice of the dense one does NOT agree (so the test cannot pass vacuously). |
| **B-2.3** no silent positional fallback | **PASS by construction.** The parameter is not an `Option`; a new scorer cannot omit it. |

**End-to-end P-SCORE evidence.** `bake_verdict` on shipped **B** at the default
(current-extractor) 372 root, with the whole B + B-2 stack in place, reads
**CID22 SROCC 0.8821166166351724** and **kon504 |0.5193759178072009|** — which
reproduce the two values `CLAUDE.md` recorded for that bake on that root BEFORE
this lane existed (**0.88212** and **|0.51938|**) to every published digit. Two
independently-recorded numbers, both unmoved.

**Honest scope:** B-2.3 is enforced by the type, not by a runtime refusal. A
scorer that deliberately passes `Positional` for a dense bake still mis-scores
it — the guarantee is that doing so is now a written decision at the call site
rather than the default.

### D. Delete the superseded derivations

`ComputeSet::from_block_profile` and `fold_engine::wide_bake_v2_read` collapse
into `Plan::derive_with_layout`. This is phase 5's own named next step; its unit
evidence already exists (`from_block_profile_agrees_with_the_id_space_derivation`)
and phase 5 correctly refused to claim it without the census.

**Gate D.1** — the 445-bake servability census is re-run and is **identical**:
same SERVED count, same per-bake plan. Not just "still zero refusals" — the same
plan for the same bake.
**Gate D.2** — both symbols are **gone from the tree** (grep returns only
historical prose in `benchmarks/`), and every test that exercised them exercises
the id-space derivation instead.

### E. The runtime speaks ids

`Zensim` gains id-addressed feature access; positional accessors are **deprecated
with shims for one release**, not removed. `feature_set_id` gains the
**layout-free** id form (`<compute>/<era>#<hash>`) with every existing
`<compute>@w<layout>/<era>#<hash>` kept as a **registry alias** — with a dense
wire the layout component is redundant, and the user has authorized naming
changes.

**Gate E.1** — every existing registered id string still resolves, via alias, to
the same slot set and the same `slots_hash8`.
**Gate E.2** — the id-addressed accessor and the positional one return the same
value for every id in every identity layout.
**Gate E.3** — P-API: the delta is enumerated in §3 below before it lands.

### F. `--regime` becomes a derived print

`--regime N` resolves through the registry and **prints its derived meaning**
(phase 5 already does this); the remaining width-keyed branches become alias
lookups. Regime enums that no longer select behaviour are deleted or deprecated
per §3.

**Gate F.1** — every `--regime N` invocation in a committed script produces a
**byte-identical verdict** (modulo the wall-time line) before and after.
**Gate F.2** — the `--regime 944` silent-mis-scoring class stays structurally
unreachable (phase 5's G5.2, re-run).

### G. Docs

`FEATURE_SET_IDS.md` rewritten to the clean contract with the era zoo demoted to
a **migration appendix**; `CLAUDE.md`'s regime/pool/skip sections rewritten **in
place** to the new truth (dead guidance deleted, measured history left in
`benchmarks/`); cookbook + `DATA_PROVENANCE` pointers for converted tables;
ledger ROUND rows.

**Gate G.1** — no doc left in tree describes a deleted symbol as live. grep for
each deleted symbol across `*.md` returns only past-tense/benchmark prose.

---

## 3. Public API delta — enumerated BEFORE it lands

Pre-registered so that a delta appearing outside this list is a gate failure.
`docs/public-api/zensim.txt` is the supported surface. Everything here is
**QUEUED for 0.3.0**, batched, not shipped piecemeal, and **nothing is published**.

**Proposed REMOVALS (breaking; all currently in the supported surface):**

| symbol | why it goes |
|---|---|
| `feature_v2::FeatureRegime` (6 variants) | a regime is a width plus a set of structural zeros — exactly the concept being retired. Replaced by a declared feature set id. |
| `feature_v2::ZensimV2Result::regime()` | returns the above |
| `feature_v2::V1PoolsMode` (4 variants + `CARRIER_SLOTS`) | "which pools ran" is a `Plan` property, derived from the declared ids, not a caller-set mode |
| `feature_v2::V1FreeExtras` (3 variants) | same |
| `feature_v2::V2NewFeatureToggles::v1_pools` (field) | same |
| `feature_v2::ZensimV2Result::v1_pools()` | same |
| `feature_set_id::FeatureSetId`'s **LAYOUT component** — `new`/`from_slots` lose their `layout_width` parameter, `layout_width()` is removed, `Display` emits `<compute>/<era>#<hash8>` | added 2026-09-06 by increment E, per this section's own decision rule (a delta appearing outside this list is a gate failure, so it is listed BEFORE it lands). **With a DENSE wire the layout token carries no information the hash does not**, and it actively lies: increment B-2 measured a dense `B` and its wide twin producing `basic+peaks+masked+iw@w95/unknown#9403d2a7` and `…@w372/unknown#9403d2a7` — the SAME compute tokens and the SAME slots hash, differing only in a component that describes the wire shape rather than the feature set. `check()` then reported `LayoutDiffers` on every dense-bake/wide-table pair, which is noise in a refusal surface. The width does not vanish: it moves to `FeatureSetRef::layout` (`Option<usize>`), where it is a property of the ARTIFACT rather than of the set, and `LayoutDiffers` fires only on a real shortfall (the consumer needs a wider row than the producer emits). **Every `@w<N>` string still PARSES** — the width is accepted and discarded — so every registry key, every stored `feature_set_id` and every published id resolves to the same set and the same `slots_hash8`. |

**Proposed ADDITIONS:** an id-addressed feature accessor and the declared feature
set of a result. Named in the increment that lands them; each is additive.

**DEPRECATE-SHIM (kept for one release, `#[deprecated]`):** positional feature
accessors and any width-returning method a downstream may call.

**Decision rule if an increment needs surface not listed here:** it STOPS and
adds it to this list in its own commit first.

---

## 4. Sequencing, risk, and what this lane will NOT do

A → B → **B-2** → D are ordered (nothing may be CONVERTED before B-2, and D
needs B's census). C is independent. E and F need B-2. G is last.

**Both of D's symbols are additionally blocked on the `append2_dst_activity`
decision**, because collapsing `from_block_profile` into the id-space derivation
adopts `false` and moves shipped C / CHdr scores. That is the pivotal blocker of
this whole program and it is one line plus a decision.

**The largest risk is C**, because it rewrites stored artifacts. Mitigated by:
the source root is never modified, gate C.3 scans every cell of every dropped
column rather than sampling, and the dense output is proven to score
bit-identically before any consumer is repointed.

**Explicitly out of scope:** any kernel/arithmetic change (this is dispatch and
wire shape); any renumbering (append-only stands); flipping F4 or F5 (they stay
registered `Proposed`); any crates.io publish; bigcodec/KADIS conversion (fleet
jobs, registered not run); `zenanalyze-api` (frozen).

</details>

### Qualification interface registration — September 7 continuation

Extend the existing `freeze_check` decision owner with `--qualify --fulleval
<json>`. It reports `qualified`, `failed`, or `incomplete` plus each finding;
research selection remains distinct. Require the Rust scoring identity,
resolved compatible provenance, both G-ADDR tiers and all five codec families,
and the five product gates in `MODEL_SELECTION_SCORECARD.md`. The latter use
content-bound evidence records (`product_evidence`): gate state, candidate
SHA-256, Rust surface, instrument, positive sample count, and the path/SHA-256
of the measuring owner's JSON artifact. Missing or stale evidence is incomplete,
never pass. Known failures remain failures even when other evidence is missing.
The board may display this report, but must not synthesize qualification from
composite or from the absence of a failing badge.

### Spline-writer completion — September 7 continuation

The transform-screen top-N/max-ID semantics landed in `6822f095`. The four
remaining spline writers and obsolete V9 full pipeline are now archived after
caller checks; current generic fitting uses the actual Rust pre-calibration
surface. Historical fit and runtime-boundary differences, the exact f32
V47 primitive fixture and intentional corrections are recorded in
[the spline-owner audit](../benchmarks/spline_owner_cleanup_2026-09-07.md).
The distinct equal-count training fitter remains a named recipe.

### Feature-plan completion — September 7 continuation

C/CHdr now use explicit IDs and training-consistent activity semantics. Both
legacy plan derivations are retired. The 984-case canonical-pixel census, its
semantic negative control, HDR parity and eight feature-build arms pass.
The broad historical-bake score correction is recorded explicitly in the
[feature-plan audit](../benchmarks/feature_plan_cleanup_2026-09-07.md). Fresh
wall-time evidence was rejected for contention; the useful cheap-wide features
and buffered compatibility paths remain.

### Remaining admission and target-loop interface registration — September 7

Before implementation: extend the unpublished validation loader's existing
column-discovery owner to accept CSV header names and expose header-only feature
inspection to trainer admission. Admission checks the selected IDs against
actual columns and each resolved producer, including per-file manifests and
formula/decoder stamps. Historical replay records unresolved semantics but
cannot excuse missing requested columns or malformed declarations. Dense tables
remain explicitly refused by callers that have not migrated; no zero-fill claim.

The unpublished standalone `zensim-target` tool will expose
`target_search_with_bake(..., &mut zensim::BakeScorer)` for the actual candidate
measurement caller, sharing the existing search controller. `TargetResult.profile`
becomes optional so a candidate cannot be mislabeled as B. Named-profile calls
preserve their profile; the candidate experiment records the complete model/head
composition by content hash. No supported `zensim` signature is removed.
The existing demo matrix will accept explicit sources/candidates and emit
measurement artifacts; it must fail on missing inputs rather than skip them.

### Admission and positional CLI retirement — September 7

The trainer/metadata checklist items are complete. See the [implementation and
verification record](../benchmarks/trainer_admission_cleanup_2026-09-07.md).
Unknown decoder declarations do not confer qualified provenance; historical
replay remains explicitly unqualified.

### Current documentation and launch-tool consolidation — September 7

CLAUDE is now the concise working-rules file; WAVE_PLAYBOOK is the current
workflow/tool map; RESEARCH and ITERATION_PROTOCOL point there. Original dated
notes are retained under `docs/history` (exact originals in Git). Completed
root reports/plans were moved there and Markdown links updated. Contracts for
splits, feature identity, reproducibility and selection remain separate owners.
The later TID train-only ruling now reaches the split checker. C/CHdr and
corruption decisions have explicit completion updates.

Four unused Hetzner launchers are retired after caller checks. The unique
`rebuild_derived.py` stays. The historical worker selector reads preserved
private aliases from local configuration; mapping equality was checked. The
full linter's 26 pre-existing private-identifier hits are resolved without
exemptions, and all 604 remaining script checks pass. Model/data evidence and
unique compiled experimental consumers remain; no bulk deletion of bakes,
profiler evidence or data was justified by the caller/storage audit.


### Scientific completion — September 7 local / September 8 UTC

The bounded scientific checklist items are complete. Nine actual fits completed:
three unchanged A_plain seeds and three unchanged H_anchorlad seeds exactly
reproduce their September 6 verdict/ladder panels; three paired H+rav1e seeds
measure the missing-codec data control. CID22 means are 0.889095, 0.874350 and
0.875244 respectively. All six H models pass the identity/negative-tail contract
but still fail all five codec-floor regression gates. No new winner is qualified.
See [the registered controls and full seed results](../benchmarks/cleanup_scientific_controls_2026-09-07.md).

The [360-cell actual-loop record](../benchmarks/cleanup_target_loop_2026-09-07.md)
reports emitted bytes, reconstruction/score checks, target errors, passes,
latency/memory and independent judges. It fails the three-pass target-error
screen and makes no encoder-RDO or contended performance claim. This evidence
does not justify feature/head/loss deletion; buffered/shared-feature paths retain
real callers and remain deliberately supported under the user's compute priority.

Qualification now requires compatible training and evaluation admission for
every ensemble member/companion. Root and per-file sidecar hashes invalidate
reuse even when admission is unknown or fails early. Historical replay and
unstamped tree companions cannot acquire a provenance pass from the primary.
The actual known-primary/unknown-companion and decoder-sidecar controls pass.
The latest full evaluations preserve all six scientific panels, complete all
27 coherence cells, and receive the explicit owner decision `failed`.


### Final acceptance record

The operative checklist is complete; [the completion record](../benchmarks/cleanup_completion_2026-09-07.md)
links the exact removals, retained conditional work, nine training fits, target
loops, final CI/board checks and evidence locations. A completed cleanup does
not confer model qualification. Superseded proposal/status paragraphs above
remain dated history, not another pending work queue.
