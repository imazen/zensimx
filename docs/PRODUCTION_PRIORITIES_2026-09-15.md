# Controlling production plan — September 15, 2026

**User-directed priority, updated after the recovery assessment.** This plan
supersedes conflicting older task lists, proposed work orders and paused goal
text in this repository. Preserve higher-level instructions, split protections,
registered numerical gates and historical evidence. An old experiment remains
a record of what happened, not authority to restart its abandoned work.

## Owner queue update — October 26, 2026

Zensim training and evaluation take priority. Finish the registered Mlic AIC2026
public_train comparison and preserve its existing scientific and production
qualification gates. Model-agent availability is separate from experiment
progress; verify run artifacts and failures directly.

**Queued: seamless HDR frontend, after adoption of a stable candidate.** Record
the adopted candidate's exact identity and the evidence supporting the decision
before starting the frontend work. A provisional training winner alone does not
release this dependency. The frontend should reuse the adopted scorer and the
existing color/HDR contracts; its implementation is not started by this queue
entry. “Optimal” remains relative to the agreed evidence and gates, not a claim
of a proven global optimum.

The owner's follow-up asks whether the zenpapers weighting and ordering
proposals were adopted. The registered AIC continuation still uses inherited
collection masses with pair-acceptance correction and `--pair-sampling uniform`.
These prospective recipe comparisons remain outstanding:

- Binomial vote likelihood retaining ties.
- Collection masses proportional to the square root of effective sample count,
  preserving the registered total new-data mass in a future comparison.
- Per-item confidence weights with their linking uncertainty accounted for.
- Same-source, between-codec sampling strata at matched distortion distance.

The existing stratified sampler is not evidence that this specific sampling
scheme is active. Each proposed change needs its own preregistered comparison;
the running AIC arms and their selection rule stay frozen. The source review is
`zenpapers/docs/mlic-recipe-review-2026-10-26.md`, committed revision `4e81c03a`.

For the queued HDR metric frontend, preserve the SDR fast path and native input
precision; establish cost with measurement before claiming no performance
penalty. The current declared-HDR route uses f32 rows, SIMD PU encoding and the
shared streaming feature walk (`feature_v2_stream.rs`, `color.rs`). It still
needs qualification of near-identity arithmetic, the PU floor in dark regions,
display parameters and mixed SDR/HDR behavior. The existing luminance
cancellation diagnostic is in `metric/bake.rs`; changing feature arithmetic
requires a new formula revision and qualification. Reuse the existing
[HDR plan](HDR_PLAN.md) and [PU21 reference](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2021_PU21.pdf).

**Owner refinement: maintain speed and seek better quality than PU21.** PU21
is the comparison control, not the predetermined frontend. Consult zenpapers
and the prior BHdr results before adding features. The October 26 evidence
review and proposed matched comparisons are recorded in the
[HDR/SDR alignment plan](PLAN_HDR_SDR_ALIGNMENT.md#october-26-2026-evidence-and-speed-requirement).
A cheaper approximation of PU21 alone does not establish a perceptually better
representation. Preserve native precision and the SDR fast path; qualify the
complete scoring, preparation and steering costs on permitted hardware.

## Production deliverable

The deliverable is a production-qualified Rev3 model with one useful target
score, fast complete Rust inference, useful spatial steering, correct declared
color/HBD/HDR inputs, and a thresholded integrity companion. Preserve a fast
option and a richer quality control while doing bounded science. An aggregate
ranking win, an attractive map or a finished implementation is insufficient.

The matching Squintly task is [the paid-study handoff](https://github.com/imazen/squintly/blob/main/docs/STUDY_READINESS_2026-09-15.md).
In a standard sibling checkout Squintly is `../../squintly` from this repository's
root. That handoff owns phone presentation, worker instructions, study design
and the pre-spend decision. Prepare it alongside this work; **do not wait for
paid labels to fix demonstrated implementation/controller failures**.

## Start from the actual completed state

Read, in order:

1. This plan and [the recovery result](../benchmarks/recovery_completion_2026-09-15.md).
2. [The scorecard](MODEL_SELECTION_SCORECARD.md),
   [data split rulings](DATA_SPLITS.md), [dataset chronology](DATASET_HISTORY.md),
   and `../DATA_PROVENANCE.md` relative to the repository root.
3. [The current owner map](WAVE_PLAYBOOK.md),
   [target protocol](TARGET_STEERING_PROTOCOL_2026-09-08.md), and
   [corruption activation contract](CORRUPTION_ACTIVATION_2026-09-13.md).
4. The relevant papers and summaries in `../zenpapers`, including subjective
   scaling, JPEG AIC methods and HDR imaging. Read primary sources before
   translating a summarized method into a new experimental protocol.

The September 15 result supersedes September 14's unfinished extraction/map
list. Baseline recovery, prepared-map reuse, native HDR serving, corrected HDR
extraction and native HDR judge generation have already been done. Reuse them
after receipt verification; do not recreate their pipelines.

Pinned implementation: zensim `e246d954ddc412fcc442c1190e994ecc4839a90e`,
zenmetrics `266967e9`, JXL encoder `c1ab16c9`. Fetch each repo before changing it;
preserve and rebase local work as needed, read its local instructions, and pin
new producing revisions. Existing checkout only. Do not wait on CI; use tight,
appropriate local checks and the repository's safe push owner.

Artifact authority: `/var/tmp/zensim-validation-2026-09-15/` (called `R` below).
The served `recovery-completion-2026-09-15/` packet beside the summer gauntlet
contains a replay archive and downloadable results. If local artifacts are
absent, recover that packet and verify its hashes; paths alone are not receipts.
Final model identity is `R/recovery/calibrated/FROZEN.json`; verdicts are
`R/qualified/*.full.json`, `*.measurement.json`, `*.qualification.json` and
the corresponding result files. Preserve these immutable artifacts.

| Frozen control | Role | SDR composite | Disposition |
|---|---|---:|---|
| `R915_y60_h32_ens5` | Rev3 fast control | .846044 | Product FAIL |
| `R915_basic228_h128_ens5` | Rev3 quality control | .875950 | Product FAIL |
| `MT914_matched_B` | Matched incumbent | .838237 | Comparison, not a newly qualified release |
| `MT914_matched_D` | Matched incumbent | .830760 | Comparison, not a newly qualified release |

Each recovered candidate has five equally weighted members, seeds
17101/17103/17107/17111/17113 and disjoint registered sampler windows.
Use complete calibrated compositions through `BakeScorer`, including any
companion and spline. B/D use their own Rev1 feature contract, never Rev3 rows.
No named profile or default changed. The completed fits took about 53 minutes
with two running together; this was not a five-minute experiment.

## Priority and dependency order

Work in the order below. A task may proceed while an independent prerequisite
is being prepared, but do not start another broad model/feature campaign.
Maintain one short execution ledger with hypothesis, input receipts, owner,
command, budget, result, failed cells and next decision. Do not invent another
scorer, controller, statistics implementation or report system.

### P0 — close evidence plumbing and prepare the human-study handoff

First enumerate the failed and unmeasured scorecard rows for the frozen controls
from saved owner output. Separate implementation defects, model deficiencies,
controller deficiencies and missing qualification evidence. Record historical
public-panel exposure. This inventory must not silently become TRAIN examples.

Close the decoder-declaration admission gap in older EVAL manifests using
actual producer receipts; if those cannot establish the contract, regenerate
the affected assets under a new version and retain the historical limitation.
Admit the registered Rev3 pixel identity and negative-tail probe to G-ADDR.
Existing exact-identity tests do not substitute for a missing qualification
instrument. Do not fabricate decoder declarations or mark missing data passed.

In Squintly, complete the separate agent work packet before full paid
collection. Its current historical study is not a current-model, multi-observer
phone qualification. The human work is to provide perceptual judgments under
a verified presentation condition, not diagnose decoder/ICC bugs or choose
metric thresholds. Phone zoom must preserve source pixels without smoothing;
normal viewing and magnified inspection are different recorded conditions.
Unverified phone HDR is not HDR ground truth. See the Squintly handoff for
pilot, assignment, display and analysis gates.

Exit: the execution ledger names every missing instrument and existing owner;
the study task has an executable protocol, explicit blockers and a pre-spend
decision. This does not require a study to have run before P1 starts.

### P1 — fix attainable-target tails with the model held fixed

The richer model's three-shot p95 errors are JPEG 3.815, JXL 3.078, WebP
7.502; worst errors are 6.003, 8.984 and 17.814. Fast WebP p95 is 11.786.
These miss the registered three-shot bar: median <=0.5, p95 <=1, max <=3,
and undershoot by more than 1 <=1%. The one/two-shot bars remain unchanged.

Use `zensim-target::target_search{,_with_bake,_with_backend_and_bake}`,
`SeedCurve`, `native_probe` and `demo_matrix`. On admitted TRAIN source
families, save a trace for each encode: knob, decoded identity, score, bytes,
latency, seed/slope estimate, bracket and stop reason. Reproduce the observed
classes of failure on TRAIN; do not tune on the exposed validation images.

Classify misses before modifying the controller:

* Seed/slope error or too little TRAIN calibration coverage.
* Endpoint handling versus an interior request.
* Discrete representability or a nonmonotone codec/score response.
* Premature stop, invalid bracket or fallback behavior.

The current controller has a TRAIN seed, slope-based second shot and a
measured secant third shot; its default tolerance is one score point. Inspect
that contract against the product bar, but do not assume the stop tolerance
explains large tails. Compare fixed B/D and Rev3 models on matched requests,
admission and work budgets to distinguish controller and scoring effects.

Establish each image/codec's witnessed attainable range before scoring its
steering outcome. Bounds and oracle ladder measurements stay hidden from the
runtime controller and are charged separately. Min/max alone do not establish
that every intermediate target is attainable: preserve discrete witnesses,
uncertain cases and 100% request disposition. Evaluate 1/2/3 shots on the same
frozen requests, with per-codec TRAIN-calibrated seeds. No floor clamp, hidden
extra encode or post-hoc omission of difficult requests.

Exit: TRAIN mechanism diagnosis plus a frozen controller comparison ready for
the existing qualification owner. Advance only changes supported by TRAIN
evidence; report remaining failures even if no controller change can solve them.

### P2 — recover local quality ordering with bounded training changes

Aggregate quality is recovered; local ordering is not. CID22 B8–B9 correlation
is .509198 for B, .458658 fast and .479244 rich. Rich floor ordering passes
20/26 JXL cells versus mentor 25/26, and 16/39 rav1e versus 25/39. Both
candidates pass 25/39 JPEG cells versus mentor 26/39. These measured failures
motivate a TRAIN-side hypothesis, not fitting against those human responses.

Start with the proven SafeSyn/CID22 TRAIN recipe and existing Rev3 caches.
Test nearby-quality, within-image ranking supervision on source-disjoint
TRAIN development families, including high fidelity and codec-floor strata.
Keep ordinary honest low-quality encodes distinct from corruption. Record
loss/sampler changes and actual group weights, including checkpoint selection.
A monotone spline cannot repair reversed ordering.

Use the existing five-minute feature-development screen only for cheap
mechanism screening after builds/caches are ready. For the first experiment,
compare baseline versus one local-ranking change on basic228, with Y60 as
the speed control. Do not run a Cartesian product of scales, kernels, widths,
heads and losses. Predeclare a small screen budget and TRAIN advancement rule;
only a surviving hypothesis receives the established multi-seed confirmation.
If the small screen cannot resolve the question, report that limit instead of
claiming the feature set has reached its ceiling.

Add features or change arithmetic only when independent numerical checks and
a controlled TRAIN comparison identify missing information or a real defect.
Use the existing Rev3 per-scale planners and references; changing feature era
requires re-extraction and new identity. Prioritize removals that eliminate
real passes/buffers/work, not zero-weight columns with no runtime effect.

Exit: a reproducible TRAIN improvement with no registered TRAIN regression,
complete Rust packed/pixel parity and a measured cost. Freeze the full recipe
and composition before the next EVAL/public TEST assessment.

### P3 — make JXL spatial guidance deliver independent quality/byte value

The JXL complete-rectangle/max integration is fixed. The neutral control has
756 identical ladder pairs and the fast old/new comparison has 1,782 identical
records. Rewriting that adapter is not the next task. Native independent RD
still has content regressions; good JPEG M3a does not establish JXL utility.

Use existing `diffmap_block_coherence`, native intervention instruments,
JXL `zensim_diffmap_rd` and `rd_probe_analyze_2026-07-18.py`. On TRAIN,
separate three questions in this order:

1. Does predicted rectangle gain track the actual finite-edit Zensim gain?
2. Does actual Zensim improvement track independent-judge improvement?
3. Does the encoder spend bits in the useful places at matched quality?

Retain neutral/active, scalar-controller and actual-edit controls. Rectangular
hard-max terms are non-additive: summing tile gains does not estimate a union.
Measure map construction and query work, shot count and codec bytes. Report
per-source/content failures and uncertainty over source families.

The current native JXL adapter uses default refinement; the rich 91.73 ms
prepared benchmark includes opt-in finite-moment refinement. Do not attribute
that optional cost to this native adapter or assume enabling it fixes RD.
Compare that option only under a registered TRAIN intervention experiment.

Exit: the scorecard's independent-judge RD and coherence criteria on frozen
assessment. Sparse eight-family exploratory RD cannot alone qualify broad
content coverage. Resolve the JXL mechanism first, then adapt and qualify the
existing AVIF/JPEG/WebP integrations; a JXL pass does not qualify those codecs.

### P4 — train correct native HDR and preserve one public score

The native path is implemented and exercised: 214 SDR and 495 HDR audit pairs
per composition. Corrected HDR TRAIN has 7,425 pairs, 495 reference variants,
33 source families, live peaks and fresh CVVDP/PU-SSIM2. Do not reuse older
zero-peak/mislabeled HDR caches. UPIQ uses native EXR through zenexr/upstream
Rust EXR, not an 8-bit conversion.

SDR-trained weights regress on UPIQ: BHdr .753314 SROCC, fast .696017, rich
.704373. That rejects shipping these weights as unified HDR; it does not show
that a properly mixed-trained shared model is impossible. Try controlled
SDR/HDR TRAIN supervision with explicit target-scale alignment, source-family
separation and supported viewing metadata. Shared weights are the first
hypothesis. A metadata-selected internal head is a fallback only when TRAIN
evidence justifies its complexity; users still control one target score.

Read the HDR and subjective-scaling material in `zenpapers` with the Squintly
agent. Treat input correctness, display simulation and learned perceptual
calibration as separate claims. ICC must be interpreted by the existing full
CMS owner before comparison in a declared common linear space. Preserve
precision, transfer, primaries, luminance, range, alpha and orientation; never
relabel samples. Same-profile 8-bit shortcuts need their explicit supported
contract; they do not justify mixed-profile or HDR shortcuts.

The current native common-primary CVVDP judge models a BT.709 HDR display with
reference-measured peak. It is not wide-gamut phone-display qualification.
Neither a phone's HDR capability flag nor an HDR file proves its displayed
luminance or tone mapping. Keep SDR phone observations, wide-gamut SDR and
physically verified HDR conditions separate. Integer nearest-neighbor zoom
changes apparent artifact size; record it and do not merge magnified results
with ordinary-view judgments as if they were the same target scale.

Exit: independent native-input parity and an explicitly mixed-trained frozen
candidate meeting the existing SDR and HDR ranking/steering contracts. Missing
ICC/display/input coverage stays unqualified; no UPIQ-driven refit.

### P5 — qualify the integrity head as an error detector

Follow the scalar activation threshold before `min(perceptual, corruption)`;
ordinary encoder quality loss and spatial allocation must not activate it.
The September 14 pilot had two honest development activations, including a
JPEG q5 case lowered from 55.67 to zero. A negative perceptual score can also
mask an activation that incorrectly refuses steering. Check raw activation,
score lowering and prepared-session disposition separately.

Chronology matters: the later
[color-resolution report](../benchmarks/integrity_color_resolution_2026-09-14.md)
already restored the mobile color-managed fixture source. Its 705-pair audit
detects 30/31 severe proxies, 19/20 real bugs and 8/8 channel swaps, with 0/3
known-valid activations. Whole-image progressive-AC truncation on mobile source
8014 still has head probability zero and permits steering despite a negative
base score. Repair this demonstrated miss without turning valid severe
compression into an error. The original AVIF pink-cast producer cause remains
unresolved; do not convert it into a confidently labeled decoder bug.

Freeze a representative color-correct TRAIN packet, stratifying known bugs,
catastrophic failures, recoverable defects, valid low-quality outputs and
ambiguous cases. Developers establish provenance/causality. Human observers
can assess visible harm under a valid reference condition; they cannot infer
software causality from pixels. Ambiguous activations are not automatically
false positives or true bug detections. Calibrate the activation threshold on
TRAIN calibration families only, with separate fit/development families.

Extend the existing strict trainer/ZCTH exporter/Rust composition and
`corruption_gate_eval.py`; retain complete feature-read and f32 parity audits.
Require the current severity-aware contract, including honest activation and
native steering protection, before attaching the head to a release candidate.

### P6 — meet full runtime cost and close the final qualification packet

RGB8 1024² complete scalar p95 is 9.77 ms fast / 18.79 ms rich; prepared with
opt-in finite moments is 26.95 / 91.73 ms. At 2048² fast prepared/scalar also
slightly exceeds 3×. Relative D/fast-SSIM2 and incremental-memory clauses are
unmeasured. Whole-process RSS includes input buffers and is not incremental RSS.

Profile existing extraction, ensemble probes, max-map construction, optional
finite integrals and rectangle queries separately, then measure the full API.
Maintain mathematical/SIMD/reference parity. Optimize actual work and memory;
do not remove useful features merely because their weights are small. Use the
existing latency instruments, matching reference caching and refinement modes,
at least 30 accepted paired rounds and the strict resource admission rule.

Freeze final bytes, all heads/splines, controller, supported input contracts,
populations and gates. Run the existing full evaluation and qualification
owners end to end. Lead reports with the registered composite and its coverage,
rank/band/within-image panels, outlier envelopes, raw scatter density/geometry,
clumping, saturation, tails and spatial/target/runtime results. MAE is auxiliary.
Populate the gauntlet comparison and a dated discussion with real owner output;
codec q versus score axes must include measured negative values.

Exit: every applicable scorecard row has admissible evidence and passes for
the same final composition. A FAIL or INCOMPLETE prevents production-qualified
claims. A narrower SDR/codec capability can be reported honestly but does not
complete this all-purpose objective. Update profile/API docs only with the
proved scope. Publishing still follows the repository's release authorization.

## Rules for the next executing agent

* All fitting, feature/loss/seed/checkpoint choices, normalization, splines and
  controller calibration use TRAIN only. Preserve source-family boundaries.
* EVAL gates assess frozen candidates. Published TEST is allowed only where
  no EVAL exists, with declared exposure and no adaptive reuse; secret holdouts
  remain untouched. New paid training and confirmatory labels need distinct
  family roles assigned before mining/collection.
* Use failure classes to formulate TRAIN experiments. Do not repeatedly inspect
  named holdout images/residuals to tailor a model. Every repeated public-panel
  exposure belongs in the ledger; it is not fresh confirmation.
* Each bounded experiment gets a preregistration and fresh result directory.
  Retain failures and actual budget usage. Cheap screens are not qualification.
* Continue the highest-priority actionable task, with short progress reports.
  Do not spend cycles waiting on CI, rerunning unchanged passing tests, broad
  fleet searches already resolved in the history, or recreating completed data.
* Push reviewed work through `scripts/safe_push.sh -r <reviewed SHA>`, verify
  remote ancestry and leave commands/artifacts sufficient for another agent.

Success is measured evidence for a useful product, not the number of fits,
features, documents, tokens or human judgments collected.
