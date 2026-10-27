# Production release scorecard

## October 27 AIC development comparison

The owner directs the new Mlic/Neff/Vote/Order AIC TRAIN comparison to use
complete panels and other statistics **without strict SROCC rejection or a
scalar automatic winner**. Confidence joins after separate admission/fitting.
Review source/codec/local rankings, calibration/tails, seed/source uncertainty,
mixtures and production behavior together. Keep historical screen results
visible, but do not use their cutoffs to exclude arms from this new comparison.
See [the registered protocol](../benchmarks/byconstruction_2026-10-22.md#aic-adaptation-across-training-recipes-prospective-panel-study-2026-10-27)
and [current queue](../SESSION-RESUME.md#current-work--october-27-2026).

This is a prospective development comparison, not a retroactive change to
frozen experiments or a release waiver. In-sample AIC/full E21 panels remain
explicitly in sample. Preserve protected roles, pixel/numeric correctness and
provenance; freeze any eventual assessment before permitted EVAL reads.
Serving, steering, runtime and HDR requirements remain separate. No recipe
has been adopted through this comparison yet.

## September 14 split clarification

The later [user clarification](DATA_SPLITS.md#september-14-clarification-test-evaluation-when-no-eval-split-exists)
allows published TEST assessment of frozen candidates when a population has no
EVAL split. Freeze model bytes/composition, populations and gates before reading;
record exposure and prohibit adaptive fitting or selection against the results.
CID22 oracle TRAIN remains distinct from human TEST. Secret holdouts remain
untouched, and all fitting/calibration/feature/checkpoint decisions stay TRAIN-only.
This supersedes the blanket prohibition in the historical paragraph below,
without changing any numerical gate or authorizing secret/terminal access.

## September 13 split override (superseded as clarified above)

The later [user split ruling](DATA_SPLITS.md#september-13-user-ruling-train--eval-only-never-touch-test)
prohibits all test/terminal access, including the frozen-finalist reads proposed
below. Run gates only on explicitly admitted eval segments. An unavailable
permitted eval population leaves that gate incomplete; it never authorizes a
test read. Fitting, calibration and checkpoint selection use train only.

## September 13 corruption clarification

The later [activation and severity contract](CORRUPTION_ACTIVATION_2026-09-13.md)
supersedes the unconditional catastrophic interpretation of synthetic defect
labels below. Activation means integrity failure; valid low-quality encodes
and ordinary spatial allocation must not activate the catcher. Require zero
honest native activations and <=1% overall honest activations as well as score
lowering limits: an unchanged negative score can hide a false activation.
Review and freeze catastrophic/recoverable/ambiguous strata and appropriate
source-matched ordering anchors before the next fit. Existing aggregate
detection and universal q20 ordering remain legacy diagnostics, not evidence
of catastrophic detection at very low qualities. No existing failed candidate
becomes qualified through this clarification; the new stratified evaluation
and its preregistered acceptance bars are still missing.

## September 8 production contract — registered before new candidate validation

This section is the release contract except where superseded above. The older
measurement history below remains evidence, not an alternate set of acceptance criteria. These
new numerical tolerances are engineering requirements established under the
user's September 8 directive; they are not claimed to be perceptually validated
JNDs or previously agreed consumer tolerances. No candidate is qualified yet.
Freeze the contract and experiment manifests before evaluating new candidates.

| Requirement | Release bar / measurement |
|---|---|
| Human ranking | Preserve G-RANK below: at least incumbent CID22-band performance and no holdout collapse. Also meet or exceed SSIMULACRA2 on the registered SDR human-ranking aggregate and each supported content-class aggregate. Report each corpus/content panel separately; terminal panels are read only for a frozen finalist. HDR uses the registered HDR reference panel. |
| Dial and addressability | Preserve G-DIAL, G-ADDR REGRESSION and CONTRACT, all required codec floors, exact pixel identity at 100, no distorted image above identity, and valid negative tails. No clipping the dial to make targeting pass. |
| Corruption | Preserve the registered D228 bars: zero honest native codec outputs lowered; overall honest lowering <=1%; unique corruption detection >=95%; real-bug detection >=90%; strict below-native-q20 ordering >=99%. Require 100% detection of tested non-inert RGB swaps. Report per-origin/content/codec failures and counts, not only pooled rates. |
| One-shot targeting | On witnessed attainable requests: median absolute score error <=2, p95 <=8; fraction undershooting by more than 8 <=5%. |
| Two-shot targeting | On the same requests: median absolute score error <=1, p95 <=3; fraction undershooting by more than 3 <=5%. |
| Three-shot targeting | On the same requests: median absolute score error <=0.5, p95 <=1, maximum <=3; fraction undershooting by more than 1 <=1%. |
| Target coverage and cost | Evaluate every preregistered image/configuration/request, including failed encodes. Report 100% disposition coverage and the witnessed-attainable fraction separately. Unattainable/uncertain requests are not steering failures or silently removed. Bounds remain hidden from runtime. Actual work must respect each shot budget; native internal reconstructions and map work remain separately counted. |
| Spatial value | Preserve G-STEER M2 >=0.99 and M3 >=0.70. Each JXL/AVIF/JPEG/WebP integration must pass active/neutral and intervention controls. Against a strong scalar controller, require >=0% geometric-mean byte savings at equal quality on every independent judge, plus >=1% savings on at least one judge; no content-class aggregate regression. Use measured overlapping quality intervals, report per-image regressions and source-bootstrap uncertainty. |
| Scalar performance | On AMD Ryzen 9 9950X3D, release build without target-cpu=native, one pinned worker: complete SDR reference+comparison p95 <=50 ms at 1024x1024 and <=200 ms at 2048x2048, <=1.25x frozen D and <=fast-ssim2 p95 on the same inputs. Report cached-reference and uncached paths separately; the absolute bar applies to uncached. |
| Spatial and memory cost | Cached-reference complete score+map p95 <=3x the candidate's uncached scalar p95 at matching geometry. Peak incremental RSS, including caches/scratch/maps, <=128 bytes/pixel +64 MiB per worker; report codec RSS separately. Use >=30 accepted paired rounds, dispersion and competing-process records. Contended or incomplete timing cannot pass. |
| Input and serving correctness | All components execute through the Rust surface with packed-artifact identity. Exact declared-input tree raw/fire parity; pixel/cache composed-score parity under the declared precision contract. Preserve supported SDR/HDR transfer, primaries, luminance, alpha, stride and geometry behavior with their existing reference checks. Explicit unsupported codec/input combinations must not silently convert or masquerade as supported. |

Target bars apply per codec/configuration and per declared SDR/HDR lane; also
report content-class distributions. Calibration uses canonical training
families only. The target request grid and witnessed-feasibility rule are
frozen in each experiment under TARGET_STEERING_PROTOCOL_2026-09-08.md, before
results; use the same requests for all shot budgets and scalar/spatial arms.
HDR quality and steering use their registered HDR judges, never SDR SSIM2 on
uninterpreted HDR samples. Supported-input coverage and HDR timing must be
reported explicitly; an SDR pass cannot qualify the complete product.

Current disposition: corruption FAIL (honest protection/ordering), spatial RD
FAIL for measured policies, full targeting/performance/input qualification
INCOMPLETE, final artifact NOT FROZEN. Existing reports remain linked below and
in SESSION-RESUME.md. The qualification owner must consume evidence for every
row above before declaring this contract satisfied; its historical five-gate
JSON alone is insufficient.

Later September 8 stable SSIM mechanism: the private Rust error-moment
kernel passes the direct f64 reference and ten SIMD configurations. Kernel
means are 9.43/38.03 ms for three 1024²/2048² planes, with 384/768 KiB scratch.
No feature era or served model selects it yet. Full extraction, spatial
binding, refreshed training data and inference/RD qualification remain required;
no release criterion inherits a pass from this kernel result.
[Kernel, accuracy and integration contract](../benchmarks/stable_ssim_kernel_2026-09-08.md).

Later September 8 SSIM diagnosis: 6,812 additional pixel interventions on
15 failed training cells and eight controls isolate SSIM: its oracle
substitution resolves all 15 failures. The direct f64 window reference then
confirms nonlocal production rounding drift on the same f32 XYB pyramids.
This is an extraction-precision defect to repair with explicit feature-era
provenance before more spatial fitting. No arithmetic changes or new model
qualification occur; the current spatial screen remains FAIL.
[Family decomposition and numerical evidence](../benchmarks/nonmax_diagnosis_2026-09-08.md).

Later September 8 finite max queries: complete local refinement coverage and
exact old score/feature/density replay pass on 792 candidate audits. The
25,248 actual pixel interventions improve mean spatial rank to
0.718/0.801/0.910/0.950 at block sizes 8/16/32/64, but only 81/96 M3f cells
and 80/96 M2 cells pass their bars. Saved-data component substitutions locate
all 15 remaining spatial failures in the non-max approximation: oracle max
fixes none; oracle non-max fixes all. This is training diagnosis, with no
native RD, HDR spatial, memory/latency or model qualification.
[Exact max query, failed cells and next correction](../benchmarks/max_attribution_2026-09-08.md).

Later September 8 candidate spatial work: all 36 L8 integrands pass canonical
reconstruction and the 792-pair/map replay, with exact features and scores.
The cheaper F/D candidate still has 36 hard-max IDs in its unsupported union;
all 252 nonidentity maps remain incomplete. A compact non-additive rectangle
max reference passes exhaustive synthetic checks but is not yet Rust-served
or pixel/codec-qualified. No release gate changes or inherits a pass.
[L8 evidence and exact max-query next step](../benchmarks/l8_attribution_2026-09-08.md).

Later September 8 cheaper composition: 66% D plus 34% equally across the three
F_nonneg32 seeds resolves all 2,471 FIT/CAL training consensus pairs and passes
the preliminary cost screen (23.77/84.38 ms, essentially D). All 72 peak terms
remain missing from 252 nonidentity maps. Human/HDR/corruption/spatial/targeting
and full performance qualification remain failed or incomplete; no release
candidate qualifies. [Composition, corrected sample count and next work](../benchmarks/cheap_model_blend_2026-09-08.md).

Later September 8 activity-row storage: complete blend means improve further
to 35.33/133.88 ms, with unchanged features/scores/maps and passing registered
retention controls. Relative-to-D remains 1.49/1.53 on these means; strict
quiet/p95/HDR/memory qualification remains INCOMPLETE. No quality or spatial
gate changes. [Activity repair and controls](../benchmarks/padded_activity_rows_2026-09-08.md).

Later September 8 arithmetic-preserving row storage: blend means become
42.20/141.44 ms at 1024²/2048², D 25.34/89.77 ms, with exact feature/score/map
replay. Absolute observed latency improves substantially; the relative-to-D
bar still fails in the same build. Quiet/p95/HDR/per-worker memory admission
remains INCOMPLETE. [Kernel repair, parity and control limits](../benchmarks/padded_ssim_rows_2026-09-08.md).

September 8 A/D cost: complete blend means are 88.05/194.45 ms at 1024²/2048²,
versus D 76.41/136.04 ms. The 40-round observed extrema miss the 1 MP absolute
and SSIM2 bars and the 4 MP relative-to-D bar. Low dispersion does not repair
strict quiet admission: background activity leaves performance qualification
INCOMPLETE. The blend does not advance as-is; profile common extraction and
the full-pool increment next. [Measured cost and repaired benchmark gate](../benchmarks/model_blend_speed_2026-09-08.md).

September 8 A/D composition: the fitted 59% A / 41% D blend passes the
registered fit/calibration ordering and Rust pixel/cache screen. Its complete
spatial scores match, but all 252 nonidentity maps omit active v1 pooled-feature
terms. Native steering is INCOMPLETE until those integrands are implemented
and validated; no release gate inherits a pass from this scalar result.
[Actual model and coverage record](../benchmarks/model_blend_2026-09-08.md).

September 8 base-model screen: D resolves all 2,471 independent-judge consensus
pairs on twelve canonical training sources; the newer A_plain/H_anchorlad
MLPs show many near-lossless reversals. Generation A and B each reverse three.
All nine pass pixel/cache score parity and identity on this panel. This is
training evidence, not human-ranking or spatial qualification. No candidate
advances under the registered rule. [Results and next model experiment](../benchmarks/model_preferences_2026-09-08.md).

September 8 measurement repair: native JXL delivered decoding now matches
canonical extraction on all 756 regenerated training outputs (exact pixel
hashes and f32-reported scores). Earlier target counts belong to the old
decoder era. This closes the observed decoder inconsistency, with no change
to the failed/incomplete release disposition above. See the
[controlled decoder record](../../jxl-encoder/benchmarks/zensim_decode_contract_2026-09-08.md).

**September 8 correction — user ruling:** steering evaluation must first
establish each image's attained codec range, then compare 1/2/3-shot policies
using codec-specific calibration fitted only on imazen-26 training families.
The [current protocol](TARGET_STEERING_PROTOCOL_2026-09-08.md) defines witnessed
feasibility, separate oracle cost, source-family splits and native diffmap work
accounting. The July median <=2 screen is historical, not a validated product
tolerance. Unwitnessed requests cannot be used to declare steering failure.

**September 7 measurement record:** the existing `zensim-target` controller now
accepts complete candidate `BakeScorer` models. The [360-cell actual-loop
record](../benchmarks/cleanup_target_loop_2026-09-07.md) measures targets, bytes,
passes and independent judges, and misses the legacy three-pass error screen.
That mixed feasible/unwitnessed aggregate does not establish model failure. Its
scalar-q interpolation is diagnostic; it makes no encoder-RDO gain claim.
The codec-native starting-quality/RDO instruments below remain distinct work.
No product gate inherits a pass from the three-seed training reproductions.

**Admission prerequisite — user ruling 2026-09-07:** a new model must execute
and serve entirely in Rust through a zensim surface API. Evaluate through that
API, including multi-head/corruption routing, splines and final composition.
A Python-only prototype or a Rust evaluation-only scorer does not meet this
requirement. Cached features require matching identity and the same zensim
scoring surface, checked against the image API. This prerequisite is separate
from—and cannot be replaced by—the quality gates below. B/C/D are replaceable;
no consumer calibration to their current numeric outputs constrains replacement.

**Why this exists.** Until 2026-07-18, zensim model selection was a two-panel offline exam
(rank + dial). But zensim ships to make *codecs hit targets*, and that ability was never
measured — and could not be: `DiffmapResult::score()` returned the legacy V0_2 score for every
profile (fixed `834b4387`), so no encoder loop ever actually tracked a candidate. With the fix
+ the coherence instrumentation + the codec-in-the-loop probe, every candidate bake now takes
a five-gate exam. **A bake without all five rows is not a ship candidate.** SDR and HDR use
the same gates (HDR swaps the corpora/judges — see §HDR).

| gate | question | instrument | pass bar (SDR) | cost |
|---|---|---|---|---|
| **G-RANK** | ranks like humans? | `bake_verdict` (CID22 + LIVE/CSIQ/PIPAL + nonphoto + KonJND + AIC) | ≥ incumbent on CID22-band, no holdout collapse | ~1 min |
| **G-DIAL** | monotone calibrated dial? | dial panel (quarantined_v2 grid) | G1 p5≤25 ∧ p95≥85; G3 mono ≥0.93 | incl. |
| **G-STEER** | can its diffmap steer? | `diffmap_block_coherence --bake` (M2 ceiling + M3 deployable map; fold per family) | M2 ≥0.99; M3 ≥0.70 | ~3 min |
| **G-RD** | saves real bytes at equal *judged* quality? | probe matrix + independent judge panel (`rd_probe_2026-07-18.sh` + analyze) | ≥0% on ALL judges (no gaming regression), photos | ~30 min |
| **G-TARGET** | codec hits attainable targets fast? | per-image bounds + frozen train-calibrated 1/2/3-shot probes | September 8 production contract above: per-shot error/undershoot bars, complete coverage and explicit cost | incl. + separately reported bound oracle |

Operational notes (learned the hard way — see `benchmarks/rd_probe_results_2026-07-18.md`):

- **G-RD/G-TARGET require a dial spline in the bake bytes.** A raw-scale bake cannot take
  targets (the winner MLP was excluded from zenjpeg targeting until `bake_dial_refit
  add-spline` existed — the generic, MLP-capable spline injector; rank-invariance of the
  spline MUST be verified: SROCC identical pre/post on the full panel).
- **G-RD is judged by OTHER metrics** (ssim2 + butteraugli + a fixed zensim build) — a
  candidate cannot win by grading its own homework. A regression on any independent judge at
  equal claimed quality is the gaming signature (measured: B+Trained-map −0.7% butteraugli).
- **G-STEER's fold is per-family**: signed fold for MLP gradients, abs fold for sign-mixing
  additive solves (`−|s|` through the signed path ≡ abs). The M2 ceiling is 1.0 for both
  additive and piecewise-linear (LeakyReLU) models — the axis that matters is **basic-input
  spatializable mass**, not additivity (`benchmarks/mlp_diffmap_coherence_2026-07-18.md`).
- **Steer-mass pre-screen (free):** `closed_loop.diffmap_basic_fraction` in the metrics
  sidecar. A candidate with low basic-block mass is structurally capped as a steerer BEFORE
  any training investment (B = 0.62 → 0.66 M3 ceiling; **BHdr = 0.43** — worse).
- The July statement that both codecs' tables are legacy-V0_2-seeded is stale.
  Later JXL/JPEG/WebP/AVIF work added several profile-specific tables and heads.
  Pin the actual table, scorer and codec revision; never infer compatibility
  from its filename or an old status paragraph. Refit on train only.

## G-DIAL vs G-ADDR — and the cross-bake selection rule (2026-09-06)

**G-DIAL asks "monotone calibrated dial on the standard grid?"; it does NOT ask "does the
dial reach the floor and ceiling a codec loop needs?"** That second question is G-ADDR
(`bake_verdict`'s dial-addressability gate, owner `zensim-validate/src/dial_addressability.rs`;
full spec `benchmarks/dial_addressability_gate_2026-09-04.md`). **Current tier
definitions, checked 2026-09-07:** REGRESSION is carried by A7r's per-codec
floor representability against the mentor on the same instrument. A1–A6's
score-value pins are report-only by default. CONTRACT is C1–C6: monotonicity,
flat/dead-zone fraction, negative-tail sign, identity in-band and nothing
scoring above a perfect copy. **Per user rule 2026-09-04, dial addressability is a HARD ship gate — "any
model that limits dial range cannot ship" — independent of this scorecard's five gates.** A
bake can pass G-RANK/G-DIAL/G-STEER/G-RD/G-TARGET and still fail G-ADDR's CONTRACT tier (the
shipped SDR dial itself fails two of six contract rows).

**This scorecard exams ONE bake at a time; `freeze_check --select` is the rule that compares
MANY.** Until 2026-09-06, `--select`'s PRIMARY (profile floor count) and TIE-BREAK
(`selection_composite`) were completely blind to `dial.addressability` — measured on the
best-of-all wave picking a CONTROL arm at G-ADDR contract 4/6 over arms at 6/6, because
neither key can see a contract failure. **Fixed at the owner**: a candidate (or seed group)
that MEASURES a G-ADDR CONTRACT-tier fail is now an absolute selectability veto in
`--select`, and `A7r`'s per-codec floor-representability folds into `--floor-basis all`'s
floor count. `--floor-basis legacy` reproduces the pre-fix rule byte-for-byte, audit only.
Full record + before/after re-runs: `benchmarks/select_gaddr_prefilter_2026-09-06.md`.

**Qualification clarification, 2026-09-07:** a non-vetoing research selection is
insufficient for shipping. `dial_addressability::Verdict::shippable()` requires
**both REGRESSION and CONTRACT to PASS**; missing probes and failed per-codec
floors do not qualify. The five product gates above are required as well.
`freeze_check --select` enforces measured contract failures but can still choose
a recipe with INCOMPLETE coverage and failed A7r floors. The later
[`board_ladder_ruler_2026-09-06.md`](../benchmarks/board_ladder_ruler_2026-09-06.md)
§5–6 demonstrates exactly that on its selected fast-class recipe. Report
research selection and product qualification separately; the absence of a
NOT-SHIPPABLE badge is not a qualification certificate.

## Tuning with the scorecard (not just picking)

G-RD/G-TARGET are objectives, not only gates: recipe/blend iterations can be selected by
bytes-saved-at-equal-judged-quality + residual, with G-RANK as the guard — e.g. the
screen-content column (every zensim driver negative on screens in the 2026-07-18 probe) is a
*measurable* retrain target: add screen/nonphoto mass, re-probe, watch the column. Steering
quality is also trainable: a mean-pooled-only basic-feature variant makes the M3 fold exact
by construction.

## HDR variant of the exam

Same five gates with: G-RANK → UPIQ/HDR panels (`upiq_panel.py` guard per the BHdr ship
policy); G-DIAL → the BHdr dial grid; G-STEER → PU-linear pairs through the coherence tool
(extension pending); G-RD/G-TARGET → jxl HDR ladder (intensity_target path) judged by
`zenmetrics --hdr` (cvvdp/HDR judges). The steer-mass pre-screen applies immediately:
**screen every BHdr candidate on `diffmap_basic_fraction` before training** — the 2026-07-18
audit measured shipped BHdr at 0.43 (57% of its steer mass unspatializable).

**HDR steer-mass landscape (measured 2026-07-18, 63 bakes across the linear-probe HDR
families — family medians of `diffmap_basic_fraction`):**

| family | med steer mass | n | note |
|---|--|--|---|
| hdrbroadplh1 (shaped, lasso) | **0.963** | 1 | most steerable HDR bake measured |
| hdriwmix (iwssim-teacher mixes) | 0.762 | 7 | steerable family |
| canonhdr40 / canonhdr15 / canonkjhdr15 | 0.58–0.65 | 18 | canonhdr15-bvls = the KonJND-HDR record holder (0.6696) |
| **bhdr_linear_shaped_cvvdpmix (SHIPPED)** | **0.435** | 1 | 57% unspatializable |
| hdriw | 0.359 | 7 | |
| hdrmix (shaped/anchored lineage) | 0.161 | 16 | |
| hdr / bhdr_anchored2 / hdrcodc | 0.01–0.07 | 6 | effectively unsteerable |

Implication: the shipped BHdr's whole shaped/anchored lineage is a steering dead-end; if the
HDR closed loop matters, the next BHdr campaign should start from (or constrain toward) the
hdrbroadplh1/hdriwmix/canonhdr families and hold the UPIQ guard — do NOT train more hdrmix-
shaped variants and hope the map follows. Rank-vs-steer for these families must be settled by
the HDR G-RD leg, not another proxy.

## Provenance

Scorecard rows live in each bake's `.metrics.json` sidecar (+ probe TSVs under
`/mnt/v/output/zensim/rd-target-eval-*/`); the dashboards read sidecars. Instruments:
`zensim/examples/diffmap_block_coherence.rs` (--bake), `scripts/v_next/rd_probe_2026-07-18.sh`
(+ `rd_probe_analyze_2026-07-18.py`), `bake_dial_refit add-spline`, `emit_bake_metrics.py`.
