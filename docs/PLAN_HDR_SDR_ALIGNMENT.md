# PLAN_HDR_SDR_ALIGNMENT — one dial across SDR, HDR, and tone-mapping (drafted 2026-07-14)

## October 27 continuation and precision accounting

No stable candidate has been adopted from the new AIC comparison. Frontend
implementation remains queued behind that decision; research and exact
qualification planning may proceed. The [current run index](../SESSION-RESUME.md#current-work--october-27-2026)
records dependencies, source owners and the still-unfinished confidence arm.
The goal is better perceptual quality than PU21 while maintaining speed, not
merely a cheaper approximation to the same transform.

Keep three distinct precision questions separate:

- **Inputs:** preserve native u16 PQ/HLG codes or f32 absolute-light input with
  declared primaries/display metadata. Never narrow to RGB8 for convenience.
- **Feature arithmetic:** test near-black PU floors, highlight/output clamps,
  color transforms, near-identity cancellation and accumulation against the
  existing f64 oracle. Rev7's stable forms are distinct from older-revision
  cancellation findings; changing arithmetic requires a new feature identity.
- **Deployment:** the queued AIC full panels compare final-f32 models with
  packed, TRAIN-calibrated serving models. That difference combines packing
  and output calibration; it cannot isolate quantization loss. Native pixel
  frontend precision and runtime costs require separate measurements.

Do not promote BHdr's historical pooled rank result into an isolated utility
claim for its frontend or highlight features. Preserve the source/exposure and
negative ablation evidence below. Preregister matched refits for PU21, ICtCp
and a compact monotone luminance/chroma alternative after stable adoption;
freeze source roles, display parameters, seeds, budgets and controls before
new outcomes. Measure complete scoring, reference preparation and steering
cost across the production geometries on permitted quiet hardware. No speed
or no-penalty claim is established for a new frontend yet.

## October 27: draft frontend registration (not frozen)

A draft for the comparison above, written before any HDR outcome of the new
recipes. It freezes after stable adoption names the architecture and the two
owner decisions below are made; it changes no running study.

**What exists.** Corrected HDR TRAIN has 7,425 native-PQ pairs (495 reference
variants, 33 source families) with HDR-VDP-3 and CVVDP teacher labels
(HDRTEACH). Teacher-only HDR training at Rev5 (E27) did not pass: hp4 kept
SDR as-good and gained within-reference agreement with both teachers
(+0.0012), but pooled HDR-VDP-3 rose (+0.123) while pooled CVVDP fell
(−0.089). Human UPIQ-HDR training (V40/E31, uh4, TRAIN weight 4.344) kept
SDR as-good; on its 50 held-out development observations (four references)
pooled SROCC rose from 0.79–0.85 to 0.85–0.94 across 40 cells. These are
Rev5 results with the PU21 frontend; no frontend comparison exists.

**Arms.** One adopted Rev7 recipe, three frontends, all else matched:
F0 canonical PU21 (reproduction control), F1 ICtCp with explicit intensity
and chroma scaling, F2 one compact monotone luminance curve with separate
chroma handling, frozen before any comparison outcome. Each frontend needs
its own feature identity, Rev7 HDR extraction of the TRAIN pairs and UPIQ-380,
and native serving; SDR inputs keep the unchanged SDR path. Same SDR legs,
HDR legs (teacher pairs at the E27 convention, UPIQ-380 at E31's weight),
seeds, budgets, sampler, epoch rule and calibration for every arm.

**Assessment, all registered before outcomes.** SDR E21 out-of-sample panels
under the as-good rule; HDR TRAIN development (UPIQ development references,
teacher-labelled hdr_v3mix VAL) as development evidence only; one human HDR
holdout (decision 1); R1 sub-domain identity (p95 |Δ| ≤ 2, rank ≥ 0.99); the
numerical controls listed above against independent f64 references; and
complete native scoring, reference preparation and steering cost at the
production geometries (decision 2). "Better than PU21" means F1 or F2
improves the human HDR holdout panels over F0 with whole-source uncertainty
reported, with no SDR regression and no cost regression on that hardware.
Full panels decide; no single SROCC threshold.

**Numerical controls already in place at Rev7** (`zensim/tests/native_hdr_color.rs`,
2026-10-27): native PQ/HLG codes in sRGB, Display-P3 and BT.2020 match an
independent f64 display-light decode within 1e-3 relative; one-code u16
steps survive the supported entry; PQ changes confined above the declared
1000-nit peak clip to identity while changes below it stay visible; padded
strided rows give bit-identical features to packed rows. Each has a failing
negative control (wrong reference lift; 4000-nit peak; stride misread).
Near-black lift, transfer joins, caches and maps still need Rev7 controls.

**Owner decisions needed.** (1) A human HDR holdout. HDR-VDC and AVT keep
only Rev2/944 tables and need fresh sign-off to re-extract through the frozen
drivers; alternatively admit another organized library dataset (for example
the two Rousselot HDR compression sets, licence to confirm) as eval-only.
UPIQ-380 is TRAIN since October 7 and cannot serve. (2) Timing hardware:
the scalar-performance row names the 9950X3D, where benchmarks are currently
not permitted.

## October 26, 2026: evidence and speed requirement

The controlling [production queue](PRODUCTION_PRIORITIES_2026-09-15.md) places
implementation after adoption of a stable candidate. The owner now explicitly
requires maintaining speed and wants something better than PU21. This section
records evidence and proposed comparisons, not a launched experiment or a
change to the frozen AIC continuation.

The earlier BHdr work is useful evidence. Its historical UPIQ pooled SROCC was
0.7536 versus CHdr's 0.6664 on the same pairs, each through its own feature
regime. That comparison changed the model, training and feature bank together;
it does not identify which HDR feature family caused the difference. The
earlier BHdr promotion comparison also selected the best of seven candidates
on that panel: the subsequent audit reported selection-adjusted p=0.221 and
withdrew the original significance claim. Preserve that exposure history.
Sources: [BHdr lineage and audit](../benchmarks/bhdr_improvement_split_lineage_2026-07-12.md),
[BHdr/CHdr comparison](../benchmarks/balance_campaign_2026-08-28.md).

Feature utility has mixed historical evidence. Removing BANDVIS hurt the
tested 944 model, but eight highlight-bin slots were zero on those SDR rows;
that test does not validate their HDR utility. Removing the tested CSFW
weighted family helped. Its subsequent HDR DMEAN transfer probe also lost
rank. Reuse the implementations and preserve these negative controls instead
of treating each old family as a demonstrated improvement. Sources:
[BANDVIS ablation](../benchmarks/bandvis_loo_944_2026-07-28.md),
[CSFW ablation](../benchmarks/csfw_g6_loo_2026-07-29.md),
[HDR DMEAN probe](../benchmarks/hdr_dmean_commensurability_2026-07-29.md).

Current Rev7 serves basic/peaks/v2 and supported global-affine features;
`feature_plan.rs` does not admit the older append/append2/CSFW regimes.
Restoring a family or changing the frontend therefore needs an explicit
feature identity, matching extraction, fitting, serving and maps. Existing
native HDR conversion preserves u16 PQ/HLG or f32 nits and declared primaries.
The later opsin clamp and PU21's 0.005-nit floor remain separate boundaries
to test. Rev7 already implements stable luminance difference forms and
bounded f32 accumulation into f64; historical cancellation results for older
revisions are not evidence of the same defect in Rev7.

The smallest proposed matched comparison retains the adopted architecture,
input pattern, TRAIN populations, source splits, seeds, loss, sampler, epoch
rule and calibration procedure, while refitting each frontend's scaler/head:

- Current canonical PU21 as the reproduction control.
- ICtCp with explicit intensity/chroma scaling as a color-aware alternative.
- One compact monotone luminance curve with separate chroma handling, frozen
  from independently admitted sensitivity/TRAIN evidence before comparison.

The compact curve is a hypothesis directed at both quality and cost. ICtCp
is not assumed cheaper, and PQ alone is not an established improvement over
PU21. Existing PQ code-decoding lookup tables cannot be substituted for an
LMS encoding transform across a color matrix. A compact bandwise adaptation
model using existing scales is a later candidate; the failed CSFW summaries
do not validate it. Primary basis: [PU21](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2021_PU21.pdf),
[ICtCp](https://professional.dolby.com/siteassets/pdfs/ictcp_dolbywhitepaper_v071.pdf),
[castleCSF](https://www.cl.cam.ac.uk/research/rainbow/projects/castleCSF/).

Before fitting, freeze numerical controls for the same displayed light in
linear/PQ/HLG and sRGB/P3/BT.2020 descriptions, native 10/12/16-bit steps,
near-black and transfer joins, highlights, negative common-basis components,
strides, tails, caches and attribution maps. Use independent f64 references
and preserve existing tolerances. Routing equality is not SDR/HDR dial
continuity: test equal displayed-light pairs and display-white/peak/ambient
progressions, retaining the original human linking and seamlessness gates.

Measure complete native scoring, reference preparation, cached scoring and
steering costs, with the existing quiet-machine paired methodology and memory
evidence. Historical PU microbenchmarks and route repairs do not establish a
replacement's current cost. No replacement has yet shown both a quality gain
and acceptable end-to-end cost for Rev7/Mlic. Keep existing speed gates, and
register any additional adoption criterion before observing results.

User directive (2026-07-14): examine all techniques for an optimal HDR metric +
historical achievements, and design SDR↔HDR alignment such that (a) similar
perceptual differences in SDR and HDR content get similar scores, and (b) a
tone-mapped rendition of an HDR pair scores near the HDR pair — "close enough
to make human sense."

Companion to `PLAN_HDR.md` (training + dial continuity) and `PLAN_BEAT_A.md`
(Bet2 human triplets). Evidence base: `benchmarks/
bhdr_improvement_split_lineage_2026-07-12.md` §1–§8.15.

## 1. The alignment problem, formalized

zensim is a user-facing dial: "give me 85" must mean the same visual quality
whether the asset is SDR, HDR, or an SDR tone-map of an HDR master. Three
requirements, in decreasing strictness:

- **R1 — sub-domain identity.** SDR-range content scored through the HDR path
  must equal the SDR path. (Already a pre-registered gate:
  G-HDR-SDR-CONSISTENCY in PLAN_HDR — p95 |Δ| ≤ 2 pt, rank ≥ 0.99. The SDR
  case is the sub-domain limit; no luminance seam.)
- **R2 — cross-domain equal-difference.** If humans judge an SDR pair and an
  HDR pair equally far from their references (in JND), both get ≈ the same
  dial value.
- **R3 — tone-map near-consistency.** `M_sdr(TM(R), TM(D)) ≈ M_hdr(R, D)`,
  with divergence permitted exactly where tone mapping genuinely changes
  visibility (highlight quantization crushed by the curve; banding revealed
  by range compression). NOT strict invariance — "human sense" means the
  deviation should be explainable, not zero.

**Exchange currency: JND/JOD.** The only psychophysically defensible
cross-domain unit. cvvdp emits JOD for any display model (SDR or HDR
photometry); the AIC-3 triplet method reconstructs JND scales; UPIQ's whole
contribution is re-aligning four datasets onto one JOD axis. The dial should
be a monotone, domain-blind function of JND-from-reference.

## 2. Technique inventory (what we hold today)

**Luminance encodings / front-ends**
- PU21 u8-shell (v1): PU-encode absolute nits → rescale to u8 → SDR feature
  extractor. Display-relative; auto-normalizes brightness (loses absolute-
  luminance masking, gains scale comparability).
- PU-linear (v3, `compute_pu_linear_extended_features`): absolute-nits PU
  values straight into the 372-D extended-feature front-end. The production
  BHdr regime. ⚠ Regime purity is load-bearing — §8.13's confound came from
  mixing regimes in one gram; tooling now enforces
  (`--features-name`, upiq_panel warning, GROUPS comments).
- PQ code-value domain for distortion GENERATION (kadis-hdr,
  normalize=truncate never mapmm) — synthesis-side, not metric-side.
- `--hdr-transfer pu-rescale` scoring shells per metric: cvvdp + butteraugli
  faithful linear planes; ssim2/iwssim/zensim PU21 u8 shells; dssim
  HdrUnsupported by design.

**Teachers (measured)**
- cvvdp-mix `0.5·ssim2norm + 0.5·clamp((JOD−6)/4)` — the shipped BHdr lever
  (§6). cvvdp-scalar-ONLY: falsified twice (V41 + 2026-05-27) — emulating
  cvvdp's output ≠ having its mechanism.
- iwssim-teacher (iw_logn spread): mechanism diagnostics strong, selection
  transfer falsified (§8.5–8.6). Teacher-ceiling measured (§8.3).
- ssim2: reliable q50–90 band, saturates HQ (measured: cvvdp-agree 0.82→0.48
  by zone).

**Calibration & packing**
- Monotone PCHIP output spline (dial), fit on the PROJECTED+QUANTIZED net;
  `bake_dial_refit` subcommands incl. `shared-anchor` (whole-spline refit to
  a shared anchor — the co-calibration primitive R2 needs).
- winsor_p99 feature guards — measured to matter: they tame the JXL
  near-lossless OOD tails (~24× L2) and are part of why the baked BHdr
  outscores its raw fit head on UPIQ.
- f16 QAT-native packing (SROCC-neutral by construction).

**Model families (measured)**
- Linear 372→1 shaped: the champion family — deterministic, no collapse
  axis (44/44 byte-identical refits), 1–12 KB. B (SDR) + BHdr both live here.
- MLPs: higher ceiling on some axes, collapse modes (2/9 seeds), KonJND/f16
  trade; per-sample-α, routers/ensembles for G5: falsified.

**Instruments**
- UPIQ **full** set is local (verified 2026-07-14): 4,159 rows on ONE
  JOD-aligned scale — tid2013 3,000 + live 779 (SDR, `is_hdr=0`) +
  narwaria 140 + korshunov 240 (HDR). We have only ever extracted the HDR
  380. ⚠ tid2013 overlaps training (integrity-guard grade); **live(779) is
  clean** of our corpora. HDR-380 remains confirmation-only (burn ledger
  ~26 looks — ~22 + the 4 registered Appendix Q phase-1 candidate looks,
  2026-08-05; see `sota944_campaign_2026-08-03.md` Q.R4).
- Dial grids (densified), bake_verdict rank panel, upiq_panel (--strata,
  --compare, regime-warned), KonJND PJND anchor.
- Corpora: jxl-HDR (PU-linear), kadis-hdr (both regimes now), UltraHDR/HEIC
  gain-map ingest (`ultrahdr` crate: gain-map math + tone mapping — the TM
  operator source for R3 bridges).

## 3. Historical achievements + falsification ledger (the short version)

Achievements:
- BHdr (linear, cvvdp-mix, PU-linear): UPIQ pooled **0.7536** / narwaria
  **0.7834** / korshunov **0.9175** — above ssim2-HDR (0.704) and Profile A
  (0.694), just under raw cvvdp (0.758), below iwssim-HDR (0.808).
- B↔BHdr routing identity test passes byte-for-byte on descriptor-flagged
  SDR content (the R1 seed).
- HDR corpus estate: jxl-HDR + kadis-hdr (11,400 cells × 5 metrics + both
  feature regimes), fleet + jobexec ScoreFile-HDR machinery.
- Pre-registration discipline that caught its own confound (§8.11→§8.15).

Falsified (do NOT retry without new evidence): cvvdp-scalar target ×2;
CSF-feature engineering for the AIC-3 gap; synthetic corpus breadth for UPIQ
transfer (clean, §8.15); iw-teacher selection; G5 single-MLP weighting +
regime-routed ensembles; corpus-refit of the cid head.

## 4. Alignment mechanisms (design)

**A. Sub-domain identity (R1) — structural, mostly done.** Keep the single
PU-linear substrate; BHdr restricted to ≤ ~100–200 nits IS the SDR case.
Strengthen from "gate" to "training constraint": add a distillation term on
SDR corpora — penalize |BHdr(pu_linear(sdr)) − B(sdr)| — so identity holds
by construction, not luck. Cheap: no new labels, SDR corpora are huge.

**B. Cross-domain equal-difference (R2) — calibrate on one JOD axis.**
1. Extract UPIQ **SDR-half** features (both regimes; live-779 first —
   clean). Instrument: regress dial vs UPIQ-JOD across `is_hdr` ∈ {0,1};
   ALIGNED ⇔ one monotone mapping fits both halves with no domain dummy
   (report the domain-offset coefficient + per-half residuals).
2. Joint spline co-calibration: build ONE multiband anchor whose
   `target_score` lives on a JOD-derived 0–100 scale spanning SDR + HDR
   cells; refit B's and BHdr's splines against it
   (`bake_dial_refit shared-anchor` — rank-invariant, kills systematic
   scale seams). Near-threshold pinning: "visually lossless" ≈ dial 63 in
   BOTH domains (KonJND anchors SDR; cvvdp-JOD ≥ ~9.5 proxies HDR until
   human PJND exists).

**C. Tone-map bridge (R3) — teacher-explained divergence.**
1. Build the bridge corpus: for each HDR pair (R,D) in jxl-HDR/kadis-hdr,
   produce (TM(R), TM(D)) with 2–3 operators (ultrahdr gain-map path,
   BT.2446-style, simple photographic). Score: dial_hdr(R,D),
   dial_sdr(TM pair), cvvdp_hdr (HDR display model), cvvdp_sdr (SDR display
   model on the TM pair).
2. The teacher's Δ_JOD = cvvdp_hdr − cvvdp_sdr is the VISIBILITY-CHANGE
   ground truth (how much the TM actually hid/revealed). Alignment metric:
   residual = (dial_hdr − dial_sdr) − s·Δ_JOD after a single global slope s;
   gate on p95 |residual| (proposal: ≤ 5 dial pts; tighten after measuring).
   This formalizes "close enough to make human sense": deviations are
   allowed exactly where the visibility teacher says they should exist.
3. MEASURE FIRST (instrument before lever): run the bridge on current
   B/BHdr before training anything — the misalignment magnitude and its
   structure (offset? content-dependent? TM-dependent?) decides which lever
   (spline co-calibration alone vs consistency-trained head).
4. If a lever is needed: add the bridge consistency term to the next BHdr
   registration — loss += λ·|(M_hdr − M_sdr∘TM) − s·Δ_JOD_teacher|. This is
   supervision we can synthesize at scale TODAY (no human labels needed for
   the CONSISTENCY signal; humans still gate via UPIQ/own-triplets).

**D. Display conditioning (forward-looking).** A peak-nits/display scalar
input (A_Phone precedent: narrow dial was CORRECT cvvdp-emulation for that
display). One head serving 600-nit phone vs 4000-nit monitor is the honest
generalization of "HDR metric"; defer until R1–R3 land.

## 5. Ordered program (each step lands an artifact; registrations before fits)

1. **UPIQ-full extraction** (SDR halves, both regimes) + cross-domain
   instrument report — the R2 baseline number for current B/BHdr.
   Artifact: `benchmarks/upiq_crossdomain_baseline_<date>.md`. (tid2013 leg
   labeled integrity-guard; live leg is the headline.)
2. **TM-bridge corpus + baseline measurement** (no training): ultrahdr TM +
   1–2 simple operators over a stratified subset (e.g. 1–2k pairs), cvvdp
   dual-display scoring via existing fleet machinery. Artifact: §-doc with
   the misalignment decomposition. Gates proposed here get REGISTERED
   before any lever is fit.
3. **Joint spline co-calibration** (cheap lever, rank-invariant) if step
   1–2 show a systematic seam. Artifact: co-calibrated bake pair + before/
   after instrument numbers.
4. **Next BHdr head registration** adds: SDR-distillation term (R1) +
   bridge-consistency term (R3) with pre-registered gates on the
   instruments from 1–2 (+ the standing UPIQ-HDR confirmation discipline).
5. **Human currency** (parallel track): Bet2 triplet ingest against
   BTC-PTC-24 format (serves AIC-HDR2025 on release day + our own HDR
   triplet study on imazen-26 captures — the only path to owned human-HDR
   supervision; §8.15 says synthetic breadth cannot substitute).

## 6. Honest caveats

- UPIQ-SDR's tid2013 leg overlaps training → guard-grade only; live-779 is
  the clean SDR leg (n modest; CI-report everything).
- UPIQ-HDR burn ledger stands — steps 1–3 don't touch it; step 4's
  confirmation uses it ONCE per registration, as always.
- TM choice injects its own prior; that's why ≥2 operators and a
  teacher-explained-divergence formulation rather than raw invariance.
- cvvdp as the Δ-visibility teacher inherits cvvdp's own HDR biases
  (JOD 6..10 band compression at HQ); acceptable for a CONSISTENCY signal,
  not as an absolute target (that lesson is already paid for — twice).
- LIVE image files may not be on disk yet (scores are; check
  `upiq_extracted` coverage before step 1 and pull LIVE release if needed).

### October 27: native input baseline at Rev7

`just mlic-native-input-tests` preserves the five legacy native-HDR tests and
adds a separate Rev7 child-process check of the supported 720-feature HDR
entry. Running the legacy append-family tests wholesale with
`ZENSIM_FORMULA_REV=7` fails before color arithmetic because that feature family
is deliberately unsupported. The rejection and old test expectations remain
unchanged; this failed invocation is retained in `native-hdr-rev7.log`.

The new check uses a 97×65 image with native u16 codes and a one-code red-channel
distortion. For PQ and HLG, each in sRGB, Display-P3 and BT.2020, the native-u16
features match the equivalent normalized-f32 code features exactly, remain
finite, and differ from identity. All six cases pass. This proves the tested
input-container and supported-entry behavior; it does not prove every possible
code step survives display clamps, establish a new f64 error bound, qualify
Mlic's full model/input composition, or show HDR perceptual quality or speed.
The original five tests also pass, retaining their independent f64 color/display
reference and prepared-map checks under their original arithmetic.

Evidence: shared output `zensim/mlic-precision-20261027-run01/`, including
`native-hdr-build-initial.log`, `native-hdr-rev7.log` and
`native-hdr-supported-rev7-run02.log`. The first build/default-five-test run:
run-heavy rc=0, 147s, peak RSS 1.92 GiB, minimum available 21,921 MiB, peak load
3.80. The final six-test run including explicit Rev7 child: rc=0, 9s, peak RSS
0.59 GiB, minimum available 23,443 MiB, peak load 1.41. These are validation
resource receipts, not performance qualification. No frontend arithmetic,
feature identity, trained artifact or running experiment was changed.
