# TOP MODELS COOKBOOK — the science + exact reproduction paths (2026-07-18)

> **Current work, October 27:** the active Mlic/Neff/Vote/Order AIC comparison
> uses complete panels without strict SROCC rejection. Confidence still needs
> admission, linking and native fitting. No new serving model is adopted.
> Exact run identities, completed/queued work and recovery commands live in
> [SESSION-RESUME](../SESSION-RESUME.md#current-work--october-27-2026), with
> registrations/results in [the byconstruction record](../benchmarks/byconstruction_2026-10-22.md).
> The roster below remains dated evidence. In particular, use the latest
> [split rulings](DATA_SPLITS.md), not an old roster's corpus defaults.

> **September 14 reading correction:** the statements below describe their
> recorded experiments, not current universal limits. Non-additive max features
> now have separate finite rectangle queries; opt-in L2/L4/L8 moment corrections
> also execute through the complete Rust candidate API. They do not turn maxima
> into additive density. See [finite-moment evidence](../benchmarks/finite_moments_2026-09-14.md).
> The historical observation “M2 = 1.0 for every architecture measured” does not
> guarantee finite-edit linearity: the current TRAIN [basic/peak comparison](../benchmarks/product_peaks_2026-09-14.md)
> includes H128 failures. A piecewise-linear head can cross activation boundaries.
> Current H32/H128 training and admission use the [wave playbook](WAVE_PLAYBOOK.md),
> not the old recipe defaults below. The [September 13 split ruling](DATA_SPLITS.md#september-13-user-ruling-train--eval-only-never-touch-test)
> prohibits historical test/terminal reads even for a frozen finalist. None of
> these historical roster entries automatically qualifies under the current
> [production scorecard](MODEL_SELECTION_SCORECARD.md).

> **Current-reading note (2026-09-07):** this is a dated recipe/evidence
> archive with later addenda. Current integration and bake identities live in
> [`CODEC_TARGET_METRIC.md`](CODEC_TARGET_METRIC.md); current unresolved choices
> are in [`OPEN_DECISIONS_2026-09-06.md`](OPEN_DECISIONS_2026-09-06.md).
> In particular, §1b's September 5 restriction to 372-wide serving was closed
> by the feature-plan work. C/CHdr's remaining train/serve toggle mismatch is a
> separate defect. Evaluate by declared feature IDs and era, not by choosing
> `--regime 944` for every model. A board selection is not ship qualification.

> **★ CANDIDATES-OF-RECORD (frozen by user 2026-08-28; shipped default remains B):**
> SDR **`W10L9PH_s4004_packed`** (61ebc456…; SPH1 recipe = W10L9 purity views +
> family-clean tbig HF leg; the balance campaign's sole full-eligibility pass,
> in-loop best) — recipe/evidence: `benchmarks/sdr_pure_retrain_wave_2026-08-28.md`
> + `balance_campaign_2026-08-28.md`. HDR **`HDR944_L1T1_s4005_hfpack`**
> (0a437d99…) — `benchmarks/hdr944_retrain_wave_2026-08-28.md`.


> **ERA BANNER (added 2026-08-04).** The roster, champions, and numbers below are the
> **372-era record** (v1 feature space, 372col corpora). The current era is **944**
> (folded+append+append2): its record — frozen bar, arm results, corrections, the
> seed-ensemble waves, and the stabilized ceiling (`C_co3a_s1301`) — lives in
> **`benchmarks/sota944_campaign_2026-08-03.md`**. Its original runs used
> `bake_verdict --regime 944`; current evaluation resolves the bake's actual
> feature identity and matching root (see `SESSION-RESUME.md`). **Cross-era numbers
> are NOT directly comparable**: evals are era-tagged (different feature widths, corpora
> re-extractions, and eval slices; only CID22 val — same 4,292 pairs every era — bridges,
> per the campaign doc's era-bridge section). The science, pitfalls, and reproduction
> paths below remain valid FOR THEIR ERA and most transfer; the model *rankings* do not
> automatically.

**Audience: a future agent who must understand why the top models look the way they do and
build better ones without re-learning any of it the hard way.** Read this, then
`docs/MODEL_SELECTION_SCORECARD.md` (the five-gate exam), then the per-claim benchmark docs
linked inline. Everything here is measured; nothing is aspiration.

## 0. The product frame (why every choice below exists)

zensim is a **codec-targeting quality dial**: users pick a target score, the codec converges
on it (`Quality::Zq*` in zenjpeg, the zensim loop in jxl-encoder), and the per-pixel
**diffmap** tells the encoder WHERE to spend bits (closed loop) — with a one-shot predictor
(`ZqPicker`) as the no-measurement floor. So a model is only as good as its worst of: rank
(agrees with humans), dial (monotone, calibrated, full-range), steering (its diffmap points
where its scalar rewards bits), RD (a codec steered by it saves real bytes under *independent*
judges), and targeting cost (few passes). That is the five-gate scorecard.

## 1. The validated science (each line carries its evidence doc)

**Features.** Only the basic block **f0..155** (4 scales × 3 XYB channels × 13 mean-pooled
signals) is *spatializable* — expressible by a per-pixel map. Peak/max/masked/IW poolings
(f156..371) are non-additive across blocks: NO per-pixel diffmap can express them, which is
why B (38% non-basic mass) has a hard 0.66 steering ceiling and why the top models are
basic-156-input (`benchmarks/mlp_diffmap_coherence_2026-07-18.md`). Known feature defects:
`hf_gain` is an unbounded ratio (→1e7 on OOD; tamed per-bake by surgical winsor — bounds in
§3); the IW/masked block has a `1/n`-vs-`Σw` normalization divergence + unbounded edge energy
(the o_9292 5.8M incident) — a v2 "perfectable features" regime is in design (feature-science
audit, 2026-07-18). IW-SSIM's *signal* stays in the system as a judge and TEACHER target
(cvvdp/iwssim mixes), not as input features. Literature corpus: `~/work/zen/zenpapers`
(+ PDFs at `/mnt/v/input/papers/`) — search it before designing features.

**Training.** `zensim_mlp_train` **has no linear mode** — `--n-hidden-layers 0` is ignored;
every bake it emits is a 128-hidden LeakyReLU MLP. "Linear/additive" claims about its bakes
were a systematic mislabel, since corrected (`benchmarks/additive_vs_mlp_correction_2026-07-18.md`).
Genuinely-additive bakes come only from the linear-projection solver
(`scripts/v_next/linear_projections_2026-07-03.py` + `additive_basic156_probe.py`). Loss:
**`:both` (RankNet+MSE) is the dial+rank recipe** — RankNet-only ranks well but the dial
jitters (0.47); MSE-only "smoothness" is COLLAPSE (dial 0.998 but CID22 0.085); `:both` gives
both (`benchmarks/final_metric_experiments_2026-07-18.md`). CID22 human MOS is
validation-only, forever. Data-split law: `docs/DATA_SPLITS.md`; corpus history + poison
ledger: `docs/DATASET_HISTORY.md`.

**Dial.** A monotone PCHIP output spline maps raw → [0,100]; it is RANK-INVARIANT, so fit it
post-hoc: `bake_dial_refit add-spline` works on ANY spline-less bake incl. MLPs (forwards via
`predict_transformed`, re-emits layers verbatim; validated: SROCC identical on 10 corpora,
raw[−3.2,24.2]→[14.6,95.3]). Never fit a spline on top of a spline.

**Steering (diffmap).** The gradient-linearization ceiling **M2 = 1.0 for every architecture
measured** — LeakyReLU MLPs are piecewise-linear, so their gradient is locally exact; the old
"non-additive scalars cap at ~0.87" story is dead. The deployable map is
`DiffmapWeighting::ModelSensitivity(s_k)` (custom-profiles): **signed fold for MLP gradients**
(0.759), **abs fold for additive solves** (0.849; `−|s|` through signed ≡ abs) — additive
solvers sign-mix within pooling triples, MLP gradients don't. The shipped `Trained` map +
B scalar is the WORST measured pairing (0.243).

**Closed loops (two bugs every older conclusion predates).** (1) `DiffmapResult::score()`
returned the legacy V0_2 score for EVERY profile until fix `834b4387` — no encoder loop
tracked the real metric before 2026-07-18. (2) zenjpeg's Zq correction passes were INERT (AQ
strength only nudges zero-bias rounding; byte-identical output) until the worktree's global
q-correction. Both codecs' distance/starting-q tables are still legacy-seeded — re-seed before
trusting convergence-pass counts (`docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md` §phase-2).

**RD value (the point of it all).** On photos, model-coherent maps save **+2–4% bytes at
equal independently-judged quality** vs the no-diffmap baseline, beating the native
butteraugli loop on 2/3 judges; SSE (codec default) is ANTI-correlated with the metric on
texture. Screens are the open gap: ~half model-side (fixed by data mass, §3), remainder
loop-side (photo-seeded tables). zensim-loop cost ≈ butteraugli-loop cost (~3× a plain
encode); measured loop targeting = residual 0.84 in 4 passes vs one-shot 4.33 in 1 pass
(`benchmarks/rd_probe_results_2026-07-18.md`).

### 1b. SERVABILITY and the FLOOR — two things that decide a fast-class model before rank does (2026-09-05)

Added because a campaign spent its first hours discovering both, and neither is
findable from this file's other sections.

**SERVABILITY is a LAYOUT question, and only two fast-class sets pass it.**
`Zensim::compute` emits a **372-layout** vector with `free_extras: Off`
(`feature_v2.rs:7532`, `fold_engine.rs:158`; kernel lane `8817f379`), so **no
`V1FreeExtras` slot is reachable from the production path today** and
`wide_bake_v2_read` is dead code. MEASURED end-to-end
(`zensim/examples/serve_custom_bake.rs`, which loads any ZNPR through
`ZensimProfile::Custom` and calls the real `compute`):

| bake | declared | production path |
|---|---|---|
| shipped Profile D (156) | `caller=372` | **SERVED** |
| a 372 bake that READS PEAKS (`Dpeaks372`) | `caller=372` | **SERVED** |
| any 944-declared bake (`A3b`/`A4b` class, `Fpeaks_id100negrich`) | `caller=944` | **REFUSED** — `ModelForwardFailed` |

So **train a fast-class candidate at the v1-372 layout** (`f0..227` fits inside
it) unless you intend to build the 944-layout scoring path. A 265- or 289-wide
bake trains fine and cannot be served.

**The FLOOR (`A7r`) is the binding ship clause, and it is a WEIGHTS property.**
Under the 2026-09-05 ruling `A7r` — per-codec floor representability, graded
`--floor-rule resolvable` against the mentor's own fraction — is the *only*
regression row left by default. Facts to design against:

* **Only the 156-basic slice has ever passed it, in either model class.** At
  fixed class, layout, anchor chain and instrument, 156 → 228 costs **three of
  five codecs** and 0.0315 of dial monotonicity.
* **Every failure in the measured population is an ORDERING inversion; zero are
  clamps.** So a monotone output spline cannot fix it — the `id100`/`negtail`
  chain moves `A7r` by exactly zero codecs — and neither can anything that
  changes the dial's range.
* **It survives seven training-recipe variants** (uniform pairing, either
  within-ref ladder, both, high-q-boost, KADIS, class-C): all read 5/5 failing,
  not one of 35 codec cells clearing the mentor. The recipe axis is empty.
* The one ordering-aware lever in the trainer is **`--monotonicity-reg`**, and
  it is wired **only on `--per-sample-alpha-head`** — as are
  `--konjnd-aggregation-*`, `--pjnd-passthrough-*` and `--monotone-cbc`.

**And `--coarse-decay` is NOT wired on that head** (fixed 2026-09-05 to fail
loud): it rides `apply_post_adam_penalties`, which only the plain training loop
calls. An alpha-head arm therefore needs its own no-coarse-decay plain control,
or it differs from its baseline by two things.

**Identity: the product already returns 100 for byte-identical input.**
`Zensim::compute` short-circuits to `(100, 0, zeros)` before the model
(`metric.rs:3509/5225`). C5 is still the right gate — it governs NEAR-identity,
which is where a near-lossless dial lives — but a C5 failure is never a claim
that `zensim(x, x) != 100`. For a 944-layout reader set, the identity vector is
not zero and the contamination above 5e-3 is exactly four slots
(`LUMA_MEAN_REF` f926/931/936/941, 0.45–0.64 % of layer-0 mass); the `id100`
anchor chain takes the contract 5/6 → 6/6 with rank bit-unchanged.

**THREE MEASURED FACTS ABOUT THE FAST CLASS ITSELF (2026-09-05 results).**

* **A 156-plus-peaks model IS 944-competitive.** `S372_S228_H128_p` at k = 3 —
  372 layout, 228 slice, `--hidden 128`, 37,923 B — reads composite **0.8732** /
  CID22 **0.8896** / KonJND **0.4999** against era-closed 944-leader bars of
  0.8626 / 0.8877 / 0.4782, on verified-identical rulers (504 KonJND refs, 4,292
  CID22 pairs). Do not assume the fast class is behind on rank; it is not.
* **CAPACITY IS NOT A LEVER — start at `--hidden 32`.** Across six set×width
  pairs, H32 vs H128 moves composite by −0.0038…+0.0021, inside every seed
  spread, at **30–47 % fewer bytes**. The default 128 buys nothing measurable.
* **MORE COMPUTE MAKES THIS RECIPE WORSE.** The same recipe with no
  `--keep-features`, free to read all 944 coordinates, is the *lowest*
  non-degenerate cell measured (composite 0.8581, KonJND 0.4191 against the 228
  slice's 0.4543). So a KonJND gap in this class is **not** a compute gap, and
  widening the feature set is not the fix.
* **And it is not slower.** Measured 2026-09-06, 8 cells × 10 starts on an idle
  box: that candidate is **faster than `zensim_D` — Profile D through the
  standard production path — in every cell** (max ratio 0.9733) and 3.73–3.97×
  `fast_ssim2` at 1T. Its forward pass is **below the measurement's noise
  floor**. So a fast-class MLP of this shape costs nothing against the shipped
  additive head; do not trade rank away for a speed worry that is not there.
* **Beware the W4 bar arm.** `add156_156basic` builds `V1PoolsMode::Off`, which
  `fold_engine::pools_mode_for_need` never returns — and it is measurably
  **slower** than the `Peaks` walk production actually runs (7.660 vs 6.280 ms
  at capv3/1T/576²). Report against `zensim_D` as well as against the bar.
* **The per-sample-α head inverts on this recipe** (raw CID22 −0.8921 at depth
  2 — a *better* ordering than the plain path, backwards), which is why its pack
  cannot fit a monotone spline. Everything gated behind that head
  (`--monotonicity-reg`, `--konjnd-aggregation-*`, `--pjnd-passthrough-*`,
  `--monotone-cbc`) needs the head's output orientation fixed first.

Evidence: `benchmarks/fastclass2_campaign_2026-09-05.md`,
`benchmarks/kernel_fastclass_2026-09-05.md`,
`benchmarks/d_peaks_jxl_floor_2026-09-05.md`,
`benchmarks/ladder_floor_resolution_2026-09-05.md`.

### 1c. THE DIAL CONTRACT CAN BE MADE STRUCTURAL — and what that costs (2026-09-06)

Evidence: [`benchmarks/best_of_all_2026-09-06.md`](../benchmarks/best_of_all_2026-09-06.md).
Read §1b first — this section is about the OTHER half of the ship decision.

**C5 and C6 are not a fitting problem.** The gate record proves no monotone
output spline can satisfy both C2 and C6 when real cells out-rank a perfect copy
in raw space. `zensim_mlp_train --nonneg-distance` fixes it in the weights:
scale-only standardization, hidden biases frozen at 0, ReLU, output weights
projected `≤ 0`, output bias frozen at the pin. MEASURED on the 228-slot recipe
at k=3, against its own control on the identical chain:

| | control | `--nonneg-distance` |
|---|--:|--:|
| contract | **4/6** | **6/6 on every seed** |
| cells above identity | 1,642 / 1,650 / 1,182 | **0 / 0 / 0** |
| `tied` | 0.0017 | **0.0000** |
| CID22 (k=3) | 0.8891 ±0.0042 | 0.8800 ±0.0049 |

C6 → 0 **while `tied` goes down**, so the either/or is dissolved rather than
traded. **Price: −0.0091 CID22, −0.0096 AIC-3, −0.0083 composite** — all larger
than the control's own seed spread, so the cost is real.

**Three things to know before you use it.**

1. **It buys C5 and C6 only.** Not C3/C4 (that is the negrich anchor), and **not
   A7r** — the per-codec floors get *worse* on four of five codecs.
2. **The identity half is CONDITIONAL on zero-preserving feature transforms.**
   `winsor_p99` with `lo > 0` maps `0 → lo`, and the canonical 372 screen carries
   28 such guards, so `raw(identity) = 99.6138` rather than the 100.0 pin.
   `raw(x) ≤ pin` stays structural; **identity being the argmax does not**, and
   C6 = 0 becomes a measurement on 9,593 cells. C5 survives because the identity
   ANCHOR rows take the same forward. The trainer warns; making it structural
   needs `lo = 0` guards or a pin at `t(0⃗)`.
3. **`g` is CONVEX at one hidden layer**, and the plain path is 1-layer only
   (`--keep-features` is refused with `--n-hidden-layers >= 2`).

**The ladder hinge repays the C1 cost, and it is a VARIANCE result.**
`--tv-pairs-file` + `--tv-weight` (the owner; there is no `--ladder-hinge`) over
material adjacent-setting pairs — pairs the reference metric orders by ≥ 0.5
ssim2 points, built by `scripts/canonical_corpus/build_ladder_tv_pairs.py`:

| | `mono` (k=3) | spread |
|---|--:|--:|
| control | 0.94602 | 0.00436 |
| `--nonneg-distance` | 0.93245 | 0.00606 |
| `+ --tv-weight 0.5 --tv-margin 0.25` | 0.94159 | **0.00096** |
| `+ --tv-weight 2.0` | **0.94648** | 0.00861 |

`w = 2.0` fully repays the monotonicity; `w = 0.5` cuts the seed spread **6.3×**.
Both improve all five per-codec floors over the architecture alone, both cost
~0 CID22, and **neither moves A7r**. `--tv-margin` is load-bearing: a pure hinge
is minimized by collapsing every ladder flat, and flat ladders are `tied` = C2.

**Bottom line for a ship candidate:** the contract is now a *choice* rather than
a barrier, at a known price. A7r remains what it was before this lane — the
binding clause, and a weights property no packaging, spline, anchor, recipe or
ladder supervision measured here has moved.

## 2. The top models (2026-07-18) and what each is FOR

> **2026-08-05 — Profile `C` shipped (SOTA-944 era; supersedes this table's
> frontier for the 944 class).** `ZensimProfile::C` = `W10L9_s4003_packed`
> (944-regime corrected-mix, k=8-confirmed; CID22 0.8867, LIVE 0.9604, CSIQ
> 0.9331, HF-NL per-ref 0.7334, dial mono 99.32% dial-units, M3a 0.862 GOLD,
> corruption head). `B` remains the default — the C-vs-B trade + full
> provenance/repro: `docs/PROFILE_C_REPRODUCTION_2026-08-05.md` + campaign
> appendix K.R. The table below is the 2026-07-18 (372/156-era) record.

| bake | arch | headline | weakness | role |
|---|---|---|---|---|
| **`Ebothg_scr0.5_dial`** | 156→128→1 MLP + winsor + spline | CID22 0.879 · nonphoto 0.906 · **HF-NL 0.712** (best ever) · LIVE 0.959 · dial 0.985 | KonJND 0.271 | best all-around candidate |
| **`Ebothg_hfgain_winsor_dial`** ("winner_dial") | same, no bigcodec mass | **CID22 0.894** · LIVE 0.960 · CSIQ 0.958 · best jxl RD (+4.5% ssim2) | KonJND 0.335 · HF-NL 0.587 | best pure-CID22/photo rank |
| **`ADD156_safesyn_only_raw_lasso`** | truly-additive basic-156, 3.6 KB | steer 0.849 (best) · exact fixed gradient · LIVE 0.960 | CID22 0.863 | exact-map / tiny-footprint |
| **`d_sdr_add156_id100_negrich_dial`** (shipped **`ZensimProfile::D`** since 2026-09-05) | the SAME ADD156 weights (byte-identical, weight-sha `330d8c09…`) + the **id100+negrich** dial spline | rank unchanged (CID22 **0.863380**, bit-identical on 11 of 14 corpora) · dial CONTRACT **6/6** + REGRESSION **7/9** · identity **100.000** · reach 156.55 · negtail floor −213.1 · carries `zentrain.repro` | CID22 0.863 (a D-lineage property, not a dial one) · G-ADDR A7/A9 structurally unreachable | **the shipped fast SDR dial**; era break vs every pre-2026-09-05 D dial number |
| **B** (shipped) | linear-372 + spline | KonJND 0.547 · HF-NL 0.614 (pre-scr0.5 best) | steering 0.24–0.66 · loses new FR holdouts | incumbent; near-lossless-conservative |
| **`F_nonneg32`** (2026-09-06, **PROPOSED — not shipped**) | 228-slot `basic+peaks` → **32** → 1, `--nonneg-distance` (bias-frozen ReLU, output weights ≤ 0, output bias = the pin) + the negrich/id100 pack chain | dial **CONTRACT 6/6 on every seed** (C5 0 outside, C6 **0** cells above identity, `tied` 0.0000) · CID22 **0.8824 ±0.0036** (+0.019 over shipped D) · M3a **0.8212**, best in its wave · serves through `V1PoolsMode::Peaks`, the mode `D` already uses | **A7r 5/5 FAIL** — the binding clause, unmoved · CID22 −0.0091 vs its own unconstrained control · `g` is CONVEX at 1 hidden layer · the identity pin is conditional on zero-preserving transforms | **the first fast-class model to hold the dial contract.** Read `benchmarks/best_of_all_2026-09-06.md` §2.4 before quoting "structural" |

**Dial-era note (2026-09-05).** `ZensimProfile::D`'s bake was flipped to the
`id100+negrich` dial by user decision. The forward pass is byte-identical, so every
RANK number in this table is unaffected; every stored **dial** value from `D` predates
the flip and must be re-read, not rescaled (the remap is a PCHIP spline, not an affine).
Recipe: [`../benchmarks/d_id100_2026-09-04.md`](../benchmarks/d_id100_2026-09-04.md);
install + gates + the two blockers that stopped the `-peaks-` variant:
[`../benchmarks/d_ship_flip_2026-09-05.md`](../benchmarks/d_ship_flip_2026-09-05.md).

**STATUS after the 2026-09-05 G-ADDR ruling — `D` is the SDR proposal, and its
row above is now understated.** Re-graded on the FLOOR-DENSE 372 ladder
instrument under the OPERATIVE `resolvable` window, shipped `D` reads
**`SHIPPABLE (regression PASS + contract PASS)`** — 5/5 codec floor
representability (exceeding the mentor on `avif-rav1e` 0.6667 vs 0.6410 and
`jxl` 1.0000 vs 0.9615) and CONTRACT 6/6. The "REGRESSION **7/9**" in the table
is the RETIRED grading: `A1`-`A6` are `report-only` since the ruling, so the
regression tier is `A7r` alone and D passes it. **All 97 re-graded board cells
fail `A7r` on that instrument — there is no alternative candidate.** Profile B
still leads RANK (CID22 0.8821 vs D's 0.8634) while failing the dial on every
codec. `D` is on the board only as `d_id100_negrich@did100lane` (bake sha
verified byte-identical); promoting it under its own name is a registered
follow-up. Record:
[`../benchmarks/default_proposals_2026-09-05.md`](../benchmarks/default_proposals_2026-09-05.md).

**A BOUNDED `D` IS AVAILABLE AND FREE — staged 2026-09-06, NOT installed.**
`D`'s 28 declared ids include **f116 and f155**, two of the twelve unbounded
`contrast_inc` slots (`feature_defs::DEFECT_F17`), and it reads them RAW — it
has no transform block at all. The serve-time guard (winsor `[p0.1, p99.9]` on
those slots, with the output spline REFIT on the guarded net so the calibration
matches what the runtime sees) was measured against the shipped bake on the
runtime-era postC root with the FLOOR-DENSE ladder instruments and costs
**nothing**: CID22 an exact tie (paired-bootstrap CI `[+0.00000, +0.00000]`),
all twelve board corpora inside `8.2e-5`, contract 6/6, every per-codec `A7r`
floor exactly equal, identity 100.000000, inversions and the corruption gate
unchanged, and **0 of 4,292 CID22 rows perturbed**. Candidate:
`d_sdr_add156_id100_negrich_guard12_2026-09-06.bin` (1,523 B, sha256
`c2db9e29…`) at
`/mnt/v/output/zensim/rev2-d-arms-2026-09-06/guard/shipped/`. It is **not** a
rank win — it is a rank NO-OP and a small outlier-ordering loss — so the case
for it is a bounded input at a measured cost of zero, and **the user rules**.
Installing it forces a `dense_bake_flip_gate` change (re-point, never weaken):
recipe [`../benchmarks/rev2_d_arms_2026-09-06.md`](../benchmarks/rev2_d_arms_2026-09-06.md)
§12.11. Board: `D_shipped@dguard2`, `D_guard12_p999@dguard2`.

**`ZensimProfile::BHdr` — the HDR proposal is KEEP, with named caveats.** UPIQ
pooled (n=380) **0.7536**, above ssim2-PU 0.7044, above the PU-SSIM literature
bar 0.7395, and above every one of the 24 HDR944 cells and every arm seed-group
mean (best single seed `HDR944_L1T1_s4004` 0.7254; frozen `CHdr` 0.6664, which
loses by −0.0872 at paired p = 0.0000). **But: G-ADDR on HDR is NOT MEASURED and
no instrument exists that could measure it** (an HDR ladder is the registered
follow-up); `bhdr_…@cur372`'s G-ADDR verdict was cut on the SDR codec grid and
must NEVER be quoted as BHdr's HDR dial; BHdr's own promotion was
selection-adjusted **maxT p = 0.221, not significant**; and it overlaps its own
HDR census instrument (7 of 9 scenes). Same record, §2.

Bakes + `.spec.json`/`.metrics.json` sidecars: `/mnt/v/output/zensim/corr-lq/` +
`/mnt/v/output/zensim/screen-retrain-2026-07-18/`; index: `BAKE_INDEX.md` (built by
`build_bake_index.py`). Dashboard: `bandwise_dashboard_2026-07-18.html` under
`/mnt/v/output/zensim/reports/`.

## 3. Exact reproduction

**scr0.5 (and winner: drop the bigcodec line):**
```sh
C=/mnt/v/zen/zensim-training/canonical-2026-05-21/train
./target/release/zensim_mlp_train \
  --group safesyn:$C/safesyn.parquet:1.0:0.5:both \
  --group cid22_train:$C/cid22_train.parquet:1.0:2.0:both \
  --group kadid:$C/kadid.parquet:0.5:1.0:rank \
  --group tid:$C/tid.parquet:0.5:1.0:rank \
  --group bigcodec:/mnt/v/output/zensim-multicodec-probe/bigcodec_traindigits_2026-07-02.parquet:0.5:1.0:both \
  --n-hidden-layers 0 --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 --seed 13 \
  --max-features 156 --allow-narrow-features \
  --feature-transform winsor_p99:12:0,3.1492 --feature-transform winsor_p99:38:0,1.5612 \
  --feature-transform winsor_p99:51:0,3.0994 --feature-transform winsor_p99:77:0,2.1495 \
  --feature-transform winsor_p99:90:0,3.079  --feature-transform winsor_p99:116:0,2.1574 \
  --feature-transform winsor_p99:129:0,3.4202 --feature-transform winsor_p99:155:0,2.3352 \
  --out OUT.bin
# dial (rank-invariant), then the sidecar:
./target/release/bake_dial_refit add-spline --in OUT.bin --out OUT_dial.bin \
  --anchor $C/multiband_anchor_dial100.parquet --target-col target_score
python3 scripts/v_next/emit_bake_metrics.py OUT_dial.bin
```
(The winsor bounds are safesyn p99s for the 8 `hf_gain` features, recorded from the shipped
bake's own metadata — recompute the same way if safesyn changes. bigcodec at w=1.0 is
SATURATED: +0.008 nonphoto for −0.035 KonJND and −0.025 LIVE — don't.)

**ADD156:** `python3 scripts/v_next/additive_basic156_probe.py` (imports the linear solver's
`MixGram`/`fit_spline_knots`/`bake_candidate`; slices the Gram to f0..155 — that IS the
w[156:]=0 constraint). **B:** `benchmarks/profile_b_methodology_2026-07-12.md`.

## 4. Doing a NEW model properly (the loop)

1. Search `zenpapers` for the science; write the 4-line hypothesis + falsification
   (CLAUDE.md "Principled experiment workflow").
2. Recipe: start from §3; keep a held-out val group; NEVER add CID22 MOS or AIC/T0 corpora.
3. Train seed-13 first; `add-spline`; `emit_bake_metrics.py` (the sidecar IS the scorecard
   row); check the panel + dial + steer-mass (`closed_loop.diffmap_basic_fraction` — for HDR
   candidates gate ≥0.5 BEFORE more training).
4. If it survives: coherence (`diffmap_block_coherence --bake`, right fold per family) and
   the RD probe (`scripts/v_next/rd_probe_2026-07-18.sh` with the worktree binaries +
   `rd_probe_analyze`) — bytes at equal JUDGED score is the anti-gaming verdict.
5. Rebuild the dashboard (`bandwise_dashboard.py --bakes ...`), write the benchmarks doc with
   honest losses, commit data + doc, update memory. Ship swaps are user-gated.

**Extracting a NEW table? Use the research mode, and its manifest answers
"what is column N?" for you** (2026-09-05). `ZENSIM_AB_MODE=research` on
`zensim/examples/v2_ab_extract.rs` (env `ZENSIM_RESEARCH_{SET,WIDTH,ERA,REVISION}`)
writes the CSV **and** a `_MANIFEST.json` with one provenance row per column —
name, family, scale/channel, statistic, per-slot cost, difference-form,
monotone direction, owning kernel, resolved revision, any live defect id, and
whether the plan populated the position or left the structural fill. It also
REFUSES loudly, before decoding a single image, if the requested set or era
cannot be produced. Bit-identical to the production walk at the same plan
(60 CID22 pairs, byte-identical CSVs) and **+0.2 % cost**; a narrower set is
genuinely cheaper (`basic+peaks+masked+iw@372` runs at 0.49×). Record:
[`../benchmarks/feature_system_phase2_2026-09-05.md`](../benchmarks/feature_system_phase2_2026-09-05.md).

**Pitfall list (all measured, all in git):** trainer-emits-MLP-always · MSE-collapse-fake-dial
· spline-on-spline · abs-vs-signed fold per family · `DiffmapResult::score()` pre-834b4387 ·
inert zenjpeg passes · legacy-seeded codec tables · SROCC-only verdicts · train==val KADID/TID
· dial-grid quarantine (`_quarantined_v2`) · hf_gain unbounded · IW 1/n-vs-Σw · bigcodec mass
poisons LINEAR CID22 (MLP absorbs it) · KonJND anti-correlates with bigcodec mass in this
family · **ext-lineage KADID target stored INVERTED** (below).

> **⛔ KADID orientation (2026-08-04) — read before any recipe that touches a `kadid` group
> or cites a KADID number.** `ext720`/`ext924`/`ext944` `ext_kadid.parquet` store
> `human_score = (5−dmos)/4`, the INVERSE of the canonical `(dmos−1)/4`. Training on an
> ext root teaches the model to rank KADID **backwards** (measured dose-response: train
> weight 0.50 → mean −0.457 vs true quality, 1.50 → −0.925), and every SROCC measured on
> an ext root is sign-flipped. **The recipe above is SAFE — `$C` is
> `canonical-2026-05-21/train`, which is correctly oriented** — and that is exactly why
> `winner_dial` is **+0.9464** on KADID rather than the −0.9464 the board used to print.
> Gate any new corpus table with
> `scripts/canonical_corpus/check_target_orientation.py` before training on it.
> Full determination: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F;
> ledger `docs/DATASET_HISTORY.md` §3.20.
