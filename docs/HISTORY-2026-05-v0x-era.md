# HISTORY — May-2026 V0_x / PreviewV0_5 era (excised from CLAUDE.md)

Excised from CLAUDE.md 2026-07-19 per its own supersession banner; current
guidance: docs/TOP_MODELS_COOKBOOK.md

Sections below are verbatim as they stood in CLAUDE.md on 2026-07-19:
May-2026 training goals + three-trail SOTA + shipping/experiment-rigor
policies, the 2026-05-1x eval mandates (SROCC-only ban / full-stat panel /
10-band rule / multi-target TODO), V_20 + V39 learnings, the
canonical-2026-05-18 corpus archaeology, the V_X principled experiment
workflow, the V0_1-era weight-training status, the V0_7 e1 fill status, and
the interactive comparison-site spec (implemented since: `site/compare.html` +
`site/js/compare-worker.js`; its 'Status: not started' line was stale).

## Training goals (priority order, locked 2026-05-10, revised 2026-05-11)

zensim is a **user-facing quality dial** — users type a target zensim score
and the codec stack picks an encode that hits it. Every training and
evaluation decision flows from this:

1. **Match-or-exceed fast-ssim2 across all quality bands** (PRIMARY goal,
   revised 2026-05-11 per user directive). Consistent results relative to
   subjective quality across the full 0-100 range matters more than dial
   smoothness. Specifically: per-band SROCC on KADID/TID/CID22 must
   match-or-exceed fast-ssim2's per-band SROCC. CID22 aggregate SROCC
   must reach fast-ssim2's level (0.8895) — this is the **shipping bar**.
   Tighter calibration scatter (lower per-bin residual) is the secondary
   shipping consideration. **Adding new bakes and swapping the shipped
   weight is explicitly permitted to achieve this.**

2. **CID22 SROCC is the gold standard for cross-band evaluation.** Sneyers
   / Ben Baruch / Vaxman *AIC-3 Contribution from Cloudinary: CID22*
   (2023, JPEG WG1 `wg1m99012`) is the only large held-out human-MOS
   dataset that exercises **codec-output distortions specifically**.
   KADID-10k and TID2013 are **NOT compression-tuned** — KADID's
   distortions are ~95% non-compression (blur, noise, color, geometric);
   TID2013 is similar. Use them as **integrity guards** alongside CID22,
   not as the only optimization targets. Per-band CID22 SROCC is the
   shipping gate for goal #1.

3. **Smoothness AND monotonicity** are first-class objectives but
   secondary to band coverage (revised 2026-05-11 per user directive).
   The user typing "give me zensim 85" still needs monotonic behavior,
   but the bar has been **lightly raised to accommodate band coverage**.
   Bumpiness target: **≤ 6.0%** non-monotonic q-step rate on JPEG unified
   parquet (raised from V0_2's 4.86% floor → V0_7's 5.5% → V0_8's
   6.0% — the latest raise permits V0_8's B1-for-smoothness trade,
   which closed B1 SROCC from -0.027 to -0.014 vs ssim2 at the cost
   of +0.41% non-mono). ssim2 GT is 5.08%. TV regularization in
   `train_v_next_mlp.py --tv-weight 10..30` is the lever. If a bake
   achieves goal #1 and band coverage but exceeds 6.0% bumpiness,
   surface to user for case-by-case decision.

3. **Anchor at perceptibility thresholds.** KonJND-1k (`/mnt/v/dataset/konjnd-1k/`,
   1008 src × 504 JPEG + 504 BPG, mean PJND scored against ssim2 ≈ 63
   per CID22 paper Table 4) is the anchor. A trained model must score
   at-PJND pairs ≈ 63 ± 5; if it saturates to 100 there, "visually
   lossless" calibration is broken. Validate every champion via
   `dataset_metric_baseline --konjnd ...`.

4. **Filter synthetic training data by ssim2 ↔ butteraugli agreement.**
   The 218k clean safe-synthetic includes pairs where ssim2 and
   butter disagree on relative quality ranking — those are noisy
   labels for ranking-based training. Drop pairs (or whole curves)
   where the two metrics' relative ranking disagrees within an
   `(image, codec)` group. The CID22 paper (Tables 3, 6) flags
   regimes where ssim2 is less accurate (very high q, very low q);
   concordance with butter is the simplest cross-check available
   without human MOS.

5. **The CID22 paper governs ssim2-accuracy regions.** From Tables 3
   (per-codec SROCC) and 6 (pairwise SROCC vs absolute):
   - ssim2 is most reliable in q-band ~50..90
   - ssim2 less reliable at q > 95 (saturation, near-lossless tail)
   - ssim2 less reliable at q < 30 (extreme distortion outside its
     training distribution; KADID-style analytic distortions overlap
     with this)
   - When training, weight pairs in q-band 50..90 higher OR drop
     pairs outside [30, 95] OR weight by butter-concordance (which
     captures the same signal indirectly).

### What NOT to optimize

- Aggregate (KADID + TID + CID22) / 3 SROCC as a SOLE target. KADID and
  TID alone are not weighted for compression; they're valuable as part
  of "match-or-exceed ssim2 across all bands" (goal #1, 2026-05-11),
  but not as the only signal. Always report all three plus per-band
  CID22 alongside any aggregate.
- Synthetic ssim2-target val_srocc as a primary target. Synthetic
  val tracks the trainer's own loss, not held-out human judgement;
  it has been > 0.99 across most of our 30+ training runs while
  CID22 stayed at 0.85-0.88. Use synthetic val only as a sanity
  guard against pipeline breaks.
- Metrics that average over very-low-q (q < 30) and very-high-q
  (q > 95) ssim2 — those bands are unreliable per the CID22 paper,
  but per-bin SROCC at 5-unit granularity in those bands IS the
  pathology view we use for goal #1 (band coverage).

### Three-trail SOTA (added 2026-05-18, expanded 2026-05-19)

The PreviewV0_5 slot now ships **three parallel variants** defending
distinct Pareto frontiers. Read `zensim/SOTA_TRAILS.md` for the full
framework, gate criteria, and candidate matrix.

| Variant | Bake | When to use |
|---|---|---|
| `PreviewV0_5` / `PreviewV0_5Balanced` | V_22-mix-LARGE+iwssim (41 KB) | General-purpose. Best balanced-corpus coverage (KADID 0.9677, TID 0.9729, KonJND 0.8927). |
| `PreviewV0_5Compression` | V_24-per-sample-α s4 (44 KB) | Codec selection / quality dials. Wins CID22 0.8641, AIC-3 0.8183, TID 0.8893 decisively vs prior 372feat ship per § A.9; KADID/TID/KonJND vs Balanced within −0.10 noise tolerance. Uses `zentrain.per_sample_alpha_head` runtime dispatch in `zensim::metric::forward_one_bake` (landed 2026-05-18). |
| `PreviewV0_5Ensemble` | V_05-ensemble classifier (23 KB) + Balanced + Compression bakes | Single-profile alternative when downstream can't pick a trail per-call. Routes each pair through a 300→64→1 classifier; SROCC tracks max(Balanced, Compression) to within 0.014 per corpus. |
| `PreviewV0_5Tuner` | V_tuner-v2-s2 calibrated (261 KB, F32) | **Codec auto-targeting / quality dials.** User types "score 70", codec binary-searches the q yielding zensim≈70. Strict monotonicity 92.78 % on the 50-image × 19-q JPEG sweep with 0.44 % tied — vs V0_5 ships' 71–86 % strict monotonicity AND 57–76 % tied (clamp-flat dead zones). **NOT for general ranking** — CID22 0.8786 (competitive) but KADID/TID/KonJND drop −0.20 to −0.66 vs Balanced. See `PreviewV0_5Tuner` variant doc for the caveat. |

**Gate per trail (formal):**

- **Balanced trail.** A>>B on CID22 decisively per § A.9 AND not
  decisively B>>A on any of {KADID, TID, KonJND, AIC-3} aggregate.
- **Compression trail.** A>>B on ≥1 of {CID22, AIC-3} decisively per
  § A.9 AND not decisively B>>A on the other compression corpus AND
  mean SROCC regression on {KADID, TID, KonJND} no worse than −0.10
  on any single corpus.
- **Tuner trail.** Strict monotonicity ≥ 1 pp better than every V0_5
  rank-trail ship on the JPEG 50-image × 19-q sweep AND tied rate ≤ 5
  % AND dynamic range ≥ 50 score units. NO SROCC gate — rank-honest
  cross-corpus performance is explicitly secondary for this trail.

A new bake ships to the trail whose gate it passes. If passes
multiple trails: ship to all. If passes none: add row to candidate
matrix in `SOTA_TRAILS.md` and move on. The Tuner trail's 2026-05-19
ship explicitly fails the rank-trail gates by construction
(safesyn-only training) — follow-on tuners that close the rank gap
without losing monotonicity are an open research direction.

**Don't bump the crate version** when rotating a trail's bake (per
user 2026-05-18: "we don't want crate bumps every time we get a nice
bake"). The `ProfileParams` static slot owns the bake bytes; rotation
is a patch-level swap.

**Runtime status (2026-05-18, post-dispatch)**: per-sample-α
dispatch landed in `zensim::metric::forward_one_bake`,
`bake_verdict::score_row`, and `bake_compare::score_corpus`.
V_24-per-sample-α s4 packed (CID22 0.8641, AIC-3 0.8183, packed
44 KB, md5 `f09a9abdce00805000c1d112c2421b2d`) replaced
V_22-372feat s5 on the compression trail. The dispatch reads the
`zentrain.per_sample_alpha_head` metadata, treats `Predictor::predict`'s
output as the post-LeakyReLU hidden vector (because the bake's
final layer is an `n_hidden × n_hidden` identity passthrough), and
mixes a rank head + pool head via a per-sample sigmoid gate.
Regression test at
`zensim-validate/tests/per_sample_alpha_runtime.rs`.

### Shipping policy (revised 2026-05-14 — gates are ADVISORY)

The shipped weight at `zensim/weights/v0_X_<date>.bin` may be added,
swapped, or rotated to advance goal #1 (match-or-exceed ssim2 across
all bands) AND to **dramatically improve low-q (B0..B5) bands**, the
regime where compression product decisions live.

Per user directive 2026-05-14: **CID22 aggregate SROCC and
non-monotonic q-step rate are advisory, not hard ship-blocking
gates.** A bake that drops CID22 by 0.005 while gaining +0.05 on
B0/B1 IS the winning trade. Surface the per-band picture; let the
user make the call.

When swapping, REPORT (don't block on):
1. Per-band SROCC (10-band B0..B9 + legacy 4-band CID22 cuts) vs
   fast-ssim2 baseline on KADID, TID, AND CID22. Flag bands where
   the new bake loses ssim2 separately from bands where it wins.
2. Non-mono q-step rate on JPEG unified parquet (raw + after
   soft-iso, aggregate + per-band). Historical reference: V0_2
   4.86 %, V0_8 5.87 %. Above 6 % is **noted, not blocked**; user
   decides if the trade is worth it.
3. Apply affine calibration via
   `scripts/v_next/affine_calibrate_znpr_v2.py` so calibrated output
   range matches truth distribution (p5..p95 ≈ ssim2 truth p5..p95).
4. Archive the prior shipped weight at `zensim/weights/archive/`
   (with date stamp) for reproducibility.
5. Update CHANGELOG.md with verification numbers in the `[Unreleased]`
   section.
6. **Land a paired methodology doc** at
   `benchmarks/v0_X_methodology_YYYY-MM-DD.md` BEFORE flipping the
   `include_bytes!` in `zensim/src/profile.rs`. Template:
   `benchmarks/v0_18_methodology_2026-05-13.md`. The doc MUST cover
   (a) architecture + parameter count + bin size + md5,
   (b) full trainer command + every hyperparameter + every input
   file's MD5 + row count,
   (c) lineage for built-from-prior-bakes constructions (ensemble,
   concat, finetune, KD) — each component documented to the same
   depth,
   (d) calibration script + α/β,
   (e) held-out SROCC on KADID/TID/CID22/AIC-3/AIC-4/KonJND with
   the **10-band width-10 grid (mandatory)** AND the legacy 4-band
   CID22 Table 5 cuts AND step-5 (20-bin) per-corpus,
   (f) non-mono q-step rate (raw + after soft-iso, aggregate +
   per-band),
   (g) data-lineage table (path / MD5 / row count / CID22-contam
   status) for every training input,
   (h) honest gaps — what the new bake does WORSE than the prior
   ship and why shipping anyway is the right trade.

A bake without a methodology doc = **untrustworthy bake**. Numbers
can be reproduced; without methodology they can't be verified,
can't be improved on, and can't survive context loss. Effective
2026-05-13.

### Experiment-rigor policy (added 2026-05-14, user directive)

**Push every experiment to the paper-claimed benefit before
falsifying it.** A single half-tuned run that fails to reproduce a
paper's headline number is not a falsification — it's a
hyperparameter / seed / recipe miss. Per paper:

1. Quote the paper's claimed lift (e.g., "IW-SSIM +0.006 SROCC vs
   MS-SSIM on TID2008") AND the paper's experimental conditions
   (corpus, training split, hyperparams).
2. Run our adaptation. If our result is BELOW the paper's claim by
   more than measurement noise: extend the sweep (more seeds,
   broader hyperparam grid, longer training, alternative
   reimplementations) before declaring failure.
3. Falsification requires: a documented sweep that exhausted the
   paper's described configuration space PLUS at least one
   reasonable extension, AND the result still fails to match.
4. Land a `benchmarks/v0_X_method_<paper>_2026-MM-DD.md` doc with
   the paper claim, our sweep grid, the result-vs-claim table, the
   git commits, and the input MD5s.

**Architecture is open.** Adding features (LMS / opponent channels,
IW-weighted pooling, distortion-manifold encoder, JND-anchored
calibration) is welcome if scientifically motivated. Don't be
precious about the 228-feature input shape or 228 → H → 1 MLP
topology.

**B0..B5 lift is the dominant priority.** A bake that wins B6..B9
but loses B0..B5 is the wrong direction — low/mid-q is where
compression product decisions live.

CID22 training data still must NOT be added to the trainer (the 49
held-out reference images stay sacred). All training continues on
synth-only `/mnt/v` corpus.

### Long-term goals (added 2026-05-11, user directive)

The recovery cycle continues — **V0_8 shipped 2026-05-11 (eve)**
(TV=15 seed=1, superseding V0_7's TV=10 seed=1 within the same
session). V0_8 trades smoothness for CID22: CID22 SROCC = **0.8948**
(+0.0053 above fast-ssim2's 0.8895) and **B1 SROCC -0.014** (a 50 %
reduction in V0_7's -0.027 B1 gap). Non-mono = **5.87 %**, over the
prior 5.5 % gate — the gate is **raised to 6.0 %** to permit V0_8.
Trained on safe-synthetic CSV with 1,015 perceptual-duplicate sources
removed (28 % of original 218k pairs); h=128, TV=15, seed=1, KonJND-
aligned. Affine-calibrated (α=31.1041, β=-4.3882, R²=0.76).

> **⚠️ V0_8 CID22 SROCC IS INFLATED (added 2026-05-12)**: the
> perceptual-overlap cleanup used to produce the V0_8 training CSV
> was at a looser threshold than d≤16. The 156,420-row clean CSV
> still contained **11,629 contaminated rows (7.43%)** mapping to
> 361 hex-hashed source files that were perceptual-near-duplicates
> of the 49 CID22 held-out references (22 of 49 leaked).
>
> The 2026-05-12 purge deleted those 361 sources + 30.6 GiB of
> encoded variants + .features.bin caches + tower mirror, then
> rebuilt the clean CSV at 144,791 rows (manifest at
> `benchmarks/contaminated_sources_purged_2026-05-12.txt`).
>
> V0_15 retrain on the truly-clean CSV is in flight. Expected
> honest CID22 SROCC: **0.890-0.892** (V0_8's 0.8948 was inflated
> by ~0.005 SROCC due to training-set leakage). Until V0_15 lands,
> V0_8 remains the runtime ship but its number should be treated
> as upper-bound, not benchmark.

**Runtime score-mapping fix landed in same commit**: the V0_4 slot's
profile now sets `skip_score_mapping = true`, so the V0_8 bake's
MCOS-aligned raw output (0..100 range) is returned directly without
the V0_2 `100 − 18·d^0.7·sign(d)` transform that was producing
garbage. All 5 V0_4 runtime tests now pass.

Per-band wins ssim2 in B2/B3 (+0.015/+0.051); near-parity B0/Near-PJND;
**B1 closes from V0_7's -0.027 to V0_8's -0.014** (next-cycle target
remains: full ssim2 match on B1).

Going
forward, the priorities are:

1. **Pure-Rust training pipeline that runs in WebAssembly on background
   workers** with **CubeCL acceleration**. Interactive exploration:
   user adjusts weights / targets in browser → background worker
   retrains → updated plots stream back. Replaces the current Python
   trainer (`train_v_next_mlp.py`) which can't ship to browsers.
   Owner crate: TBD (likely a new `zensim-train-wasm/` workspace
   member). Dependencies: CubeCL (GPU compute in WASM), wasmtime
   (host runtime), zenwasm-abi (existing host-cdylib loader).

2. **Reproduce CID22 paper methodology end-to-end**. The 2023 Sneyers
   / Ben Baruch / Vaxman paper (and any subsequent revision) is the
   methodology spec for image quality metric evaluation. Required:
   - Match the per-codec SROCC numbers (paper Table 3) for ssim2,
     butteraugli, and our zensim profiles.
   - Match the pairwise SROCC numbers (paper Table 6).
   - Match the per-band statistics (paper Table 5 cutoffs).
   - Match the PJND calibration (paper Table 4, KonJND-1k anchor).
   Use the same training/validation splits the paper describes.
   When our numbers diverge from paper numbers by > 0.01 SROCC,
   investigate before shipping.

3. **Read both revisions of the CID22 paper (~30 pages each)** and
   maintain `docs/CID22_PAPER_NOTES_2026-05-07.md` as the synthesis.
   Anything that contradicts our internal practice should be flagged
   in the synthesis and resolved.

4. **Commit regularly**. Every tick must produce a measurable advance
   (training step, eval, plot, doc update with new facts). No
   SKIP-only ticks. If stuck waiting on a long-running job, switch to
   one of the long-term goals above.

### Reference materials

- Paper PDF: `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`
- Distilled notes: `docs/CID22_PAPER_NOTES_2026-05-07.md`
- Table 2 / 4 / 5 extracts + per-band rule: `docs/CID22_TABLES_2_4_2026-05-10.md`
- KonJND anchor cross-validation: `benchmarks/baseline_metrics_with_konjnd_2026-05-01.md`
- 2026-05-10 champion + recipe + Phase 4 plan: `benchmarks/champion_2026-05-10.md`,
  `docs/phase4_reference/README.md`

## SROCC-only verdicts BANNED + ssim2-target training bias (added 2026-05-15)

**STOP USING SROCC ALONE AS A VERDICT GATE.** Every ship/no-ship,
falsified/promising call MUST cite the full Mohammadi 2025 panel
(SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE). When the panel
disagrees with SROCC, **the panel wins**.

### Why this is now non-negotiable

The V_20a IW-SSIM and V_20 IW+ext+transforms falsifications used
SROCC-on-CID22 as the primary gate and called the IW direction
"falsified." That call is structurally rigged:

1. **The training corpus uses ssim2 as the golden target.** Every
   V_X bake (V_18, V_19, V_20 IS, V_20 extended, V_20a IW, V_20
   IW+ext+transforms) is trained on safesyn pairs whose
   `human_score` column is the ssim2-derived score, NOT a human MOS
   and NOT an IW-SSIM score.

2. **SROCC against the resulting predictions favors ssim2-shaped
   surfaces.** A bake trained to predict ssim2 will produce
   ssim2-shaped output. Evaluating that bake against an
   ssim2-derived ground-truth-equivalent column reports high SROCC
   when the bake is "more ssim2-like" — but IW-SSIM-quality bakes
   are deliberately NOT ssim2-shaped (Wang & Li 2011: that's the
   point of weighting by information content).

3. **The IW bakes win on TID human MOS, PWRC, and Z-RMSE — the
   "important metrics" per Mohammadi 2025.** V_20 IW+ext+transforms
   on TID: SROCC 0.9710, PWRC 0.9822, Z-RMSE 0.231 — best TID
   result ever measured, +0.018 SROCC over V_18 ship, −0.063 Z-RMSE
   (~22 % less calibration error). The same bake's CID22 SROCC
   0.4632 looks catastrophic — but CID22 was evaluated using human
   MOS that is itself ssim2-aware (CID22 weights tuned on the same
   reference images). The single-stat SROCC verdict is misleading.

### What replaces SROCC-only

- **Mohammadi 2025 full panel** at aggregate AND per-band level.
- **Multi-stat agreement** as the ship gate: a bake ships when at
  least 3 of 5 stats (SROCC, PLCC, KROCC, PWRC, Z-RMSE) agree on
  improvement vs the prior ship on the held-out corpus.
- **Per-band Z-RMSE** is load-bearing — it measures absolute
  calibration error after a 4-parameter logistic rescale, which
  decouples it from the training-target metric's shape. Two bakes
  with the same SROCC can have wildly different Z-RMSE.

### What changes operationally

- Every prior "falsified on SROCC" verdict in `benchmarks/v0_20*` /
  `benchmarks/v0_19*` / `benchmarks/v0_18_1*` is **provisional**.
  Re-evaluate against the full panel before treating the
  hypothesis as dead. Where the full panel has been collected
  (commit `0653c818`, `4b557c00`, `c0200d6c`) the verdict stands;
  where only SROCC was reported, treat the "falsified" label as
  suspect.
- New training experiments should use **multiple training targets**
  (ssim2 + IW-SSIM at minimum) so the trained bake isn't shaped by
  exactly one metric's biases. We don't currently have IW-SSIM
  scores on the safesyn corpus — that gap is the next infrastructure
  build (see "Multi-target training corpus" below).
- Per-codec / per-band evaluation should compute the full panel,
  not just SROCC per band. The current `dataset_metric_baseline`
  emits the full panel at aggregate but only SROCC + CI per band;
  closing that gap is queued.

### "ssim2 favoring SROCC" antidote

When you see a bake report "won TID + KADID, lost CID22 on SROCC,"
don't call it falsified yet. Check:
- CID22 PWRC: did it lose by > 0.005?
- CID22 Z-RMSE: did it lose by > 0.030?
- TID + KADID delta on the FULL PANEL: did they win across all 5
  stats?
- AIC-3 / AIC-4: does the bake also win there? Those are
  independent compression-focused holdouts.

If most of the panel agrees on the wrong direction, the
falsification stands. If only SROCC says "fail" while PWRC + Z-RMSE
say "wins," the SROCC-on-ssim2-trained-corpus bias is the
explanation — surface the result, don't bury it as "falsified."

## Multi-target training corpus — TODO (2026-05-15)

The safesyn training corpus at `/mnt/v/output/zensim/synthetic-v2/
training_safe_synthetic.csv` carries `human_score` = ssim2-derived
score (per the existing methodology). This is the proximate cause
of the "ssim2-favoring SROCC" bias documented above.

To produce bakes that aren't ssim2-shaped, the corpus needs
additional target columns:

- **IW-SSIM score** per pair (computed via the official Wang & Li
  2011 reference implementation — `pyiqa.iwssim` is the canonical
  Python reproducibility path).
- **CVVDP score** per pair (Mantiuk et al. — requires GPU + display
  calibration model; longer-term).
- Optionally: **butteraugli 3-norm**, **dssim**, **fast-ssim2**
  (already have these via the per-pair eval pipeline, but not
  baked into the training CSV).

Trainer changes needed:
- `--target-column NAME` flag to switch the regression target.
- `--target-mix ssim2:0.5,iwssim:0.5` for multi-target weighted
  supervision.

Until this lands, every V_X bake we ship is structurally an
ssim2 predictor with input-shaping / feature-set variations on top.
The principled workflow's "hypothesis-first" step should henceforth
include "what is the training target, and is the verdict gate
appropriate to that target?"

## Statistical rigor — mandatory full-stat reporting (2026-05-14)

Every eval that emits SROCC MUST also emit, in the same report:

| Stat | What it answers for zensim |
|---|---|
| **SROCC** (Spearman rank correlation) | Rank agreement — "if I rank by zensim, does that match human rank?" |
| **PLCC / Pearson** | Dial-honesty — "does a 1-point change in zensim correspond to a 1-point change in human MOS?" Pearson on the calibrated metric output is the **load-bearing stat for user-facing dials**, since users type a target zensim score and expect linear response. |
| **KROCC** (Kendall tau) | Rank-difference variant; sometimes more stable than SROCC at small n |
| **Outlier Ratio (OR)** | Fraction of predictions outside ±2σ of subjective. Reveals model-vs-truth pathologies SROCC hides. |
| **PWRC** (Pearson Weighted Rank Correlation) | Hybrid stat from the IQA literature — rank-transform inputs, weight by importance, Pearson. Mohammadi 2025 recommends it as one of the load-bearing five. |
| **Z-RMSE** (per-sample-σ-normalized RMSE) | From Mohammadi 2025: `Z-RMSE = √((1/n) Σ ((Ŝ−μ)/σ)²)` where μ/σ are bootstrap subjective stats per stimulus. Penalizes errors LESS where humans disagreed, MORE where the JND is sharp. **The single best stat for "does this metric track the consensus when there IS one?"** Required when the corpus ships per-sample σ (AIC-3 / AIC-4 / CID22 all do via bootstrap). |
| **MRR p-value** (Meng-Rosenthal-Rubin paired SROCC test) | When comparing two metrics A and B against the same MOS set, "is A−B difference real?" — needed because A and B are correlated via shared MOS. |
| **Wilcoxon signed-rank on residuals** (with `r = Z/√N` effect size) | Non-parametric companion to MRR — captures different aspects of significance. |

**Source rationale**: Mohammadi/Jenadeleh/Sneyers/Saupe/Ascenso 2025
("Evaluation of Objective IQA Metrics for HF Image Compression",
arXiv:2509.13150, IEEE Access) demonstrate that SROCC alone is the
**single most misleading practice** in IQA evaluation. Different
metrics disagree by different magnitudes on each statistic; the joint
pattern is the real signal. SSIMULACRA2's SROCC 0.905 vs CVVDP's 0.960
looks like a 5 % gap, but Z-RMSE shows the actual scale gap is 5×
(47.63 vs 9.45). zensim inherits SSIMULACRA2's HF saturation; we need
to MEASURE it before we can improve it.

**Mandatory tool support**: `dataset_metric_baseline`, any new
per-pair eval binary, and any web-site comparison view MUST emit
this full stat set. SROCC-only reports are a regression and should
not be accepted.

For per-band tables (10-band B0..B9 + legacy 4-band cuts), emit the
full stat set per band per corpus — yes, that's a lot of columns,
but the joint picture is what catches "winning B0..B5 while losing
B6..B9" tradeoffs we'd otherwise miss.

When `--per-pair-output` is set, the per-pair CSV becomes the input
for the bootstrap σ computation; the eval should compute and cache
bootstrap σ per (corpus, sample) once and reuse it across metric
comparisons.

For statistical-test outputs (MRR, Wilcoxon), report **p-values + effect
sizes**, never just p<0.05/p>0.05. The user reads the trade-off
explicitly.

## Per-band reporting rule (10 bands required, 2026-05-14)

Every CID22/KADID/TID/AIC-3/AIC-4/KonJND eval MUST report **10
bands**, not 4. The 4-band CID22 Table 5 cuts are kept alongside
for compatibility with the 2023 paper, but the **10-band grid is
the primary release gate**.

The 10 bands tile the 0..100 MCOS / SSIMULACRA 2 score range with
uniform width 10:

| Band | Score range |
|---|---|
| **B0** | 0 ≤ s < 10 |
| **B1** | 10 ≤ s < 20 |
| **B2** | 20 ≤ s < 30 |
| **B3** | 30 ≤ s < 40 |
| **B4** | 40 ≤ s < 50 |
| **B5** | 50 ≤ s < 60 |
| **B6** | 60 ≤ s < 70 |
| **B7** | 70 ≤ s < 80 |
| **B8** | 80 ≤ s < 90 |
| **B9** | 90 ≤ s ≤ 100 |

`Near-PJND` (58 ≤ s ≤ 68) is reported as an additional sub-band
(spans B5+B6) — KonJND's PJND mean lands here and a regression in
this region breaks "visually lossless" calibration.

Legacy 4-band CID22 cuts (B0<50 / B1 50-65 / B2 65-90 / B3 ≥90)
are reported alongside the 10-band grid in every eval that touches
the CID22 corpus, since the 2023 paper's Tables 3-6 use them.

For each (model, dataset) eval the harness MUST emit the **same
full statistical-rigor panel per band** as the aggregate (per the
"Statistical rigor" section above): SROCC, PLCC, KROCC, OR, PWRC,
Z-RMSE, plus per-band MAE, non-mono q-step rate, and n. SROCC alone
per band is **NOT** sufficient and produces misleading rankings —
especially at low n where the SROCC CI exceeds 0.3.

For each (model, dataset) eval the harness MUST emit per band:
1. **SROCC** (Spearman rank, with 95% bootstrap CI)
2. **PLCC** (Pearson on calibrated outputs)
3. **KROCC** (Kendall-τ)
4. **OR** (outlier ratio — predictions outside ±2σ of subjective)
5. **PWRC** (Pearson-weighted rank correlation, Mohammadi 2025)
6. **Z-RMSE** (σ-normalized RMSE; per-stimulus σ on AIC-3/AIC-4/CID22
   where bootstrap σ is available, corpus-wide σ elsewhere)
7. **MAE** (mean absolute prediction error in score units)
8. **Non-monotonic q-step rate** (adjacent-q reversals within each
   curve, segmented by lower-q band)
9. **n** (sample count) — flag bands with n < 30 as "noisy
   estimate" (CI widths exceed ±0.3 SROCC; rankings between bakes
   are not statistically distinguishable at this n)

**Why the full panel per band, not SROCC-only**: SROCC is
calibration-invariant and bounded [-1, 1]. PWRC + Z-RMSE capture
different failure modes — PWRC weights important pairs higher,
Z-RMSE measures absolute σ-normalized error. A bake can win SROCC
on a band while losing Z-RMSE (badly miscalibrated to scale) or
losing PWRC (wins on tied/duplicate pairs that don't matter). The
joint pattern is the real signal.

**Current gap**: `dataset_metric_baseline` emits the full panel at
the aggregate but only SROCC + CI per band. Extending per-band
emission to the full panel is a queued fix (~3 hr) — DO this before
the next ship-grade comparison.

**Why 10 not 4**: aggregate SROCC hides band-specific failures.
4 bands hide them less, but still merge product-distinct regions
(e.g. B2: 65..90 covers both "subtle artifacts" and "near-lossless"
which behave very differently). 10 bands × width-10 surfaces
boundary effects at every 10-zq step — the granularity at which
codec consumers actually tune.

**Why this matters**: zensim is a user-facing dial. A user typing
"give me zensim 70" lives in 10-band B7. A user typing "zensim 55"
lives in B5. If the metric is well-calibrated at B9 but breaks at
B5, low-q encodes get the wrong settings.

Until the harness emits the 10-band grid, treat any "champion"
claim as provisional. Aggregate numbers and the 4-band CID22 cuts
are pipeline-health checks, not release gates.

**Site requirement**: the interactive comparison site at
<https://imazen.github.io/zensimx/> MUST render the 10-band table
alongside the legacy 4-band table for every (corpus, X, Y)
selection.

## V_20 input-shaping + multi-bake runtime — learnings (added 2026-05-15)

After running V_20 input-shaping (feature transforms applied
pre-scaler), V_20b distortion manifold (Su 2023 contrastive
pre-train), D1 (3-way concat with transforms), D2 (V_18 + V_20 IS
runtime ensemble), and D3 (tighter transform subset) — these
learnings shape future V_X experiments. Read before designing new
training experiments or ship candidates.

### Pearson screen is necessary, not sufficient

The greedy correlation screen at
`scripts/v_next/v0_20_feature_transform_greedy_screen.py` finds
features where |Pearson(transform(feat), MOS)| beats Identity by
some threshold. The screen reliably filters which features WORTH
RUNNING MLP variants on (cuts ~1600 brute-force cells to ~100). But
**Pearson lift on raw features does NOT guarantee MLP-training
SROCC lift**. The MLP already does non-linear absorption of feature
non-linearities; the transform's benefit is at the standardize
step, not the MLP's expressivity. Train + eval to confirm.

### Training-safety gates on the screen (mandatory)

`log` requires `min(feat) > 0`; `log1p` requires `min(feat) > -1`.
Without these gates the screen accepts transforms that produce NaN
on real training data (the screen drops NaN rows in Pearson, but
the trainer doesn't drop bad rows — NaN cascades to standardize +
gradient → NaN loss at epoch 0). Already wired into the screen
script; preserve when adding new transforms.

### V_20 input-shaping is a B3 specialist, not an aggregate win

V_20 IS single-MLP closes CID22 B3 [30, 40) gap by **+0.129 SROCC**
(0.0246 → 0.1534, also passing fast-ssim2's 0.1335 floor) but
**costs −0.014 CID22 aggregate** (B4–B8 each lose 0.02–0.06). Same
shape as V_20a multi-output. The mid-band regression is intrinsic
to the feature shaping, not removable by ensemble averaging.

For a clean ship: pair V_20 IS with V_18 ship via the multi-bake
runtime (PreviewV0_4). **D2 α=0.4 raw-space ≡ α=0.7 z-norm** —
delivers +0.080 CID22 B3 lift at −0.008 aggregate. **But** the
hard `score.clamp(0.0, 100.0)` in `apply_mlp_scoring` flattens
V_20+ predictions on heavy-distortion images (TID B0/B1 → score=0
ties → SROCC=0). Fix when you have time: soft saturation or
per-bake recalibration so V_20 IS raw stays in [0, 100].

### V_20b distortion manifold — falsified for CID22 priority

Su 2023's contrastive pre-train + fine-tune approach **wins every
metric on KADID + TID** (SROCC +0.023 / +0.027, Z-RMSE 23–31%
reduction) — the mechanism IS learning useful structure on
training-side data. But **loses every metric on held-out CID22**
(SROCC −0.027, B3 [30, 40) no lift). This is the FRIQUEE 2017
caveat materializing: synth pre-train → authentic-distortion
transfer fails. Do NOT pursue further Su-2023-style mechanisms on
the synth corpus without first solving the CID22-transfer problem.
**The transfer fix must NOT involve CID22 human MOS in training**
(see "CID22 is VALIDATION-ONLY" section above). Allowed
directions: domain adaptation with unlabeled CID22 image pairs,
metric-anchored fine-tuning using CID22 training-only-subset
ssim2/CVVDP scores, additional authentic-distortion corpora with
their own held-out CID22-style validation gates.

### D3 tighter transforms — falsified vs V_20 IS

Cutting the transform set from 98 (lift ≥ 0.05) to 60 (lift ≥ 0.10)
gave back HALF the B3 lift (+0.055 vs +0.129) WITHOUT recovering
the mid-band cost. The "borderline" 0.05–0.10 lift transforms ARE
driving most of the B3 gain. Don't trim aggressively on the lift
threshold — let the trainer absorb the noise.

### D1 3-way concat ≈ V_20 IS single MLP

Training cycle-14 TV-regularized components with V_20 transforms +
concat at V_18's 0.65/0.30/0.05 mix gave SAME CID22 SROCC (0.8794)
as single-MLP V_20 IS. The 3-way concat does NOT stabilize the
mid-band regression. Don't run the full concat recipe just because
V_18 ship used it — the value-add depends on TV's effect on
specific bands, which input-shaping bypasses.

### Multi-bake runtime — when to use

**Use PreviewV0_4 (V_18 ship + V_20 IS @ α=0.4)** when you need
CID22 B3 lift and can tolerate −0.008 aggregate. The Zensim::compute
runtime forwards both bakes (V_18 via plain `predict`, V_20 IS via
`predict_transformed`) and mixes RAW outputs linearly before
clamp. Cost: ~2× forward pass time per call vs PreviewV0_3.

**Do NOT use** for very-low-quality input regimes (TID-style B0/B1)
until the runtime clamp issue is fixed — current implementation
degenerates SROCC there.

### Bake-metadata propagation across derived bakes (CRITICAL)

**Every tool that produces a derived ZNPR v3 bake from input bakes
MUST propagate `zentrain.feature_transforms` and
`zentrain.feature_transform_params` metadata**, or runtime
predictions will silently degrade (raw features fed into a network
trained on transformed features → wrong predictions, possibly
NaN). Caught and fixed in `concat_three_way` (commit `6ad46950`)
after D1 first eval showed catastrophic regression (CID22 0.5700
vs V_18 base 0.8880).

Tools that touch bake bytes MUST be audited for this:
- `concat_three_way` ✓ (fixed 2026-05-15)
- `affine_calibrate` — preserves metadata via byte-rewrite of final
  layer only; ✓ by construction
- `quant_compare` — converts F32 → I8; ⚠ verify metadata
  preservation before any future use
- Any future `bake_optimized` / `zerobias_rebake` consumer in this
  repo — ⚠ verify

### Runtime forward path MUST dispatch to `predict_transformed`

Both `dataset_metric_baseline`, `ensemble_mix`, and zensim's own
`apply_mlp_scoring` were originally calling `Predictor::predict`
unconditionally. For V_20+ bakes with feature_transforms metadata,
this feeds raw features into a transform-expecting network and
produces garbage (often NaN). All three are fixed; new tools that
forward bake bytes through Predictor MUST check
`model.has_nontrivial_feature_transforms()` and dispatch to
`predict_transformed` when true.

### Affine-calibration policy (don't double-apply)

V_18 ship's affine α=28.0366, β=−5.0738 is a **distance→score**
transform (β is negative because raw is distance). Bakes whose raw
output is already approximately score-shaped (e.g., V_20 IS,
trained directly against MOS targets) DO NOT need this affine.
Applying it inverts and shifts predictions away from [0, 100]; the
`score.clamp(0, 100)` then flattens many predictions to 0 →
degenerate ranking.

Per-bake calibration check: if the trainer's loss was RankNet on
distance, apply V_18's affine. If the trainer's loss was MSE on
MOS (or similar score-shaped target), the bake is already
calibrated — no affine needed.

### Soft-clamp the multi-bake output (TODO)

The current `score.clamp(0.0, 100.0)` at the end of
`apply_mlp_scoring` is a hard boundary. When the multi-bake mix
goes below 0 or above 100 (V_20+ bakes extrapolating outside their
training distribution), the clamp creates ties → SROCC=0 on
affected bands. Replace with a soft saturation
(e.g., `100 / (1 + exp(-(raw - 50) / 10))`) or per-bake
recalibration so raw outputs stay within [0, 100].

## V39 ship + dial/spline/anchor learnings (added 2026-05-25)

The 2026-05-25 evening session shipped **V39** as `PreviewV0_3`
(`zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`). It beats
the prior V0_3 (v_tuner_v11) on **5 of 6** held-out corpora across the
full panel (SROCC/PLCC/Z-RMSE/DS-AUC) AND the G1 dial — CID22, KADIK,
TID, KonJND, AIC-3 all improve. **AIC-4 (n=300) is the exception**:
V0_3 wins its SROCC 0.9284 vs V39 0.9051 (V39 still wins AIC-4 DS-AUC).
So "universally better" is accurate for the 5 compression-focused
holdouts, NOT literally all 6 — don't overclaim it.
**Goal status (CODEC_TARGET_GOALS.md):** V39 PASSES G1 (dial 1.00) +
G7 (CID22 0.879 ≥ 0.85), soft-passes G8 (AIC-3 Z-RMSE 0.584) + G9
(DS-AUC 0.739), but **FAILS G5** (KonJND HF-rank 0.42 < 0.70 floor) —
the structural learning-metric HF-failure zone the doc itself flags;
full goal achievement needs the unstarted HF-corpus acquisition.
Full lineage + numbers: `benchmarks/v5_vs_v03_comparison_2026-05-25.md`.
Hard-won learnings (re-read before training or evaluating a
codec-target bake):

1. **SROCC-only verdicts hide a BROKEN DIAL — always run the
   `bake_verdict` goals scorecard (G1 dynamic range).** Bakes trained
   without an output calibration spline collapse the dial to a
   near-constant (~65 score, G1=0.00) even at SROCC 0.88. A codec
   binary-searching for "score=30" can never find it. This was
   invisible under SROCC-only comparison; the scorecard (added this
   session, auto-runs after every train) catches it. G1 is goal #1
   for a reason.
2. **The output calibration spline is MANDATORY for a codec-target
   bake, and SROCC is rank-invariant under it.** A monotone PCHIP
   spline stretches a compressed-but-well-ranked output to the full
   [0,100] dial WITHOUT changing rank order. So the recipe is: train
   for rank (V32-style hybrid MSE+RankNet), then fit a spline from a
   multi-band anchor. That's V39.
3. **The anchor is for SPLINE FITTING ONLY — use `--anchor-loss-weight
   ≤ 0.01`.** V37 used 0.5 and the anchor MSE fought RankNet, collapsing
   held-out CID22 SROCC 0.88→0.55. Any meaningful anchor weight
   destroys rank. The anchor parquet needs a per-row `target_score`
   column spanning 0–100 (NOT the constant `--anchor-target-score`)
   so the spline gets multi-band knots.
4. **Normalize all group `human_score` targets to a common scale.**
   5-group MSE-only training diverges after epoch 0 when scales differ
   (cid22_train raw MCOS [3-94], konjnd [-66,96], others [0,1]). This
   was the "training regression" that blocked reproduction for hours.
5. **CVVDP-emulator training is a DEAD END** (V41). Training toward
   CVVDP scores gives WORSE human-MOS (CID22 0.66 vs 0.88). Emulating
   CVVDP's OUTPUT ≠ having its CSF mechanism.
6. **The AIC-3 "0.80 vs 0.96 CVVDP gap" is a measurement artifact, NOT
   a feature limit.** CVVDP's 0.96 was a 5-image subset; on the full
   600-pair/10-ref set CVVDP is 0.79 pooled / 0.93 per-ref, and our
   bake scores 0.9475 per-ref — ABOVE raw CVVDP. Mixing real CVVDP in
   as an input feature adds +0.004 (noise). **Do NOT invest in
   CSF-approximating features to "close the CVVDP gap"** — the residual
   pooled number is a cross-ref absolute-scale-calibration problem (the
   dial), not missing feature information. Spike:
   `benchmarks/aic3_cvvdp_feature_spike_2026-05-25.md`. This SUPERSEDES
   any earlier "CVVDP gap requires CSF feature engineering" claim.
7. **dynamic-range-floor regularizer overshoots** (V40, weight 0.3 →
   output >100, saturated top). Promising for forcing spread but needs
   careful tuning + the eq-pool substrate; the cross-codec-eq aux loss
   is "not yet wired" for 2-layer/skip mode.
8. **DATA BUG (fixed, pending promotion): kadid/tid `iwssim` =
   human_score copy (target leak), `ssim2_gpu` = ref-vs-ref misjoin.**
   Shipped bakes SAFE (trained on `human_score`); any multi-target
   `iwssim` bake on kadid/tid is INVALID. Corrected
   `*_fixed_2026-05-25.parquet` siblings written; build script now has
   `_validate_metric_columns()` guard. Detail:
   `benchmarks/DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`.
9. **G5 (KonJND HF ≥0.70) is a CHARACTERIZED Pareto limit — falsified
   across two architectures (2026-05-26).** (a) Single 2-layer MLP +
   konjnd-aggregation head (now wired + gradient-verified): clears 0.70
   at agg-weight≥0.05 but craters CID22/KADIK/TID/AIC-3 — Pareto
   tension (`v42_konjnd_agg_2layer_G5_sweep`). (b) 2-bake regime-routed
   ensemble {V39 + HF-specialist}: best KonJND 0.7014 but CID22 −0.028
   — CID22's near-lossless tail OVERLAPS the KonJND regime in feature
   space, so the router can't separate them and the specialist can't
   rank CID22 (`v43_g5_regime_routed_ensemble`). **Don't retry single-MLP
   weight tuning or naive ensembles for G5** — both are dead. Closing it
   needs a better HF feature REPRESENTATION (not data: KonJND is already
   in training; not CSF: AIC-3 spike falsified it). Also: raw bake output
   is a ~0.2-wide flat band (the dial lives in the spline) — so any
   soft/continuous blend of two bakes scrambles rank; only a HARD
   per-pair switch preserves it.
10. **"Universally better than V0_3" is FALSE — V39 wins 5/6, V0_3
   significantly wins AIC-4** (paired bootstrap Δ+0.023, p=0.001). AIC-4
   is HOLDOUT-ONLY — do NOT recipe-search to win it (holdout-fishing).
   Use the FULL panel + paired significance for any A/B verdict, never a
   raw SROCC point estimate (a 0.023 gap at n=300 looked like noise by
   the single-SROCC CI ±0.063 but is real under the paired test).

## Canonical training/validation corpora (2026-05-18, R2-verified 2026-05-25)

**This is the single canonical training/validation parquet set. All
new V_X training recipes use these. Old per-experiment parquet dirs
in `/mnt/v/zen/zensim-training/2026-05-{15,17,18}-*/` are FROZEN;
rebuilds and new joins go here.**

> **CURRENT canonical (2026-05-25):** the working set is
> `canonical-2026-05-21/`. The 2026-05-18 layout below is preserved
> for archaeology; the 2026-05-21 set is the one v11 was trained on,
> mirrored to R2 at `s3://zentrain/canonical-2026-05-21/` (26 objects,
> 1.32 GB, all sizes byte-equivalent to local as of 2026-05-25). The
> 5 v11 training inputs (`safesyn.parquet`, `cid22_train.parquet`,
> `kadid.parquet`, `tid.parquet`, `konjnd-dense.parquet`) live under
> `canonical-2026-05-21/train/`. The full v11 reproduction kit (bake
> + recipe + commit-pinned input sha256s) is at
> `s3://zentrain/v11-reproduction-kit-2026-05-25/`. All shipped bakes
> (41 files, 4.6 MB) are mirrored at `s3://zentrain/bakes-2026-05-25/`
> for cold-start independence from the zensim git repo.
>
> **⚠ 2026-07-15 DATA FIX (DATASET_HISTORY §3.18):** `train/{kadid,tid}.parquet`
> had the V39-#8 corruption (`iwssim` = a byte-identical `human_score` copy;
> `ssim2_gpu` = ref-vs-ref misjoin, pinned ~100). RESOLVED by promoting the
> verified `*_fixed_2026-05-25` siblings (`scripts/canonical_corpus/promote_fixed_kadid_tid_2026-07-15.py`):
> real iwssim (SROCC-vs-MOS +0.850/+0.779) + real ssim2 ([−367,100]/[−96,90],
> SROCC +0.813/+0.846); **all 372 features + human_score/cvvdp_*/pjnd_target
> byte-identical** (zero feature drift, so v11 — trained on `human_score` — is
> unaffected). New sha256: kadid `38735c46…`, tid `efff176a…`. Corrupt originals
> preserved at `<c>.CORRUPT-v39bug.pre-2026-07-15.bak.parquet` + R2 `_archive/` +
> Tower. R2 + Tower re-synced. **NOTE:** the archival `canonical-2026-05-18/train/`
> kadid/tid were NOT promoted (still corrupt; `*_fixed_2026-05-25` siblings sit
> there if ever needed) — that layout is archaeology-only.

### Local + R2 locations

- **Local**: `/mnt/v/zen/zensim-training/canonical-2026-05-18/`
- **R2**: `s3://zentrain/canonical-training-2026-05-18/` (bucket
  `zentrain`, endpoint
  `https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`)
- **Manifest**: `_MANIFEST.json` at the root of both, with per-file
  row count, byte size, sha256, target-column list, and per-parquet
  metadata block (canonical_corpus, canonical_role, source_paths,
  schema_version, training_use, schema_note).
- **Total size**: 766.5 MiB (under the 5 GB budget; safesyn dominates
  at 563 MiB, score sidecars are ~4 MiB).

### Layout

```
canonical-2026-05-18/
├── _MANIFEST.json                              13 parquet entries + sha256 + per-file metadata
├── train/
│   ├── safesyn.parquet                         196,086 rows × 372 features (synthetic-safe, CID22-leak-purged)
│   ├── kadid.parquet                            10,125 rows × 372 features (KADID-10k DMOS)
│   ├── tid.parquet                               3,000 rows × 372 features (TID2013 MOS)
│   ├── konjnd-dense.parquet                     20,160 rows × 372 features (KonJND-1k densified, +pjnd_target)
│   └── cvvdp_iwssim_LARGE.parquet               73,300 rows × 300 features (iwssim∩cvvdp overlap; NO IW-pool block)
├── val/                                        HOLDOUT-ONLY. Never train on val/*.parquet's human_score.
│   ├── cid22.parquet                             4,292 rows × 372 features (MCOS, gold-standard)
│   ├── kadid.parquet                            10,125 rows × 372 features (DMOS, integrity audit alongside train)
│   ├── tid.parquet                               3,000 rows × 372 features (MOS, integrity audit alongside train)
│   ├── konjnd.parquet                            1,008 rows × 372 features (PJND anchor, original)
│   └── aic3.parquet                                600 rows × 372 features (AIC-3 CTC JND)
└── scores/                                     Raw-score sidecars. Preserved for any future re-mix.
    ├── cvvdp_imazen_v0_0_1.parquet            1,169,500 rows (image_path, codec, q, knob_tuple_json, cvvdp_imazen_v0_0_1)
    ├── iwssim_imazen.parquet                     75,300 rows (same key + iwssim_imazen_v0_0_1)
    └── ssim2_imazen.parquet                      55,000 rows (same key + ssim2_gpu)
```

### Canonical schema (training + validation parquets)

⚠ **2026-05-28 column rename**: `konjnd-dense-norm.parquet` and
`konjnd-dense.parquet` gained `active_mix_norm` / `active_mix_raw`
columns as aliases of their respective `human_score` columns. The
`human_score` column on these parquets carries the per-pair active mix
output — NOT the per-ref mean PJND. val/konjnd's `human_score` column
DOES carry the mean PJND (and equals train's `pjnd_target` per ref).
New recipes should use:
- `active_mix_norm` (in train, [0,1]) for per-pair training on active mix
- `pjnd_target` (in train) when targeting val/konjnd's `human_score` directly
Old recipes pinning `--target-column human_score` are PRESERVED via
the legacy `human_score` column on both train parquets.

Every train/* and val/* parquet uses the same column ordering so
trainers can switch corpus paths without touching schema-handling
code:

```
ref_basename : string                source image identifier
human_score  : float64               per-corpus native anchor
  - safesyn:       synthetic ssim2-derived target
  - kadid:         DMOS (1-5, lower=better)
  - tid:           MOS (0-9, higher=better)
  - konjnd-dense:  the active mix output (NOT PJND — use pjnd_target column for PJND).
                   ⚠ 2026-05-28: ALSO available as `active_mix_norm` (in
                   konjnd-dense-norm) / `active_mix_raw` (in konjnd-dense), added
                   as clearer aliases of `human_score` to disambiguate from
                   val/konjnd's `human_score` = mean PJND. New recipes should use
                   `active_mix_norm` / `active_mix_raw` for explicit intent;
                   `human_score` retained for backward compat with shipped V_X recipes.
  - cvvdp_LARGE:   ssim2-derived training anchor
  - val/cid22:     MCOS / 100 (validation-only)
  - val/kadid:     DMOS
  - val/tid:       MOS
  - val/konjnd:    mean PJND threshold (= train's `pjnd_target` column for same ref)
  - val/aic3:      score.jnd (signed JND units, AIC-3 CTC)
active_mix_norm                       : f64 (konjnd-dense-norm ONLY; alias of
                                         `human_score`, added 2026-05-28). Per-pair
                                         active mix normalized to [0, 1].
active_mix_raw                        : f64 (konjnd-dense ONLY; alias of
                                         `human_score`, added 2026-05-28). Per-pair
                                         active mix on the raw [-66, 96] scale.
cvvdp_score, cvvdp_log_norm           : f64 (null if not available)
iwssim, iwssim_log_norm                : f64 (null if not available)
ssim2_gpu, ssim2_log_norm              : f64 (null if not available)
pjnd_target                            : f64 (KonJND PJND threshold; null except for konjnd-dense).
                                         Per-ref constant (replicated across all 20
                                         distortion levels per ref). EQUALS val/konjnd's
                                         `human_score` for the same ref.
mix_cv25_iw75 … mix_cv75_iw25          : 10 cvvdp×iwssim mix variants
mix_cv33_iw33_sm33                     : 3-way mix with ssim2 (only kadid + tid have non-null)
mix_target                             : alias for active --target-column at consumer's choice
f0..f371                               : 372 features (basic + peak + masked + IW-pool blocks)
                                         LARGE: f0..f299 only (300 features, no IW-pool block)
```

Missing target columns are explicit nulls, not silently dropped —
every parquet has the full superset of target columns so the trainer
sees a stable schema across corpora.

### Per-canonical-parquet detail

| Canonical path | Rows | Feature cols | Non-null targets | Bytes (zstd-15) | sha256 (prefix) |
|---|--:|--:|---|--:|---|
| `train/safesyn.parquet` | 196,086 | 372 | cvvdp_score, cvvdp_log_norm, iwssim, iwssim_log_norm, ssim2_gpu, ssim2_log_norm, mix_cv25..cv75 grid (no `mix_cv33_iw33_sm33`) | 590.5 MB | `1ee0565fb6cb` |
| `train/kadid.parquet` | 10,125 | 372 | all of the above + `mix_cv33_iw33_sm33` | 31.5 MB | `037685df82f5` |
| `train/tid.parquet` | 3,000 | 372 | all of the above + `mix_cv33_iw33_sm33` | 9.4 MB | `9d6b6b78b987` |
| `train/konjnd-dense.parquet` | 20,160 | 372 | `human_score` + `active_mix_raw` (2026-05-28 alias) + `pjnd_target` + mix_cv25..cv75 grid (cvvdp/iwssim/ssim2 are all null) | 67.3 MB | `803f1f571032` |
| `train/konjnd-dense-norm.parquet` | 20,160 | 372 | `human_score` + `active_mix_norm` (2026-05-28 alias, both [0,1]) + `pjnd_target` + mix_cv25..cv75 grid | 67.3 MB | `49b65c4890fb` |
| `train/cvvdp_iwssim_LARGE.parquet` | 73,300 | **300** | cvvdp_score, cvvdp_log_norm, iwssim, iwssim_log_norm, mix_cv40_iw60 (no ssim2, no full mix grid) | 43.7 MB | `289d45cc3169` |
| `val/cid22.parquet` | 4,292 | 372 | `human_score` only (MCOS / 100) | 13.0 MB | `6eea08253fa2` |
| `val/kadid.parquet` | 10,125 | 372 | `human_score` only (DMOS) | 30.1 MB | `04a8d5b0b5f1` |
| `val/tid.parquet` | 3,000 | 372 | `human_score` only (MOS) | 9.0 MB | `c4a4bf0521da` |
| `val/konjnd.parquet` | 1,008 | 372 | `human_score` only (mean PJND) | 3.1 MB | `3e999a372577` |
| `val/aic3.parquet` | 600 | 372 | `human_score` only (score.jnd) | 1.9 MB | `a93aab6d2796` |
| `scores/cvvdp_imazen_v0_0_1.parquet` | 1,169,500 | — | raw `cvvdp_imazen_v0_0_1` per (image, codec, q, knob) | 3.4 MB | `cfa285e198a8` |
| `scores/iwssim_imazen.parquet` | 75,300 | — | raw `iwssim_imazen_v0_0_1` | 0.8 MB | `9c29fc8d3dc0` |
| `scores/ssim2_imazen.parquet` | 55,000 | — | raw `ssim2_gpu` | 0.3 MB | `76bb7731c33b` |

`aic4` is not yet in the canonical set — features for AIC-4 sample
corpus are not extracted (only raw images at
`/mnt/v/dataset/aic4_sample/`). To add: run
`target/release/examples/extract_features_372col --corpus aic4 …`
and add a `build_val_aic4()` step in
`scripts/canonical_corpus/build_canonical_parquets.py`.

### Migration table (old → canonical)

| Old path (DEPRECATED, not deleted) | Canonical replacement | Why deprecated |
|---|---|---|
| `2026-05-18-v24/safesyn_4target_372col.parquet` | `canonical-2026-05-18/train/safesyn.parquet` | Joined into canonical with full target superset |
| `2026-05-18-v24/kadid_4target_372col.parquet` | `canonical-2026-05-18/train/kadid.parquet` | same |
| `2026-05-18-v24/tid_4target_372col.parquet` | `canonical-2026-05-18/train/tid.parquet` | same |
| `2026-05-18-v24/konjnd_features_mix_targets_372col.parquet` | use konjnd-dense for training, val/konjnd for eval | per-image-anchor subset of the dense set |
| `2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_372col.parquet` + `*_pjndtarget_300col.parquet` | `canonical-2026-05-18/train/konjnd-dense.parquet` | joined; `pjnd_target` lifted from pjndtarget's `human_score` |
| `2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet` | `canonical-2026-05-18/train/cvvdp_iwssim_LARGE.parquet` | schema-aligned (300-feature limitation preserved) |
| `2026-05-17-cvvdp-merged-trainer/{safesyn,kadid,tid,konjnd}_mix_300col.parquet` | superseded by `train/*.parquet` (372-feat instead of 300) | 372-col variant is canonical |
| `2026-05-17-cvvdp/*_features_*_372col.parquet` | superseded by canonical per-corpus train + val | upstream intermediate |
| `2026-05-17-cvvdp/multicodec_cvvdp_300col*.parquet`, `v15r_zenjpeg_cvvdp_*.parquet` | preserved as `2026-05-17-cvvdp-merged/unified_*.parquet` — picker work, not zensim metric — OUT of canonical scope | scope |
| `2026-05-17-cvvdp-merged/unified_*_cvvdp.parquet` | UNCHANGED — these are picker-work per-codec cuts, not trainer inputs | scope (picker, not zensim metric) |
| `2026-05-18-extfeat/*.parquet` (343-col extended-features experiment) | NOT in canonical set; experimental 343-col feature variant from EX-4. Promote only if EX-4 results ship the 343-col schema. | experimental scope |
| `2026-05-18-ssim2/ssim2_imazen_consolidated.parquet` | `canonical-2026-05-18/scores/ssim2_imazen.parquet` | identical content, canonical name |
| `2026-05-15-cvvdp-r2/cvvdp_imazen_consolidated.parquet` | `canonical-2026-05-18/scores/cvvdp_imazen_v0_0_1.parquet` | identical content, canonical name |
| `2026-05-15-cvvdp-r2/iwssim_imazen_consolidated.parquet` | `canonical-2026-05-18/scores/iwssim_imazen.parquet` | identical content, canonical name |
| `2026-05-15-full-features/*_features_372col_2026-05-15.parquet` | `canonical-2026-05-18/val/*.parquet` | per-corpus validation feature extraction is canonical here |
| `2026-05-18-v24-alpha/*.parquet` (α-sweep variants) | NOT in canonical set; α is a hyperparam, not a corpus | experimental scope |
| `2026-05-18-ssim2/{kadid,tid,large}_*target*.parquet` | superseded by `2026-05-18-v24/` then `canonical-2026-05-18/train/` | upstream intermediate |
| `2026-05-15-cvvdp-r2/safesyn_v15r_zenjpeg_cvvdp_300col.parquet` | superseded by `train/safesyn.parquet` (372 features + 4 targets) | older 300-col cut |
| `2026-05-14-clean/*.csv` | superseded by `train/safesyn.parquet` (post-2026-05-12 perceptual-overlap purge) | older clean cut |
| `2026-05-16/*.parquet` (safesyn IW-SSIM backfill intermediate) | NOT in canonical set; intermediate sidecar to the 2026-05-17 join | upstream intermediate |
| `2026-05-07/unified/*.parquet` | NOT in canonical set; older 351-col raw v15-sweep parquets. Keep for forensic. | older feature schema |

**Old paths are NOT deleted.** Many in-flight runners (V_22 retrain
finishing on `2026-05-18-v24/`, EX-4 343-feat eval on
`2026-05-18-extfeat/`, α-sweep on `2026-05-18-v24-alpha/`, konjnd-dense
agent on `2026-05-18-konjnd-dense/`) may still reference them. Cleanup
of the deprecated dirs is a future-session job — confirm zero in-flight
references first.

### Build / re-build script

`scripts/canonical_corpus/build_canonical_parquets.py` consolidates
sources into the canonical layout. Re-run when a new training corpus
or target column is added; bump the date suffix (`canonical-2026-MM-DD/`)
and update this section. The script's docstring is the source-of-truth
on schema and source paths.

### Operational rules

1. **Train recipes** load from `canonical-2026-05-18/train/*.parquet`
   only. Pass `--target-column NAME` to select among the available
   target columns. Use the per-parquet "non-null targets" column in
   the table above to know which columns have real data — null
   target columns must NOT be silently trained against (the trainer
   should fail loudly, not learn to predict nulls).
2. **Val recipes** load from `canonical-2026-05-18/val/*.parquet`
   only — never load `val/*.parquet`'s `human_score` as a training
   target. `bake_verdict` already loads from
   `2026-05-15-full-features/` by default; queued: switch its
   `--features-root` default to canonical.
3. **Schema evolution**: adding a new target column means rebuilding
   the canonical set + bumping the date. Don't sneak a target into
   `2026-05-18-v24/` after-the-fact.
4. **R2 sync**: re-upload via
   `aws s3 sync /mnt/v/zen/zensim-training/canonical-YYYY-MM-DD/ s3://zentrain/canonical-training-YYYY-MM-DD/ --endpoint-url https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`
   (with `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` set from
   `~/.config/cloudflare/r2-credentials`).

## Principled experiment workflow for V_X bakes (added 2026-05-15)

This section is the **methodology** version of the V_20 learnings
above — the ordered checklist for doing the work, not the catalog
of what we learned. Follow these steps every time a new bake / runtime
experiment is opened. Skipping steps creates the failure modes
documented in the V_20 learnings section (silent metadata loss,
double-applied affine, hard-clamp ties, untrustworthy SROCC-only
reports). Each step ends in a tangible artifact — if there is no
artifact, the step did not happen.

### Step 1 — Write the hypothesis and falsification ($\le$ 5 minutes)

Before opening a trainer, write four sentences:

1. **Hypothesis**: "Adding feature transform set X should lift
   CID22 SROCC by Δ ≥ 0.005 (or another concrete metric), justified
   by Pearson lift Δr ≥ 0.05 on the screen for ≥ N features."
2. **Falsification**: "If CID22 aggregate SROCC drops or stays flat
   across 3 seeds, the hypothesis is dead and we move on."
3. **Cost ceiling**: "Allowed budget: 1× pretrain + K× fine-tune.
   If we hit that budget without lift, abandon."
4. **Ship form**: "If the hypothesis succeeds, we ship as
   PreviewV0_N (single-bake) OR PreviewV0_(N+1) (multi-bake) with
   metadata propagated and runtime dispatch correct."

Artifact: a 4-line scratch note at the top of the experiment log
(`benchmarks/v0_X_<name>_<date>.md`). If you cannot write these
four lines, you are not ready to train.

### Step 2 — Decide the reporting panel upfront

Mandatory stat panel = Mohammadi 2025 full set (SROCC + PLCC +
KROCC + OR + PWRC + Z-RMSE), at **both aggregate and 10-band level**
(B0..B9). SROCC alone is the "single most misleading practice" —
don't lapse back to it just because the experiment is preliminary.

Held-out discipline:

- Train on synth + KADID + TID. Inspect aggregate stats and
  KADID/TID per-band signal during iteration.
- **CID22 is opened LAST**, once per experiment, at decision time.
  Inspecting CID22 mid-experiment leaks information into the
  hyperparameter choice.

Decide before the first train run which corpora produce which signal.

Artifact: a table at the top of the experiment log listing
"corpus / role / when inspected". Stick to it.

### Step 3 — Seed=1 first as cheap signal

Run a single seed=1 fine-tune before sweeping. The cost is ~20% of
a 5-seed sweep and most negative results are visible at seed=1.

Decision tree:

- **Seed=1 wins held-out signal corpus by ≥ the falsification
  threshold** → sweep 5 seeds for CI. Then open CID22.
- **Seed=1 flat or negative** → hypothesis probably dead.
  Document the negative result and stop. Do NOT sweep 5 seeds
  hoping seed=2 wins — that is p-hacking.
- **Seed=1 mixed** → before sweeping, diagnose mechanism (which
  per-band moved, which feature transforms contributed). Decide
  whether the mixed signal is worth a 5× compute spend.

Artifact: seed=1 eval log committed to `benchmarks/`, with the
decision recorded in the experiment log.

### Step 4 — Diagnose bake shape before any calibration

Every bake's raw output is either **distance-shaped** (training
target was distance, low = high quality, range often 0..30) OR
**score-shaped** (training target was MOS, high = high quality,
range often 0..100). The two cannot be mixed without per-bake
affine handling.

Diagnostic: run the bake on 100 random pairs from the safe-synthetic
training set, plot `bake_output vs MOS`. If slope ≈ -1, distance-shaped.
If slope ≈ +1, score-shaped. If slope is near zero or sign-ambiguous,
something is wrong with the bake (most likely missing feature_transforms
metadata — see Step 6).

Affine policy:

- Distance-shaped bake → apply
  `scripts/v_next/affine_calibrate_znpr_v2.py` to fit α, β:
  `score = α + β · distance`, β should be negative.
- Score-shaped bake → **NO affine**. The bake already maps to
  approximately 0..100; an affine on top inverts the response.
- Multi-bake runtime mixing both shapes → each sub-bake gets its
  OWN calibration. Mixing is in score space, not raw space. This
  is currently NOT implemented in PreviewV0_4 and is the root
  cause of the V_20_4 TID B0/B1 SROCC=0 issue.

Artifact: a `shape: distance | score` line in the bake's
methodology doc. Future tools read this to decide affine handling.

### Step 5 — Use the JSON pipeline; do not write ad-hoc serializers

For every new bake-producing tool (concat, distill, ensemble-collapse,
zerobias-rebake), emit `BakeRequestJson` and shell out to
`zenpredict-bake <input.json> <output.bin>`. The template is
`scripts/v_next/v0_20b/bake_znpr_v3.py`.

Anti-pattern: writing a Python function that emits ZNPR v3 bytes
directly. The wire format has alignment, section ordering, and
header invariants that only the Rust serializer enforces. Drift
between ad-hoc emitters and `zenpredict-bake` ships wrong-shape
bakes that load but score garbage.

Artifact: the bake-producing script's main loop ends in a
`subprocess.run(["zenpredict-bake", "in.json", "out.bin"])` call,
or equivalent. No `struct.pack` in the call graph.

### Step 6 — Propagate metadata across every derived-bake tool

For every tool that takes input bakes and produces an output bake,
audit the metadata pipeline. The fields that MUST propagate:

- `zentrain.feature_transforms` (Vec<TransformOp>)
- `zentrain.feature_transform_params` (Vec<f32>)
- Per-bake calibration (α, β) where present
- `output_specs[]`, `sparse_overrides[]`, `discrete_sets[]`

Rule for multi-source derived bakes:

- **Single-source derived** (e.g., affine rebake) → copy source's
  metadata verbatim. Audit that the byte-rewriter touches only
  the layer being modified.
- **Multi-source concat** → all inputs MUST agree on
  feature_transforms. Assert this and fail loudly if not — heterogeneous
  feature shaping in a concat is undefined behavior.
- **Multi-source ensemble (runtime mix)** → each sub-bake retains
  its own metadata; the runtime dispatches per-sub-bake (Step 7).

Smoke test: every derived-bake tool MUST ship a test that produces
a bake from synthetic inputs and runs it through `Predictor::predict`
or `predict_transformed`, asserting the output is non-NaN AND
within the expected shape (distance vs score). This test caught
the V_20 concat regression in CI.

Artifact: a `tests/` test in the bake-producing tool's crate.
Without it the tool is not landable.

### Step 7 — Runtime forward path: predict_transformed dispatch

Every eval harness, runtime path, and validation tool that consumes a
bake MUST contain:

```rust
let raw = if model.has_nontrivial_feature_transforms() {
    model.predict_transformed(&features)
} else {
    model.predict(&features)
};
```

The check is cheap (one boolean on the bake metadata) and the cost
of getting it wrong is silent garbage output. Fixed call sites as
of 2026-05-15: `apply_mlp_scoring` in `zensim/src/metric.rs`,
`dataset_metric_baseline.rs` in zensim-bench, `ensemble_mix.rs` in
zensim-validate. New tools MUST follow this pattern from the first
commit.

Smoke test: validation harness includes one test pair with known
features and known expected score, run through the production
forward path. If the test ever passes for a transform-bearing
bake without `predict_transformed`, the dispatch is wrong.

### Step 8 — Visual diagnostics before the next experiment

Before designing the next iteration of bake / direction, produce
three diagnostics from the current results:

1. **Candlestick chart** of V_X → V_X+1 SROCC deltas per (corpus,
   band). Tells you which corpus / band is improving vs regressing
   at a glance. Builder:
   `scripts/v_next/build_per_pair_candlestick.py` (or whatever
   chart-build script the experiment log references).
2. **Per-pair extract** for any failure mode (e.g., TID B0/B1
   SROCC=0 ties → pull the 50 worst-error pairs, show
   `image_path, mos, score, raw_bake_output, clamp_flag` per
   row). Reveals whether the failure is clamp, calibration, or
   missing transforms.
3. **Low-n band ceiling analysis** for any (corpus, band) with
   n < 100. Builder:
   `scripts/v_next/v0_20_low_n_band_analysis.py`. Reports the
   empirical CI upper bound — that is the maximum plausible SROCC
   at that n, regardless of bake. Rankings between bakes are
   indistinguishable when n < 30; mark those bands as such.

These are 30-minute investments that prevent 4-hour
wrong-direction experiments. Build them BEFORE asking the
question "what should we try next?"

Artifact: the three diagnostics committed to `benchmarks/` and
referenced from the experiment log.

### Step 9 — Soft-clamp the runtime, recalibrate on shape mismatch

Multi-bake runtimes that mix bakes of different shapes (distance
+ score), or that extrapolate outside training distribution, will
produce raw outputs outside [0, 100]. Hard `score.clamp(0.0, 100.0)`
creates ties → SROCC collapses to 0 on affected bands.

Two ship-safe options:

1. **Per-bake recalibration**: rebake the score-shaped component
   with explicit `α=0, β=1` metadata, then re-derive the multi-bake
   mix in score space (each sub-bake gets affine-applied separately
   before mix). Use this when the runtime is locked to a known set
   of bake shapes.
2. **Soft-clamp**: replace the hard clamp with
   `100.0 / (1.0 + (-(raw - 50.0) / 20.0).exp())`. Preserves rank
   ordering at the extremes without flattening into ties. Costs
   ~1 ns per score (single exp call). Use this when sub-bake shapes
   may evolve over time.

PreviewV0_4 currently uses hard clamp and needs option 1 or 2
applied; documented as TODO in the V_20 learnings section.

Artifact: an integration test that scores 100 heavy-distortion
pairs from TID B0/B1 and asserts no more than 5% are pinned at 0
or 100.

### Step 10 — Falsification is data; commit the negative result

Negative findings are first-class output. When a hypothesis falsifies:

- Write the falsification log to `benchmarks/<experiment>_<date>.log`
  alongside the positive logs.
- Update CLAUDE.md learnings section with the negative result
  ("V_20b synth pre-train does not transfer to CID22 — FRIQUEE
  2017 caveat materialized").
- Add a one-line entry to the V_X timeline / candlestick chart so
  the failure shows up alongside successes.
- Do NOT retry the falsified hypothesis without NEW evidence
  (e.g., different pre-train corpus, different head architecture,
  external paper showing the recipe works on similar data).

A session that produces two falsifications and zero wins is NOT a
failed session — it's a session that ruled out two directions.
Commit the work like any other.

Artifact: the falsification log + CLAUDE.md learning entry + V_X
timeline update.

### Anti-pattern catalogue (don't do these)

| Anti-pattern | Failure mode | Correct pattern |
|---|---|---|
| Train 5 seeds before checking seed=1 | 5× compute wasted on dead hypothesis | Step 3 |
| Inspect CID22 mid-experiment | Information leak into hyperparam choice | Step 2 |
| Apply V_18's α, β to a score-shaped bake | Inverted predictions, SROCC=0 | Step 4 |
| Write Python that emits ZNPR v3 bytes directly | Wire-format drift, silent garbage | Step 5 |
| Concat bakes without verifying metadata agreement | Wrong forward path, NaN cascade | Step 6 |
| Call `model.predict()` unconditionally | Transform-bearing bakes produce garbage | Step 7 |
| Design next experiment without diagnostics | 4-hour wrong-direction experiments | Step 8 |
| Hard-clamp multi-bake output to [0, 100] | Ties → SROCC=0 on heavy distortion | Step 9 |
| Bury falsifications in scratch dirs | Future sessions retry dead hypotheses | Step 10 |
| `sed`/`echo` markdown tables to terminal | Tables don't render in message UI | Render directly in chat |
| Report SROCC alone, even per-band | Hides PWRC / Z-RMSE / OR regressions | Full Mohammadi panel |
| Skip ≥ 60 s commands' output to disk | Re-run cost when context loss | `2>&1 \| tee benchmarks/<name>.log` |

_(the following subsections are from `## Weight Training & Dataset Contamination`)_

### Dual weight arrays (FIXED)
- `WEIGHTS_PREVIEW_V0_1` in `profile.rs` — the canonical source of truth
- `WEIGHTS` in `metric.rs` — now a `&[f64; 228]` reference to `WEIGHTS_PREVIEW_V0_1`
- Previously these were independent copies that could drift. Fixed in commit ae28074.

### Current embedded weights (commit ae28074)
- Source: `runs/weights_20260306T110811_gpu_ssim2.txt`
- Algorithm: Nelder-Mead, 10 restarts, concordant-filtered 218k pairs
- Training SROCC: 0.9960 (on concordant), 0.9942 (on full 344k)
- 127/228 non-zero weights

### Validation results (raw distance SROCC / KROCC)

| Dataset | Old Embedded | NM concordant (embedded) | CMA-ES 0.9983 |
|---------|:---:|:---:|:---:|
| Synth 344k SROCC | 0.9882 | **0.9942** | 0.9974 |
| Synth 344k KROCC | 0.9123 | **0.9377** | 0.9592 |
| TID2013 SROCC | **0.8456** | 0.8427 | 0.8445 |
| TID2013 KROCC | 0.6612 | **0.6657** | 0.6619 |
| KADIK10k SROCC | 0.8090 | **0.8192** | 0.8140 |
| KADIK10k KROCC | 0.6012 | **0.6139** | 0.6084 |

CMA-ES weights at `runs/weights_20260307T124130_gpu_ssim2.txt` (42 non-zero, very sparse).

### Multi-dataset training (in progress)
- CMA-ES multi-dataset objective: `0.5 * mean_SROCC + 0.5 * min_SROCC`
- 6 training runs launched: {butteraugli, ssim2} × {KADIK, CID22, TID2013}
- Safe synthetic feature cache created; subsequent runs use it (~2 min vs ~30 min)
- Known issue: most CMA-ES restarts fail (1-2/10 converge), same as coord-descent
- Logs in `/tmp/train_{ba,ssim2}_{kadik,cid22,tid}_cmaes.log`
- Weight files saved to `/mnt/v/output/zensim/synthetic-v2/runs/`

### Key weight files on disk
| File | SROCC | Notes |
|------|:---:|-------|
| `weights_20260306T110811_gpu_ssim2.txt` | 0.9960 | **Embedded** (NM, concordant) |
| `weights_20260307T124130_gpu_ssim2.txt` | 0.9983 | CMA-ES, concordant, very sparse |
| `weights_20260307T124617_gpu_ssim2.txt` | — | CMA-ES KROCC=0.9650 |
| `weights_20260307T125005_gpu_ssim2.txt` | — | CMA-ES blended=0.9816 |

### Training algorithms available
- `--algorithm cmaes` — best single-dataset results, struggles with multi-dataset (high-dim)
- `--algorithm coord` — coordinate descent, 19/20 restarts overfit on multi-dataset
- `--algorithm pairwise` — RankNet SGD, converges to embedded weights (can't escape local opt)
- Default (no flag) — Nelder-Mead with random restarts, good for single-dataset

## V0_7 e1 fill (READY, 2026-05-05)

The V0_7 post-fill plan documented at `docs/NEXT_TIER_DATA_PLAN.md` and
`benchmarks/low_quality_improvement_plan_2026-05-01.md` (visible on the
`v04-mlp` jj branch) targets the SSIM2 25-60 band where current models
drop to 0.86-0.91 SROCC. The plan was to densify with ~140k
zenjpeg-420-e1 pairs at 39 q levels, then retrain V0_7 (V0_6 dct_hf +
sampler bias).

### Status as of 2026-05-05

- **e1 fill 87% complete + assembled.** The 2026-05-04 session fixed
  sibling-repo build breakages (commits a5b0042 on zenavif, e37e5f7
  on coefficient) so the `coefficient/examples/generate_zensim_training`
  binary builds. Local generator produced **122,117 e1 pairs** before
  CUDA context corruption ended the run early (~85% of the planned
  140k). The remaining ~21k pairs can be backfilled later.
- **Extended CSV ready**: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv`
  (340,206 rows = 218k existing safe-synthetic base + 122k new e1 rows).
  Zero CID22 validation leaks (the e1 source corpus
  `/mnt/v/input/zensim/sources` was already filtered).
- **V0_7 training is unblocked**: `bash benchmarks/v07_postfill_run.sh
  /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv`
  (visible on the v04-mlp jj branch).
- The 21k missing pairs are stored as encoded **bitstreams** on disk
  (`/mnt/v/input/zensim/images/<src>/zenjpeg-420-e1/qXX.jpg`; mirrored
  2026-05-22 to `s3://codec-corpus/synthetic-v2/<src>/zenjpeg-420-e1/qXX.jpg`)
  but weren't scored before the generator died. (The `qXX.png` decode-cache
  was deleted 2026-06-22 — see "Safe synthetic dataset"; the `.jpg`
  bitstreams remain on disk + R2.) Re-running the generator
  with the same args picks them up and re-emits a fresh CSV at
  `/mnt/v/output/zensim/training.csv`. Best done after a reboot
  (CUDA context is clean) or on vast.ai (next section).

### Future zensim compute on vast.ai

Per user request 2026-05-04: zensim compute jobs (e1 fill, V0_7
training, content-class experiments) should run on vast.ai when
possible, not local. Path:

1. Build the `coefficient/examples/generate_zensim_training` (and
   `zensim-validate`) binaries on a vast.ai box. The coefficient repo
   plus zen sibling worktrees would need to be cloned. Probably ~25
   min on a fresh box for cargo to build everything.
2. Source corpus mirror: 4653 sources at `/mnt/v/input/zensim/sources`
   (~1.8 GB). Sync to R2 once, then have workers `aws s3 sync` from
   R2.
3. The metric ledger needs to be central — easiest is to have the
   worker upload its ledger.jsonl back to R2 after the run, then
   merge locally before training.
4. Cost estimate: e1 fill is ~45 min on a single CUDA box (~$0.30/hr
   with GPU). V0_7 training is CPU-bound, ~30 min on 16-core box
   (~$0.15/hr). Total ~$0.30 per V0_7 cycle.

A scaffolded launcher script is the next deliverable; not yet
written. The local infrastructure is unchanged — both paths work,
but vast.ai is preferred to avoid blocking the user's machine.


_(note 2026-07-19: the following spec was implemented — see `site/compare.html` +
`site/js/compare-worker.js`; the section's closing 'Status: not started 2026-05-12'
line was stale when excised)_

## Interactive comparison site (CRUCIAL GOAL, locked 2026-05-12)

User spec, verbatim:

> make the interactive online gh site offer the following
> interactive interface: user can tap checkboxes to select a
> superset of image corpuses and their distortions. user can
> choose an x axis of any metric — codec quality, dssim, ssim2,
> butteraugli, zensim 02, zensim 18 or latest best, or human
> reference data. user can choose y axis from any of those.
> javascript will load parquet data for the corpuses and do a
> scatter plot, as well as linear line step 5 1 to 100 or equiv
> range along x axis, and a table for srocc and other stats per
> band for comparison of y compared to x. this is a crucial goal.
> cpu work on a background worker with progress indicator.
> additionally add separate data and charts as the 2023 paper
> does. offer both scatter and candlestick and ci interval tables
> by band. allow filtering by codec and codec version and y score
> to codec param table.

**User-clarified stack decisions (2026-05-12)**:
- **Query engine**: DuckDB-WASM (SQL over parquet, HTTP-range fetch).
- **Hosting**: Cloudflare R2 with public-read buckets. Parquet
  files NOT committed to the repo (gh-pages 1 GB cap).
- **Paper reference**: 2023 edition (the one we already have at
  `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`).
  Spec's "2023 paper" replaces the original "2024 paper" typo.

Implementation must:

1. **Corpus + distortion selector** — checkbox UI, multi-select.
   Each corpus knows its own list of distortions/codecs. At
   minimum: CID22, KADID-10k, TID2013, KonJND-1k, **AIC-3 CTC**
   (`/mnt/v/dataset/aic3_ctc_epfl/`), **AIC-4 sample**
   (`/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/` +
   metric/JND CSVs at
   `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/`), plus our
   internal synthetic safe-synthetic and unified V_X parquets.
   The selector is a SUPERSET — user picks any combination
   across corpora and the site stitches the rows together.
   **AIC-3/AIC-4 are mandatory for low-q human-judgment coverage**
   (CID22's MOS distribution is concentrated in B2/B3; AIC-3 CTC
   and AIC-4 reconstructed-JND span the B0/B1 bands that matter
   most for compression product decisions).

2. **X/Y axis dropdowns** — both can be ANY metric: codec
   quality (q), dssim, ssim2, butteraugli (3-norm or max-norm or
   diffmap mean), zensim V0_2, zensim V0_18 / latest-best
   (currently V0_16), or human reference data (MOS / DMOS /
   PJND). Both dropdowns must offer the same metric set; X=Y is
   the identity sanity check.

3. **Parquet loading from JS** — fetch `.parquet` files for the
   selected corpora directly. Use Arrow-JS or DuckDB-WASM, NOT
   CSV/JSON. Parquet files are committed under
   `site/data/parquet/<corpus>/<distortion>.parquet` or pulled
   from R2. The 2026-05-07 unified parquet store at
   `/mnt/v/zen/zensim-training/2026-05-07/unified/` is the
   shipping source of truth (~2.37 M rows × 50 cols).

4. **Background worker** — Web Worker with a visible progress
   indicator (file load, decode, statistics). Main UI thread
   must not block while a 3 GB parquet decodes.

5. **Scatter + step-5 line + per-band SROCC table** — for the
   selected (X, Y, corpora):
   - Scatter plot of every row.
   - A step-5 line: bin X by 5-unit steps from 1 to 100 (or the
     X-metric's equivalent range — e.g. butteraugli's 0..30,
     dssim's 0..1), median Y per bin connected.
   - SROCC + KROCC + PLCC + RMSE per band — emit the **10-band
     width-10 grid (B0..B9)** as the primary table, and the legacy
     4-band CID22 Table 5 cuts (B0..B3 + Near-PJND) alongside for
     paper comparison. Plus aggregated, with sample counts.

6. **Candlestick + CI-interval tables by band** — separate
   visualization mode. For each band on X, show Y's percentile
   box (p5/p25/p50/p75/p95) plus a bootstrap 95% CI on the
   median. Tabulate per (codec, band).

7. **2023-paper charts** — reproduce the figures/tables from the
   2023 CID22 paper at
   `/mnt/v/zen/zensim-training/2026-05-07/papers/CID22_wg1m99012.pdf`
   (Tables 3 per-codec SROCC, 4 PJND calibration, 5 band cutoffs,
   6 pairwise SROCC). The site renders these as static tables/
   plots ALONGSIDE the interactive widget so users can compare
   our V_X numbers against paper-reported baselines on the same
   page.

8. **Codec filtering** — filter rows by codec name AND codec
   version (e.g. zenjpeg-420 vs zenjpeg-444, JXL d1 vs d2, AVIF
   speed 6 vs 8). Filter persists across X/Y/corpus changes.

9. **Y score → codec param table** — given a target Y value,
   show the codec parameters that achieve it (within ±1 zq or
   equivalent). This is the user-facing "I want Y=70, what
   should the codec do?" lookup that motivates zensim shipping
   in the first place.

10. **No regressions**: the existing methodology page +
    pre-computed chart sections (8 chart sections at
    <https://imazen.github.io/zensimx/>) stay intact. The new
    interactive widget is ADDITIVE — a new page (or a top
    section on index.html) — not a replacement.

**Status**: not started 2026-05-12. Spec captured here; first
step is to inventory available parquet sources and confirm
schema (metric column names) before designing the UI.
