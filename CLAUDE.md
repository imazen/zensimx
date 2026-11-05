# zensim

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

**Feature-gap map (read before feature work):**
`~/work/zen/zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` — the
2026-07-26 audit of the 720-feature set (v1-vs-v2 iw/masked naming trap,
ranked weaknesses, fast-CPU candidates with evidence, regime-inversion
finding, don't-build list). The f720+ append block
(`benchmarks/v2_append_block_2026-07-26.md`, `FeatureRegime::Folded720Append`)
implements its A1-A5/A9 candidates.

## Known Bugs

- **⛔ THE STORED 372-COL masked/IW BLOCK IS PRE-FIX AND THREAD-DEPENDENT — THE
  RUNTIME PROFILE B IS NOT THE EVALUATED PROFILE B (found + measured 2026-08-30,
  OPEN as a DATA issue; the extractor is correct and is NOT to be changed).**
  Every `*_372col_2026-05-15.parquet` under
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/` — **which WAS
  `bake_verdict`'s DEFAULT `--features-root` until 2026-08-30, i.e. the root
  under every `--regime 372` verdict published before then** (the default is now
  the current-extractor root; see the DONE note below) — carries a masked
  (`f228..299`) + IW
  (`f300..371`) block that today's extractor does not reproduce, and that was
  **never reproducible**: at `58e6f8d8` (the commit those tables record as their
  own build) the block is a function of `RAYON_NUM_THREADS` — 1/2/8/28 give four
  different outputs, T1-vs-T28 moving 100 % of rows on all 144 slots by up to
  |Δ| 0.086. Cause: `2dab8f30` (2026-05-17) — the activity map read `bufs.mu1`
  at strip-overlap rows the fused V-blur never writes — plus `6af83b60`'s
  pre-2026-06-09 `rayon::current_num_threads()`-derived band layout, which chose
  where those rows fell. Both are FIXED; HEAD is bit-identical across thread
  counts and across the two v1 entry paths, and `2dab8f30` → HEAD is **0 cells
  over the golden tolerance** on 4,292 × 372. `f0..155` + `f156..227` never
  drifted (bit-identical stored↔HEAD on cid22val), so this is masked/IW only.
  **What this means for numbers you read:** shipped **B** puts 23 of its 95 live
  inputs in `f228..371`, including its largest weight (`f353`, norm 182.4).
  Matched-row, same bake, same pixels — stored root → fresh root: CID22 SROCC
  **0.87638 → 0.88212**, KonJND **0.54665 → 0.64967**, AIC-3 0.77743 →
  **0.76501**, TID **0.78683 → 0.77852**, KADID 0.82008 → **0.80847** (KADID is
  B's train==val CHEAT corpus), kon504 |0.59349| → **|0.51938|**. *(The AIC-3 /
  TID / KADID figures first published here were 0.79410 / 0.79691 / 0.80426 —
  key-aligned on a non-unique `(ref_basename, human_score)` key, which collapsed
  a whole repeated-key group onto ONE fresh row: aic3's fresh table held 100
  distinct rows of 600. Corrected positionally 2026-08-30; the AIC-3 and TID
  deltas change SIGN, so the runtime B is NOT better on every holdout —
  `benchmarks/eval372_current_root_2026-08-30.md` §5. CID22 + KonJND have unique
  keys and were never affected.)* Per-pair the **dial** shifts mean **−4.98** (CID22) / **−5.86**
  (KonJND) zensim points, >0.5 pt on 99.9 %/100 % of pairs, max 17.4 — and
  `Zensim::compute` at `codec_target` matches the FRESH prediction to 8 decimals
  on 10/10 sampled pairs. **So: do NOT cite a `--regime 372` verdict number for
  a `uses_f156_371` bake as the runtime's behaviour, and do not compare a
  stored-root number with a fresh-root one.** B's training tables carry the same
  pre-fix values (`canonical-2026-05-21/train/{kadid,tid}` are row-order
  identical to the 2026-05-15 root), so B is fit on pre-fix and serves post-fix.
  Gate against recurrence: `zensim/tests/v1_feature_width_pure_function.rs`
  (`v1_372_is_bit_identical_across_rayon_pool_sizes`,
  `v1_masked_and_iw_blocks_are_thread_invariant`). Fresh tables + drift matrix +
  `_MANIFEST.json`: `/mnt/v/output/zensim/v1-extractor-drift-2026-08-30/`.
  Record: `benchmarks/v1_extractor_drift_2026-08-30.md`,
  `docs/DATASET_HISTORY.md` §3.27.
  **DONE 2026-08-30 (§3.28):** the NEW dated root is
  `/mnt/v/zen/zensim-training/2026-08-30-full-features-372/` (drop-in
  `--features-root`, `_MANIFEST.json` with `build_commit` + per-file sha256 +
  per-corpus ERA; `kon504/` side root included), and the 372-class lineage is
  re-verdicted on BOTH eras — **the shift is model-specific, not a constant**
  (exactly 0.00000 for a basic-block-only bake → |Δ| 0.489 for `cl_tfm_LQ_MLP`
  on KonJND), with **41 ordering flips** (B goes 4th → 1st on CID22 in its
  comparison set; the 2-layer blend's +0.004 CID22 win over B is an era
  artifact). Six corpora in the new root are byte-COPIES (aic4 pre-fix and
  unrefreshable), so a zero delta there is an identity, not evidence. Record:
  `benchmarks/eval372_current_root_2026-08-30.md`; registry:
  `eval372-stored-root-thread-dependent-2026-08-30` (+ the basic-only immunity
  and dial-grid entries).
  **DONE 2026-08-30 (default flip):** `bake_verdict`'s (and `bake_compare`'s)
  default `--features-root` IS the current-extractor root, owned by ONE constant
  — `zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372` (the path was a
  literal in ten `.rs` files before). Every run now prints its ruler
  (`bake_verdict: features-root era — …`), two tests pin the default, and a
  flagless verdict was verified **byte-identical** to the same run with the root
  passed explicitly (full-json sha256 `9596f1bd…`; the markdown differs only in
  the wall-time line) and reproduces the round-4b current-era numbers exactly.
  **Nothing was rewritten** — the 2026-05-15 root stays on disk and stays a valid
  STORED-ERA read (`eval_roots::STORED_FEATURES_ROOT_2026_05_15`, which the
  probe/trainer bins now name explicitly so their era choice is visible); the flip
  only changes what a flagless invocation means going forward. The dial +
  corruption grids are NOT part of the flip — they are their own pre-fix files,
  annotated `dial372-grid-thread-dependent-era-2026-08-30`.
  **DONE 2026-08-30 (board):** the 11 current-era verdicts are on the gauntlet as
  `@cur372` rows — `benchmarks/board_era_rows_2026-08-30.md`, which also records
  the MEASURED finding that **7 of the 9 "stored-era" board rows were never read
  on the stored root** (they are `--regime 720` ext720 reads, bit-exactly
  reproduced; registry `board372-row-read-on-ext720-root-2026-08-30`) —
  independently reproduced by the round-4b lane (board `cl_tfm` vs a fresh
  `--regime 720 --corpora cid22`: **BIT-EXACT on 4,292 pairs**, and 96.4 points
  from a stored-372 read; board **B** and **`T_appT_b372_lam1e-3`** BIT-EXACT
  against its stored-root re-verdicts), so **only B's pair is a clean era A/B on
  the board** and the two era-scoped registry entries were **NARROWED in place**
  to those 2 cells — no post-fix row carries a false era-stale badge. The §3.28
  roster science never used a board row and is unaffected. A `--full-json`
  verdict now also records a **`features_root`** block (resolved path + registered
  era label + root manifest sha/regime + the per-corpus file sha256s it read), so
  a row's ruler is a grep, not an afternoon of re-running and diffing predictions.
  **STILL registered, not executed:** the 372 dial/corruption-grid rebuild (needs
  a decode pass — the `q<X>.png` cache is gone) and B's training-leg
  re-extraction (~227k pairs — a fleet wave).

- **✅ RESOLVED 2026-08-06 (appendix V) — THE CID22 B9 BAND WAS DEGENERATE AND
  F8 READ AN ABSOLUTE VALUE.** Kept here because **every per-band number
  published before 2026-08-06 is still wrong** and 160 board cells still carry
  the old bands. What was measured (appendix U found it, appendix V G-V1
  re-verified it on 120 cells — a different population, a different code path):
  `zenstats::panel` returns `spearman(..).abs()` (panel.rs:1013), so
  `rank.<corpus>.bands[].srocc` was an ABSOLUTE value — **120 of 120 stored
  values equal |recomputed signed| exactly** — while **109 of 120 (90.8%) were
  NEGATIVE** and `|B9| >= 0.15` passed **82** where signed passed **2**. Since
  |·| is monotone in the depth of an inversion, the column ranked models by how
  backwards their top band was: the published leader `coherent924_selected`
  (|B9| 0.4493) is the population's most anti-correlated model at −0.4493. The
  band was also degenerate — 43 pairs from 11 of 49 refs spanning 0.0194 MOS,
  split-half r_SB 0.711 against a 0.90 bar — and CID22 B0/B1 were structurally
  EMPTY, B2 held one pair, TID B9 zero and B8 one, and the bottom band closed at
  0.0 so LIVE's 21 sub-zero DMOS pairs fell out of every band (its rows summed
  to 758 of 779). **Fixed:** band edges now have ONE owner
  (`zensim_validate::bands`, scheme `merged-decile-2026-08-06`) which merges
  deciles until every band clears **n ≥ 1000 AND span ≥ 0.08**; bands are open
  at both ends; an unusable band is **NOT-MEASURED** with a reason, never a
  zero; F8 reads SIGNED usable tails with a DERIVED floor (0.09 = the band's
  reference-clustered CI half-width). **Still live for readers:** do NOT cite
  any pre-2026-08-06 per-band value, and never `bands[].srocc` — read
  `srocc_signed` and state `n` + `span` beside it. Registry entries:
  `band-srocc-absolute-fixed-decile`, `f8-b9-abs-bar-superseded`,
  `balanced-composite-bandtail-abs`, `kadid-bands-half-corpus-subsample`.
  Record: campaign appendices U + V, `benchmarks/band_minimum_n_2026-08-06.md`,
  `benchmarks/appendixV/`.

- **⛔ `bake_dial_refit predict --ensemble` BLENDS IN RAW UNITS; `bake_verdict
  --ensemble` BLENDS IN SCORE UNITS — the two disagree on every k≥2 blend, and
  the teacher tables were built with the wrong one (found + measured
  2026-09-04, OPEN).** `predict` forwards through
  `zenpredict::Predictor::predict{,_transformed}` and accumulates `w·p[0]`
  directly, so the bake's `zentrain.output_calibration_spline` is **never
  applied**; `bake_verdict`'s `Ensemble::score_rows` applies each member's
  spline first. `predict`'s own doc claims the opposite verbatim ("same
  averaging order — after each member's own output spline, i.e. in each
  member's score units… the teacher a distillation trains against must come
  from the same forward the evaluation used"). MEASURED on 504 KonJND JPEG
  rows at `r1b-pools944-2026-08-30`, the `HYA` pair (W10L9PH + Q7b): member
  ranges **[48.7, 75.9] / [44.4, 74.1]** through `bake_verdict` vs
  **[−2.02, 5.16] / [−0.30, 0.11]** through `predict`; blend |SROCC| **0.5390
  vs 0.5073** at w=0.5 and **0.5218 vs 0.5019** at w=0.84 (both hand-reproduced
  from the per-member vectors, max |manual−tool| = 0). **It hides at k=1** —
  a monotone spline is rank-invariant, so every single-bake SROCC agrees — and
  it bites hardest when members' raw scales differ (here Q7b's 0.41 raw span
  against W10L9PH's 7.18 means Q7b contributes ~1.1 % of a w=0.84 raw blend
  and ~16 % of the score-unit one). **Blast radius: every distillation teacher
  table built by `predict --ensemble`**, including
  `safesyn_distill_hya_r4.parquet` — so `A4b`/`K2`/`K3` and the
  2026-09-04 fastclass wave were distilled against ~W10L9PH alone, not against
  `HYA_w084`. Consequence for numbers you read: the campaign's
  **super-additive KonJND peak (0.5390) is at w = 0.5–0.6, NOT at the w = 0.84
  the teacher is named for** (score-unit curve, reproduced: 0.5390 / 0.5218 /
  0.5006 at w = 0.50 / 0.84 / 1.00), so the shipped teacher is *below* ssim2's
  0.5272 on that axis. **Do not cite a k≥2 `predict --ensemble` output as "the
  ensemble"**, and do not compare a `predict`-built teacher with a
  `bake_verdict`-scored one. **FIX LANDED 2026-09-04 (`58baf010`, ancestor of
  `main@origin`)**: `bake_dial_refit predict --score-units` (default OFF, so no
  stored raw-unit recipe moves silently) routes each member through
  `bake_runtime::score_row{,_minmax}` — the functions `bake_verdict`'s scorer
  calls — before accumulating. Gate `scripts/verify_predict_score_units.sh`:
  `max|delta| = 0.0` vs `bake_verdict --ensemble` at w=0.5 AND w=0.84; a
  same-tree control proves the default path byte-identical on k=1 and both k=2
  shapes. **What has NOT been rebuilt is the affected DATA**: the
  `safesyn_distill_hya_r4` teacher table and the 31 board rows distilled from
  it. New blended-teacher builds must pass the flag. Board consequences are
  registry-scoped and rendered as of 2026-09-04 —
  `distill-teacher-raw-units-pre-58baf010-2026-09-04` (the 31 teacher-derived
  rows fail the fair filter's criterion (f)) and
  `ens-board-rows-score-units-unaffected-2026-09-04` (the 15 ensemble ROWS are
  on the CORRECT side: their numbers come from `bake_verdict --ensemble`).
  Records: `benchmarks/fastclass_distill_wave_2026-09-04.md` §6f (AMENDMENT A2),
  `benchmarks/fair_gauntlet_2026-09-04.md` §1.2.

- **⚠ `--regime 944` SILENTLY MIS-SCORES a 372-input bake that uses f156-371**
  (found 2026-08-06 the hard way, OPEN). The folded regimes zero f156-371, so a
  root-A model with weight there gets structural zeros for exactly the block it
  relies on — and gets a plausible-looking number back, with no warning:
  **shipped B reads CID22 0.3862 at `--regime 944` against its true 0.8764 at
  `--regime 372`** (v47A −0.005; ADD156 ±0.0001, immune because it is
  f0..155-only, which is what T.R4's bridge measured). Appendix T's root split
  exists for this; the invocation looks safe precisely on the bakes that are
  immune. **Check `bake_block_profile` before choosing `--regime` for any
  <944-input bake**, and treat a large 372-vs-944 gap as this bug, not a finding.
  The INVERSE instance is fixed 2026-08-27: the trainer's auto-eval hook used to
  invoke bake_verdict with NO `--regime` regardless of width (the HDR-944 L1
  bakes' auto-verdicts ran the 372 root); it now maps `--max-features` 720/944
  to the matching regime and warns on unregistered widths.
  **A CONCRETE INSTANCE IS ON THE BOARD**: `ebothg_m504` (`model.n_inputs` 504,
  `block_profile.uses_f156_371` true) publishes CID22 **0.4045** with EVERY axis
  collapsed (aic3 0.11, nonphoto 0.14, imazen26 0.15, kadid 0.18, aic4 0.19,
  tid 0.20, konjnd 0.25, csiq 0.32, live 0.40) — which is not how a promoted
  top-5 model behaves. Its `.spec.json` argv shows it was trained on the
  **ext504** root (`/mnt/v/zen/zensim-training/ext504-basic-v2-2026-07-23/`),
  but its fulleval is stamped `regime: "720"`; re-scoring reproduces the board
  at `--regime 720` (0.4045) and gives 0.6915 at `--regime 944`. **Neither is
  its native root**, so the published row is a wrong-root read and its true
  numbers need the ext504 tables. Not fixed here (it is another lane's model);
  flagged so nobody ranks against that row.

- **⛔ ext-LINEAGE KADID TARGET WAS STORED INVERTED (found 2026-08-04; TABLES REBUILT
  2026-08-05, wave 10 — pre-2026-08-05 numbers remain sign-flipped).** `ext720`/`ext924`/`ext944`
  `ext_kadid.parquet` carry `human_score = (5−dmos)/4`; the canonical `(dmos−1)/4` is
  correct because **KADID's `dmos` is a MOS in disguise** (raw crowdsourced DCR falls
  4.0789 → 2.0072 across severity levels 1–5, 349,800 ratings).
  `build_fr_corpus_pairs.build_kadid()` applied the invert-a-DMOS reflex that CSIQ/LIVE
  genuinely need to a column that was already quality-oriented. **Consequences:** every
  campaign KADID figure is the negative of the true-quality value; **110 of 188 board
  bakes are ANTI-CORRELATED with KADID's real human MOS**; the 944 models *trained* on
  the flipped column, so their inversion is real and inherited (train weight 0.50 → mean
  −0.457, 1.50 → −0.925); the era models are fine (`winner_dial` **+0.9464**, shipped
  **B** **+0.8201**, positive on 25/25 distortion types). **TID is CLEAN on every root.**
  **Read `rank.kadid.srocc_signed`, and NEGATE it only for a verdict produced BEFORE the
  2026-08-05 rebuild; never cite `rank.kadid.srocc`.** The rule is **verdict-era-scoped,
  not root-scoped** — negating a post-rebuild verdict re-introduces the inversion
  (independently re-confirmed 2026-08-06, campaign appendix T.R3: the orientation gate
  returns **+0.582360 on BOTH** the ext944 `ext_kadid.parquet` and the 372-root
  `kadid_features_372col`, and a fresh ADD156 verdict reads `+0.8082` on each root while
  its pre-rebuild board row reads `−0.8082`). `freeze_check --annotations` encodes the
  same split as `kadid-ext-root-inverted` (scope: the 188 pre-rebuild verdicts) vs
  `kadid-ext-root-corrected-2026-08-05`.
  **⚠ THE PAIRS TSV WAS NEVER FIXED (found 2026-09-01, wave-r4).** The
  2026-08-05 repair rebuilt the ext parquets in place and fixed
  `build_fr_corpus_pairs.build_kadid()`, but
  `/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv` — the artifact
  `extract_944_canonical.sh` actually feeds to the extractor — still carries the
  inverted orientation. A fresh extraction therefore copies the bug into a
  brand-new root, past every gate that only inspects ext parquets. MEASURED on
  the wave-r4 root: `ext_kadid` 10,125/10,125 rows exactly `1 − corrected`,
  orientation gate −0.582360, while all fifteen other legs matched the corrected
  era-1 root exactly. **Run `check_target_orientation.py --corpus kadid` on every
  freshly-extracted KADID table before training on it**, until the TSV is fixed
  at its owner.
  Builder fixed + `scripts/canonical_corpus/check_target_orientation.py`
  gates it; **the ext tables WERE rebuilt 2026-08-05 (campaign APPENDIX H part 1,
  `176c4268`)**: `human_score := 1 − human_score` at all three ext roots, gate now OK
  +0.582360 on every root, originals preserved as
  `ext_kadid_INVERTED_2026-08-04.parquet` (triple-mirrored, sha-recorded — ext944
  inverted sha `4dde6be2…` = the sha in every affected repro). REPRO HAZARD: re-running
  a pre-2026-08-05 repro argv verbatim now trains on corrected bytes; substitute the
  `_INVERTED_` file to reproduce. Existing bakes stay annotated, NOT retrained.
  Measured value of the fix (wave 10 L0 vs incumbent, k=3 each): CSIQ +0.115, LIVE
  +0.073, AIC-3 +0.016 (all outside noise), KADID signed +0.40, first-ever 8/8
  balanced-floor cells.
  Determination: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F
  (+ F.R1..F.R9); ledger: `docs/DATASET_HISTORY.md` §3.20; registry:
  `benchmarks/eval_annotations.json` (`kadid-ext-root-inverted`,
  `kadid-ext-trained-inverted-model`, `kadid-e1-gate-unsigned`).

### Resolved

- **✅ RESOLVED 2026-08-30 — v1's 372-feature vector could come out 93 / 186 / 279
  wide, and it was SIZE, not batch.** `docs/DATASET_HISTORY.md` §3.26 registered
  this as "a v1-372 feature vector is not a pure function of its pair … it is a
  function of the BATCH". **That framing is retracted**: the pre-fix binary
  (`6d0a393a`) gives **5 short of 5 run alone, 453 of 453 alone, 453 of the
  6,953-row batch** (§3.26 predicted 0 / 33 / 453), and the row values are
  byte-identical across every batch composition. The width is
  `2 + n_scales(W,H)·3·31` — predicted with **ZERO errors on all 20,812 stored
  rows**. The scale walk starts at `simd_padded_width(width)` but plain
  `height`, so 4 scales need **`simd_padded_width(W) >= 64 AND H >= 64`**; that
  asymmetry (`54x96` FULL vs `96x54` SHORT) is what made "too small" look
  falsified. **Mechanism:** `compute_with_config_inner` reflect-pads for every
  `Zensim::compute*`; three entries did not —
  `compute_zensim_with_config` (`metric.rs:4800`, `training`) returned a SILENT
  short vector and is called by BOTH v1-372 extractors, while
  `compute_zensim_with_ref_and_config` (`metric.rs:706`) and
  `Zensim::compute_with_ref_into` (`metric.rs:2271`, a **product** API) PANICKED
  `scale 0 width mismatch`. **Fixed** in `f9fac41e` by giving the pad decision
  ONE owner (`metric::needs_pyramid_pad` + `min_pyramid_dim_for_scales` +
  `reflect_pad_for_scales`) used at all seven pyramid entries, `num_scales`-aware
  so `--num-scales 5/6` cannot truncate either. **Gate:**
  `zensim/tests/v1_feature_width_pure_function.rs` (8 tests; 5 fail pre-fix).
  **Blast radius, measured:** 0 of the 149,195 canonical-leg pairs could
  truncate; every canonical 372 parquet is full width; the 944 fold is immune and
  byte-identical pre/post; `bake_verdict` never extracts; 19,444/19,444
  previously-372 R1b rows are BYTE-IDENTICAL after the fix and 1,368/1,368 short
  rows became 372, with `f0..f155` bit-identical to the stored 944 fold. **Still
  open (registered, not executed):** `r1b-samepair372-2026-08-30` is a
  size-correlated 6.5 % row-restriction and `r1b-372root-2026-08-30/` has three
  dangling symlinks; full-width CSVs are at
  `/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/`. Record:
  `benchmarks/v1_width_defect_2026-08-30.md`.


- **v1 golden byte-identity gate environment fragility — CLOSED-BY-POLICY 2026-08-05.**
  USER RULING (verbatim): *"the golden-gate policy is tiny tolerances, not per-class
  exactness; and full-precision fallbacks are acceptable only if runtime-optional via
  generics — prefer archmage's precision-TIERED variants."* The exact-f64 golden never
  held cross-vendor (goldens = AMD Zen 4; every non-AMD class diverges on 241-246/372
  features, all producing ONE shared alternative result set). Converted
  `zensim/tests/v1_golden_bytes.rs` to `|Δ| <= max(1e-6 abs, 1e-5·scale)` — derived
  16-17× above the MEASURED full-372 cross-class spread (max abs 6.00e-8 at f62-real;
  max rel 6.06e-7 among ≥1e-2-scale features; the triage doc's "~1e-10 rel" was the
  abs-delta column misread) — and added `v1_same_class_determinism_bitexact` (two
  same-box computes must stay bit-exact; determinism survives the conversion). The
  companion rsqrt Adam kernel got runtime-selectable precision tiers
  (`RsqrtPrecisionTier`: full/nr1/estimate, full default). Record + measurement:
  `benchmarks/v1_golden_env_triage_2026-08-05.md` §RESOLUTION; tier costs:
  `benchmarks/adam_rsqrt_tiers_2026-08-05.md`.

- **Validate-side output-spline upper extrapolation diverged from the product
  runtime — RESOLVED 2026-07-04 (`5d4978db`).** `output_calibration_spline::
  apply` extrapolated linearly UNCAPPED above the top knot while
  `zensim/src/metric.rs` caps at ≤100; the file's "bit-exact" claim was false
  above the top knot and produced dial-p95 artifacts of 300-500 on linear
  bakes. Now capped for parity (bottom stays uncapped — neg-tail corruption
  resolution). `parse_round_trip_minimal` had enshrined the divergent value
  (110); expectation corrected to 100 per the product contract.

- **konjnd-agg 2-layer w1 gradient "bug" — RESOLVED 2026-05-27 as a
  malformed test, NOT a gradient error.** The
  `konjnd_aggregation_2layer_w1_gradient_matches_finite_difference` test
  reported ~2–3% (and at one point 48%) relative error vs the analytical
  gradient. The gradients are **correct**; two compounding test defects
  produced the spurious failure: (1) the forward computes in **f32**
  (`dot_bias` casts f64→f32), so a central difference is floor-limited —
  at ε=1e-6 the rounding noise in `(f₊−f₋)` (~1e-7) swamps the signal
  (~2·ε·grad), giving O(1) relative error; (2) a **pure relative gate**
  is unbounded as the true gradient → 0 (near-zero entries like
  `gw1[4]≈-9e-4` have abs diff ~6e-6 = a correct gradient). Fixed by
  ε=1e-2 + the standard `|num−ana| < atol + rtol·max(|·|)` gradcheck
  criterion. The earlier "α≈1 drops error 48%→10.5%" observation was
  itself an artifact of the f32 floor manifesting differently across α
  regimes — debunked by a new train-core test
  (`per_sample_alpha_head::tests::backprop_heads_dl_dh_matches_finite_difference`)
  that FD-checks the head/encoder gradient directly (L=y, dl_dy=1) and
  passes cleanly. Shipped bakes were never affected. Commit: see
  `fix(#35)`.

## Canonical training data + indexes (added 2026-05-20)

**The canonical index for all ML data lives at `~/work/zen/DATA_PROVENANCE.md`.**

Quick paths:
- **Dial anchor (imazen-26, 2026-09-04) — a PROPOSAL, not a default:**
  `/mnt/v/zen/zensim-training/2026-09-04-imazen26-anchor-372/` (LAN
  `s3://zentrain/anchors/2026-09-04-imazen26-anchor-372/`). 4,000 rows, 4 codecs
  × 10 decile bands × 100, 192 origins, `target_score = max(score_ssim2,0)`,
  nothing re-encoded, **decoder era recorded per format** in `_MANIFEST.json`.
  Ships alongside `safesyn_anchor_curera_2026-09-04.parquet` — the SHIPPED
  anchor's own 2,000 rows re-read today — which is what makes the era term and
  the content term separable. MEASURED: era **+3.9/+4.8/+3.9** (CID22/KonJND/
  AIC-3) vs content **−0.4/−1.0/−0.2**, rank identical to 5 dp on all five
  corpora, all gates pass, but dial reach falls 96.85 → 85.74. **`zensim/weights/`
  is UNCHANGED; `ZensimProfile::B` still resolves to the 2026-07-07 bake.**
  Record: `benchmarks/imazen26_anchor_2026-09-04.md`, ledger §3.35. NOTE
  `extend-top` alone CANNOT fix an era skew — it keeps in-distribution knots
  verbatim and only extends above the top knot, which CID22 never reaches; use
  `shared-anchor` for the whole-spline refit.
- **Eval features root (`--regime 372`), default since 2026-08-30:**
  `/mnt/v/zen/zensim-training/2026-08-30-full-features-372/` — the CURRENT
  extractor (`build_commit ea16c7ee`). The previous default
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/` stays on disk as a valid
  STORED-ERA read; it is not deleted and not rewritten. Both are named once, in
  `zensim_validate::eval_roots` (`DEFAULT_FEATURES_ROOT_372` /
  `STORED_FEATURES_ROOT_2026_05_15`) — never re-type the literal. Every
  `bake_verdict` run prints which era it read. Era shift is model-specific: see
  `benchmarks/eval372_current_root_2026-08-30.md`, ledger §3.28.
- Trainer input: `/mnt/v/zen/zensim-training/canonical-2026-05-21/` (local) + `s3://zentrain/canonical-2026-05-21/` (R2) + `/mnt/tower/output/zensim-archive-2026-05-20/` (Tower) — **training legs are still PRE-FIX** (§3.27); the flip above is an EVAL default only
- Per-row truth: `_MANIFEST.json` in each canonical/picker training dir
- Master inventory: `~/work/zen/_ml-inventory-2026-05-20/00-MASTER-SYNTHESIS.md` (7-part forensic inventory of repos + parquets + datasets, 2026-05-20)
- Worktree audit: `~/work/zen/_ml-inventory-2026-05-20/01-zensim.md`

⚠ **CORRECTED 2026-08-30 — the 2026-05-20 byte-equivalence audit
(`10-canonical-build-audit.md`) sampled ONLY `f0..f99`.** This paragraph used to
read "confirmed current zensim main produces features bit-equivalent to all 13
canonical-2026-05-21 parquets (sub-ULP precision). No build drift; trustworthy
as-is." That conclusion is correct **for `f0..f99`** — the audit's own §1 says
*"emits f0..f99"* and its tolerance is
`max_abs_diff(extracted_f0..f99, parquet_f0..f99)` — and `f0..f99` is entirely
inside the **basic** block, the one block that did not drift. It does **NOT**
extend to `f156..371`: the masked (`f228..299`) and IW (`f300..371`) blocks of
every 2026-05-15-era 372-col table differ from today's extractor on **100 % of
rows** (max_abs 0.0374 / 0.1235 on cid22val), and
`canonical-2026-05-21/train/{kadid,tid}.parquet` carry those same pre-fix values
(row-order identical to the 2026-05-15 root on `f0`/`f228`/`f300`/`f353`). See
the Known Bug below and `benchmarks/v1_extractor_drift_2026-08-30.md`. The
audit's §5 softening of the `DATA_PROVENANCE.md` "semantically incompatible"
warning rests on the same 100-column sample and is likewise scoped to basic.
The `cvvdp_iwssim_LARGE_372col.parquet` (73,300 rows, 85.5 MB, sha256: 14c205332701b5ff6f2842a8d60f8ac1282f8be3d5cd89c11700e1e4b864a20f) lives at `canonical-2026-05-21/features/` — extracted 2026-05-20 to fill the f300..f371 IW-pool gap.

## ★ THE 924-FEATURE PARQUETS (folded+append STREAMING regime) — the current-era datasets (2026-07-27/28)

**Every canonical dataset now exists at 924 features** (regime `Folded720Append`, zensim
`0b3d16b0` C5 streaming-only, `ZENSIM_AB_MODE=foldapp`, `codec_target` profile, RAW unpadded
slices). Layout: f0..f155 folded v1-basic, f156..f371 STRUCTURAL ZEROS, f372..f719 v2-348,
f720..f923 append-204. **REGIME PURITY: never column-mix 924 rows with 720/v1 parquets**
(padded-width divergence + zeroed pools). All triple-mirrored (local + R2 + Tower, sha-manifested);
full provenance in `~/work/zen/DATA_PROVENANCE.md`. Train on THESE for all new work:

| dataset | rows | local path |
|---|---|---|
| 11 local legs (cid22val/aic3/aic4/csiq/live/kadid/tid/safesyn/cid22t201/konjnd/sdr25) | 149,195 | `/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/` |
| bigcodec fleet table `tbig_924_full.parquet` (keyed encode_sha) | 5,742,660 | `/mnt/v/output/zensim/tbig-924-2026-07-27/` |
| bigcodec 21 split views (7 picker datasets × train/validate/test, match_rate 1.0000) | 5,742,660 | `/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec/<dataset>/<split>_924.parquet` |
| `kadis700k_924.parquet` (7 byte-carried metric targets, split on source_id) | 699,999 | `/mnt/v/zen/zensim-training/kadis-924-2026-07-27/` |
| `kadis_negrich_924.parquet` (severe, score_zensim_gpu<0 — corruption-head negatives) | 167,034 | same dir |
| eval instruments: `corruption_grid_924col` + `dial_grid_924col` | 2,016 + 4,817 | `/mnt/v/output/zensim/v2-eval-924-2026-07-27/` |

Eval slices: FULL_EVAL's 924-era imazen26/nonphoto point at the **canonical bigcodec 924 TEST
views** (see `docs/FULL_EVAL.md` "924-era eval slices"); the NN-matched `ext_*_720` tables are
720-legacy, never rebuilt. First instruments on this data: P12 residual-boost ranking + P11
decorrelated-auto (both arms + the empirical S-class map) — `benchmarks/p12_*`/`p11_*_2026-07-27*`.

**Coming later: additional HDR features.** The 924 set is SDR-only by design; a future wave
appends HDR-specific features (the append-only discipline holds — new slots will EXTEND, never
renumber, and HDR rows will be their own regime/datasets, never column-mixed into these).

## ★ THE E-M CAMPAIGN (924/v3 era) — findings + recipes + the steering pivot (2026-07-28/29)

Full evidence: `benchmarks/coherent089_seeded_frontier_2026-07-27.md` (E-M1..E-M9).
Commits `555b1a48`..`aa5576f4`. Bakes: `/mnt/v/output/zensim/bakes/coherent-089/{,em2/}`.

### Rank results (924 regime = coherent by construction)
- **fold924 kw0.5, n=6: CID22 0.8825±0.0025**; **no-KADIS n=6: 0.8816±0.0016 + KonJND 0.398**
  (v3 features carry near-threshold natively; KADIS ROLE-REVERSED at 924 — suppresses KonJND,
  rescues CSIQ; crash zone kw(0.15,0.5]).
- **ATTRIBUTION (E-M6b, width discriminator): most of the CID22 lift is the DATA** — the
  924-era bigcodec slice lifts 720-width to 0.8837; v3-marginal ≈ +0.001. tbig_924_200k is
  the one non-row-identical slice (no join key) and drives the coarse-scale shift too.
- **Seed selection is VALIDATED**: sdr25 (bake_verdict corpus, never trained, not a gate)
  predicts CID22 at SROCC +0.752 over 35 bakes, rejects every collapsed seed. Best selected:
  `EM4_mask2_kw0.15_s42` CID22 0.8924 + KonJND 0.4286 (`coherent924_selected` on the gauntlet).
  **⇒ SUPERSEDED 2026-08-04 as the ship rule** (still true as a fact about sdr25): selection
  is now **rank + dial + COHERENCE**, run by the OWNER —
  **`freeze_check --select <every fulleval>`**. PRIMARY = profile floor count; TIE-BREAK =
  `balanced_composite + 0.15·M3a`; `sdr25` is a reported comparator only, **never the
  primary** (it has decoupled from CID22 five times). Rationale: the coherence study measured
  **42.3% of 944-class M3a variance is seed noise at fixed recipe** (`C_co3a` k=6 spans
  0.7367-0.8786 on corrected post-299ccc8c values — the study's own 0.718-0.826 predates the
  append2 coverage fix and understates it), so M3a is a *selectable trajectory property* and a k-seed wave that ignores it
  leaves ~0.1 M3a on the table. Three M3a states, **none of them zero**: MEASURED ranks;
  NOT COMPUTABLE (ensemble — the instrument loads one ZNPR) ranks separately and is never
  penalized; UNMEASURED is listed but **not selectable**. M3a comes free with
  `harvest_bakes.sh` (27 cells, **66 s/bake** measured); a missing one drops a `.NO_M3A`
  marker. Registered: campaign appendix E.4; workflow: `docs/WAVE_PLAYBOOK.md` step 6.
- Corruption ordering breaks at 924 in ALL arms (0.03-0.17 vs 0.214@720); occlusion blamed
  DET/ART_DEV2 but **masked RETRAIN does not recover (occlusion≠ablation)** — distributional;
  mitigate with the corruption HEAD (negrich_924), not the dial.

### The coherence resolution (THE steering pivot — build task #67)
- Mechanism: 924-era data → ~89% basic gradient mass on COARSE-scale MSE → the
  mass-blended per-pixel fold degrades to 1/8-res → M3 0.10-0.25 while **M2 ≈ 0.99**.
- Weight-decay remedies FAIL: coupled L2 is neutralized by Adam (AdamW insight); decoupled
  `--coarse-decay` bites but hits data-preference equilibrium (s3 ~50% across 100× rates).
  **Keeper: `--coarse-decay 1e-5` = KonJND +0.15, CSIQ +0.07, ~free.** Scale-mass = symptom.
- **PROVEN architecture (E-M9): per-block gradient attribution M2 = 0.999-1.000 at EVERY
  block size 16-128px** on the pathological bakes, while the signal-fold M3 inverts at 128px.
  **Ship design: per-pixel attribution DENSITY (true per-feature integrands; mean-pooled
  exact — covers the dominant MSE slot; p2/p4 value-weighted) + summed-area table → O(1)
  arbitrary-rectangle queries** for variable codec partitions (AV1 4-128px, JXL var-DCT,
  HEVC CTU). The old signal fold becomes visualization-only. Perf bar: build ≤ current
  diffmap build (same planes, different weighting, +O(N) SAT). jxl's loop is
  global-iterative (iters+1 full-frame compares) → full-map-per-iter is its contract;
  region-incremental re-query is for per-block-probe codecs (zenjpeg) — SAT re-query is
  already O(1), local re-COMPUTE bounded by blur footprint is the later optimization.
  **Order (user): get it coherent, THEN optimize.**
- **C1 SHIPPED (2026-07-29): `Zensim::compute_attribution_density` + `AttributionResult`
  (SAT, `query_rect` O(1)) with exact basic integrands** (p2/p4 removal-consistent 1/p;
  hf = signed clamped-ratio first-order — slots 10-12 are ratio-pooled, NOT means).
  M3a > M3 in 8/8 gate cells; healthy-720@128 = 0.895; but the ≥0.85 gate is NOT met:
  ATTRDIAG proves |s|-mass ≠ rank-variance — EM2's 98.4%-mass basic block true-ranks ΔS
  at only 0.33-0.54 (append-blind ceiling 0.43-0.68, NEGATIVE at 128 where the 0.5%-mass
  append block alone carries the signal). Distance = append fold + exact non-additive v2
  integrands, both measured. Raw v2map ADD into the score-unit density is unit-broken
  (swamps it) — use weights ×1/(w·h). Perf 2.4-3.4× (C2 target ≤1.1×).
  `benchmarks/attribution_map_c1_2026-07-29.md`.
- **C2a SHIPPED (2026-07-29): `compute_attribution_density_full` — exact integrands for
  ALL v2+append slots** (production-kernel pass A, 1e-9 feature parity; FD direction tests
  caught an edge-width sign bug pre-landing). Gate 5/8: **EM2 4/4 ≥0.85 — the 128px
  inversion is CURED (−0.36 fold → +0.99, at the M2 ceiling; the 0.5%-mass append block
  was the whole coarse signal)**; K720 1/4 (0.69/0.53/0.38/0.90) — miss isolated to v2
  density approximation at fine blocks (v2attr-vs-true-lin 0.61/0.40/0.18/0.89; basic holds
  0.89-0.98) → C2b lever = blur-bleed spreading + perf (full 125-138ms vs fold 11ms).
- **C2b SHIPPED (2026-07-29): bleed-spread hypothesis MEASURED-FALSIFIED + perf −30%.**
  The `I−K` adjoint is structurally wrong for residual signals (zero net mass); the 50/50
  split REGRESSED all 8 cells; window-only spread (shipped) is neutral — K720's fine-block
  gap is the finite-removal floor, not allocation. Gate unchanged 5/8 (EM2 4/4 holds).
  Perf: single-sweep + channel/row-band rayon → full 95-98ms, basic ~38ms vs fold 11-12ms
  (8.3×/3.3×; ≤1.1× structurally needs fusion into a shared 924 compare — levers + estimates
  in the C2b doc section). `blur::box_spread_sum_preserving` = the exact-sum spread primitive.
- **C3a SHIPPED (2026-07-29): the FUSED compare** — `Zensim::compute_with_ref_score_and_
  attribution` = score + steering map from ONE pipeline (v1/372 class; score BIT-identical
  to the fold path, gated; standalone paths untouched, 8-cell identical). 576²: score+map
  **14.8ms total** (the old standalone map alone was 36.8). Marginal-map bar (≤1.1× fold's
  marginal) missed at floor 5.7×@576/2.2×@1152 — the fold folds in-kernel with no pooled
  scalars; next levers ranked in the C3a doc (ref-side hf cache → in-kernel mean slots →
  stale-scalar single-pass). 924 fusion needs extractor-side hooks (that session's domain).
- **C3b (jxl loop A/B, `f195c8c0` in jxl-encoder): attr steering did NOT beat the fold at
  target-hitting** — med |achieved−target| v47A 0.807 vs fold 0.594 (and BASELINE 0.244
  beats both model maps there); shippedB 1.507 vs 0.982; 1W/7-8L per-cell, equal median
  iterations. Probe shows the tile signal is STRONG (full ratio range at 8px) and the fused
  scalar tracks decode BETTER than the fold arm — the loss is the allocation×redistribution×
  controller interaction, NOT the map. So: **M3a coherence is proven; loop VALUE is not** —
  the next lever for closed-loop wins is the controller/redistribution design, not map
  fidelity. `attr-stale ≈ attr` (0.589/1.355) ⇒ the stale-scalar single-pass ≤1.1× perf
  endpoint is semantically viable when a positive product case exists. Fused adds
  +8.2ms/compare @576² (matches C3a's marginal). Caveats: n=9 cells/bake, t=92 clamps
  saturate, t=75 controller-overshoot-dominated. Tables:
  `jxl-encoder/benchmarks/zensim_attr_ab_2026-07-29.md` (+2 TSVs; medians re-derived
  independently from the TSVs — exact match). Bonus: C3b found+bisected the pre-existing
  `from_linear_planar` sub-64 panic (missing reflect-pad; ≤63px planar refs died in the
  mean-offset pass since the entry point landed) — FIXED this commit with a
  fails-without-fix regression test (`m1_sub64_planar_precompute_scores_and_matches_
  interleaved`, planar-vs-interleaved pad-path agreement ≤1e-4).
- **#69 loop-steering study (jxl `d17cf7ce`; plan `5f7d16a3`, gates frozen): H3 MAGNITUDE
  steering is the one loop rule with value** — passes G1+G2 on v47A (t70 |err| 0.31 vs
  baseline 1.87 at bytes ratio 0.99; supervisor re-derived medians from the TSVs — exact)
  and is the only arm beating baseline on nonphoto on BOTH bakes; staleness free (G4) so
  the single-pass ≤1.1× endpoint stays viable for H3. The ratio-normalized family (C3b
  attr / H1-signed / H2-ctrl) NEVER beats the plain damped controller, and on shippedB-
  linear ALL arms fail G1. Mechanism: score-unit steps skip the normalization that erased
  the map's magnitude information. ZENSIM_H3_GAIN unswept (registered default 10.0) —
  a gain sweep is future work, not claimed. Hazard noted: unknown JXL_ZENSIM_MODEL_MAP
  values fall through to baseline silently (caught in-run by a control-arm mismatch).
  `jxl-encoder/benchmarks/zensim_attr_loop69_2026-07-29.md`.
- **★ APPEND2 COVERAGE FIX (2026-08-04, `299ccc8c`) — EVERY 944-era M3a MEASURED BEFORE THIS
  COMMIT IS TOO LOW.** `compute_attribution_density_full` sliced `s[720..min(len,924)]`, so on
  a **944** bake the whole append2/BANDVIS block never reached the density. Determination from
  the feature definitions (not the slice): `BANDVIS_GAIN/LOSS` (8 slots) are **class E** —
  plain means of a per-pixel `bounded_excess` indicator, the exact form v2 `HF_GAIN/HF_LOSS`
  already carry — so they were **real dropped coverage**; `LUMA_MEAN_REF` (reference-only) and
  `HL_BIN1/2` (HDR-gated on a structurally-SDR route) are **correctly zero**, now by an
  explicit named decision instead of an unreached bound. Measured shift on the 32-bake 944
  population (32 bakes): **M3a median +0.0487, max +0.1045, 30/32 up, 19/32 change tier —
  the GOLD (≥0.85) count goes **2 → 16**.** M3 unchanged (the legacy fold is untouched);
  372/720/924 unaffected by construction. Guarded by
  `attribution_covers_expected_slots_per_width` (probes every width, so a regime bump cannot
  silently drop a block again) + a plane-sum identity vs the production 944 features (8-9
  digits) + per-slot FD direction. Superseded numbers are registered in
  `benchmarks/eval_annotations.json`. Record: `benchmarks/attribution_append2_e1_2026-08-04.md`,
  registration: campaign appendix E.

### Trainer/eval capabilities added this campaign (all on main)
- **MANDATORY embedded repro**: every new bake carries `zentrain.repro` (inputs w/ sha256 +
  rows, seed, argv, trainer HEAD, `best_val`) via `zenpredict_bake::append_metadata_utf8`
  (section-splice, byte/score-identity gated). Embed failure = exit 4. bake_verdict emits
  `repro` (embedded > .spec.json > null+warning); dashboard badges it.
- `--coarse-l2-mult` / `--coarse-decay` (decoupled, post-Adam-step; the coupled form is a
  no-op under Adam — do not use it expecting effect). `ZENSIM_DECAY_DEBUG=1` telemetry.
- `bake_verdict`: sdr25 corpus; bands + per-codec dial curves + gates + `model` block +
  bootstrap `srocc_ci` + signed SROCC + `frac_negative` + `train_eq_val` in `--full-json`;
  `product_composite` = THE ranking composite (dashboard reads, never re-derives).
- `diffmap_block_coherence`: n_in==924 via the CANONICAL folded-append streaming extractor
  (extended path would inject untrained-weight noise into the structural-zero block);
  `ZENSIM_GRAD_MASS=1` gradient-mass diagnostic (region/v2-slot/basic-scale/top-idx);
  tie-correct midrank Spearman; dropped-mass printed for every layout.
- `run_full_eval.sh`: `ZENSIM_M3_REUSE=1` (schema re-emits carry M3 — a fulleval re-run is
  a cheap rescore over stored parquets; only re-measure when bake/parquets/fixtures change).

### Gotchas (bled for; do not re-learn)
- pq.write_table defaults SNAPPY → the Rust parquet reader has no snap; write eval grids zstd.
- f32-vs-f64 + clip-saturation traps in parquet key-joins (multiset join on
  `(source_id, f32(clip(s/100)))`).
- The JS dashboard template is a RAW Python string: `\'` becomes a literal backslash-quote
  and kills the whole <script> (blank page). Gate regen on `node --check` + the DOM-shim
  render harness (both in the pipeline now).
- pkill/pgrep -f self-match the invoking shell (locally AND over ssh); pgrep name-match
  truncates comm to 15 chars. Kill by PID; use pgrep -x with the truncated name.
- Observe-before-load on fleet nodes (node-2 7× incident = a live zensim-720 backfill
  worker, not a slow box).
- Trainer-bin globals must be set BEFORE the training call — the best_val relocation
  silently moved the regularizer setup post-training (caught by the decay-debug counter).

## EXTRACTION PERF + THE BUFFERED PATH — read before touching either walk (2026-08-30)

Full record: [`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`](benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md).

**Naming trap first.** `compute_multiscale_stats_streaming` (`streaming.rs:862`)
is the **BUFFERED** path — it materialises whole-image XYB pyramids; "streaming"
there means band-processing inside a scale. The genuinely memory-streaming v1
form is `compute_multiscale_stats_streaming_strips*`. The fold
(`foldapp_streaming_walk`) is a third thing. At least two sessions have been
misled by this.

**The fold WAS an extractor, not a metric — and as of 2026-08-30 it has a
score.** `ZensimV2Result` still has none, but
`Zensim::with_engine(ScoringEngine::Fold)` (`#[doc(hidden)]`,
`feature-regime-v2`-gated, default OFF) routes `compute`,
`compute_with_codec_hint`, `compute_extended_features`,
`compute_all_features`, `compute_with_ref` and `classify` through the fold and
returns a **BIT-IDENTICAL** `ZensimResult` — score, `raw_distance`, every
feature, and `mean_offset`. Gates: `zensim/tests/fold_engine_parity.rs` (11
tests, `to_bits()` equality over 18 geometries × {serial, rayon} × rayon pools
1/2/3/8/16) and `fold_backed_fixtures_match_golden` in `v1_golden_bytes.rs`
(the fold reproduces the PINNED golden arrays, not merely the buffered path).
Record: [`benchmarks/fold_engine_2026-08-31.md`](benchmarks/fold_engine_2026-08-31.md).

**Corrections to the paragraph this replaces**, all read from source:
`PrecomputedReference` is a pyramid cache, NOT buffered-walk state — the fold's
producer reads it directly, so the ref-cached form needed no new type;
`build_attribution_into_sink` calls only `crate::blur`, so
`compute_attribution_density*` was never walk-bound (the FUSED compare is —
`compute_zensim_streaming_with_ref_and_attr_{planes,fold}`); the width
divergence was closed by option C; and `zensim-gpu`'s oracle walk is now
selectable (`ZENSIM_GPU_ORACLE_ENGINE=fold`, zenmetrics `92bdec00`) with the
swap measured inert (103 pass / same 6 pre-existing wgpu failures either way).

**Buffered is still not removable, but for a DIFFERENT and now-gating reason:**
`feature-regime-v2` is not a default feature, so a default `cargo add zensim`
build contains no fold at all. Six entries also still route to buffered by
design — `compute_pu_linear*`, the fused attribution compare,
`compute_streaming_strips*`, weight-skipping linear profiles
(`PreviewV0_1`/`V0_2` `compute()`), any `with_stop` request, and any
`num_scales != 4`. The registered retirement proposal (prerequisites, deletion
order, what must survive) is §9 of the fold-engine note; **nothing has been
deleted and it awaits sign-off.**

**Parallelism is the structural asymmetry.** Buffered parallelises band-per-strip,
degree `layout_h.div_ceil(STRIP_INNER)` — it grows with image height. The fold
parallelises per channel, **degree fixed at 3**, over a producer with no rayon.
MEASURED 1T→8T: buffered **2–4×**, fold **1.1–1.5×**; 1T→28T buffered 4.6–6.5×,
fold 1.8–2.0×. Consequence for the SAME 216 v1 pool features: the fold is
**0.63–0.91×** buffered's cost serial, and **1.09–3.30×** at 8 threads. **Name a
thread budget before trading one path for the other.** zenmetrics dodges this by
running the fold `.with_parallel(false)` and parallelising across pairs.

**Per-pixel cost RISES with size in every arm** (buffered 25.6→31.3 ms/MP,
zeroed fold 49.2→63.7 across 576²→2304²), so an `α + β·pixels` fit returns a
NEGATIVE intercept — the linear model failing, not a fixed-cost saving. Never
quote a single "ms/MP" for these walks.

**The fold-vs-v1 372 divergence is entirely v1's mirror-padded columns.** v1
walks `simd_padded_width(w)` (`streaming.rs:871`) and mirror-fills the extras
(`streaming.rs:3185`); the fold walks `w`. `simd_padded_width` adds a further
16 whenever the 16-aligned value is ≥512 and an even multiple of 16, so
**512/576/1152/2304 are all divergent** — up to 81.6 % on a pool slot, and ~60 %
of real corpus rows. **Pre-pad the RGB input by the same reflect-101 rule and 17
of 20 geometries go BIT-EXACT** (gate:
`v1_padded_width_divergence_is_column_padding`), with no change to the 944
regime's pooling. Residual: three cells at **h = 93** at non-tight widths,
≤1.098e-6 — a pad-column × row-group-tiling interaction, not a width class.
**`--regime` note:** `folded720_v1_pools_match_v1_path` is
`#[cfg(feature = "training")]`; the plain
`--features custom-profiles,feature-regime-v2` invocation compiles it out
silently. Always include `training` when the pool block is in scope.

**OPTION C (user decision 2026-08-30): v1 stops pooling phantom columns.**
MEASURED before implementing, and it inverts two premises. (1) It is
BIT-EXACT: with no padding the fold and buffered agree to the bit at every
width tested — 127×93 went 1.739e-1 → **0.000e0**, 200×150 8.155e-1 →
**0.000e0** — so the h=93 residual reported earlier was an artifact of the
option-A PRE-PAD workaround and **does not exist under C**. (2) It is
CHEAPER, not costlier: buffered v1-372 Ir **−9.02 % at 576, −7.37 % at 1152,
+0.00 % at the tight-width control 592**. The lane-alignment padding is not
paying for itself. **Not flipped** — default untouched pending the era
rollout. ⚠ `v1_golden_bytes` fixtures are 64×64 and `simd_padded_width(64) ==
64`, so the golden set is entirely in the TIGHT class and is **structurally
blind to the defect C fixes**; a C rollout needs a non-tight golden
(200×150 or 576×96).

**Block-skipping: `V2NewFeatureToggles::v1_only`.** A 372-only request skips
every v2-era block AND its phase-A upstream (the four V-blur sweeps + the v2
activity chain). 249 M Ir vs buffered v1-372's 336 M (**0.743×**) and
944-full's 535 M (**0.466×**). Gate: `folded_v1_only_matches_full_walk`
(bit-identical emitted slots, skipped range asserted FINITE).

**MT: the fold saturated at exactly 3 threads and REGRESSED past it** (2.26×
@3T, then worse) — the channel fan-out is the whole of its parallelism. Band
parallelism inside the channel (4 bands/strip) lifts it: 944-full best 7.75 →
**6.67 ms (2.57× @8T)**, and at 8T the 944-full/944-zeroed marginal
**vanishes** (9.4 vs 9.4 ms paired). Bit-exact ONLY because the merge is
**sequential in band order** — `((0+b0)+b1)+…` reproduces the in-place loop;
a tree/unordered reduction would not (f64 addition is not associative).
Remaining cap: the serial `StripPlaneProducer`.

**dense_block_kernel is the MT ceiling and it is ERA-LOCKED — do not
restructure it without asking.** It is 23.2% of the 944-full walk and gets
3-way parallelism only. Band/row-partial merging is bit-exact ONLY when each
accumulator takes exactly one add per row, which needs `POOL_SIMD` (v4x-only)
AND `width % 8 == 0` at every scale. Neither holds generally — the
`weighted_pool_accumulate_scalar` call inside the x-loop (feature_v2.rs:2352)
and the `width8..width` scalar tail (:2416) both add per PIXEL across row
boundaries. MEASURED on the kernel's accumulation shape: 0 ulps at
`tail_k = 0`, **−2 ulps** with a scalar tail, **13 ulps** with per-pixel pools.
Amdahl UPPER bound on fixing it (dense scaling perfectly, nothing else
changing, restructure free): **1.17× @8T, 1.23× @16T** — against re-extracting
every 944 table AND re-training every 944 model. Not a silent trade.

**Y-channel imbalance has no free fix.** The 3 channel accumulators are
disjoint, so scheduling is already free and rayon work-steals; the imbalance is
that append/BANDVIS/CSFW are Y-ONLY WORK. Shortening it means splitting Y's
kernels, which lands on the same grouping obstacle. Pipelining X/B past the
barrier was analysed and rejected: X and B are the small channels, so
overlapping their phase B with Y's phase A shortens nothing.

**`fused_vblur_ssim` fission: RETIRED, no rewrite needed.** Its 4 stores / 28
spill loads look alarming but only **1 load and 0 stores are inside the
innermost loop** (188 insns of a 1750-insn function); the other 27 are
per-column-group setup. No hot-loop register pressure exists to relieve. Locate
spills against loop structure with objdump BEFORE proposing a fission.

**Two perf traps, both measured the hard way.** (a) **Fusion is not free**:
folding the activity abs-diff into the H-blur's load sites (registered lever
#1) LOSES — 944-full +1.04 %, 372-only +2.01 % — because post-rem-ring the
H-blur gathers one strided column and fusing makes it two, to save a cheap
contiguous pass. Reverted and deleted. (b) **Parallelism that allocates is
not parallelism**: `map_init(FoldPoolScratch::default)` re-allocates ~580 KB
per worker per strip per channel and made band-parallel 944-full *worse* at
3T (7.75 → 10.00); persistent per-band scratch slots fixed it. Spill audit:
`box_blur_h_inner_v4x` 1 store/2 loads (rem-ring is spill-clean),
`fused_vblur_ssim_inner_v4x` 4/**28** — the standing fission candidate.

**RSS shapes differ, and the crossover is ~1.5 MP.** Buffered scales with
AREA (3.5–3.8× for 4× pixels), the fold with WIDTH (2.1–2.2×). At 2304² the
fold's working set is 0.62–0.80× buffered's; at 1152² it is *heavier*
(62.8 vs 55.5 MB). Do not assume "streaming = less memory" at small sizes.

**Perf lever that shipped: the rem-ring.** Horizontal box blurs vectorise ACROSS
rows, so each x-step assembled a vector from 16 (or 8) strided scalar loads —
twice, add-side and remove-side. For every `x >= diam` those are the SAME column
(`x - r`), so half the gathers were redundant. A stack ring of the last `diam`
add-vectors removes them, bit-exactly. MEASURED (callgrind Ir, 576², v3 tier):
buffered v1-372 **−8.29 %**, zeroed-944 fold **−5.57 %**, live-pools fold
**−6.22 %**. Gates: three `to_bits()` reference tests in `blur.rs`, each with its
negative control run.

**Known wart, deliberately NOT fixed:** `fused_blur_h_mu_inner_{v4,v4x,v3}` use
`sum += add - rem` in their scalar tails and `(sum + add) - rem` in their vector
bodies, so tail rows differ from vector rows in the last ulp or two (2528.7349 vs
2528.7344). Pre-existing; `fused_blur_h_ssim` already fixed it via a masked
vector tail. Converting `mu` would move v1's shipped bytes — golden-gate policy,
not a drive-by.

**Profiling here:** `perf` is unusable (`perf_event_paranoid = 4` under WSL2
refuses even user-space events). Use callgrind on a `--no-default-features`
build (valgrind cannot execute AVX-512) — and remember the resulting profile is
the **v3/AVX2 tier** (CORRECTED 2026-09-01: archmage's own token docs —
`X64V3Token` = "x86-64-v3" = AVX2+FMA+BMI1/2, Haswell 2013 / Zen 1 2017; there
is no dedicated SSE4.2-only tier in this dispatcher's ladder — this line
previously said "v3/SSE", the same one-tier-off mislabeling
`benchmarks/profile_d_notax_2026-09-01.md` found and fixed in
`feature_v2.rs::harness_active_tier` and flagged, unfixed, in several other
campaign docs from 2026-08-28/30/31), so kernel *ratios* may shift at v4x. Put
the profiling `CARGO_TARGET_DIR` under `target/` — `.gitignore`'s `/target`
does NOT match a sibling `target-cg`, and 54 MiB of build output reached main
that way once.

### Profile D's gating tax is REMOVED — `feature-regime-v2` is now default-on (2026-09-01)

Full record: [`benchmarks/profile_d_notax_2026-09-01.md`](benchmarks/profile_d_notax_2026-09-01.md).

**The fold engine + `V1PoolsMode`/`V1FreeExtras` (everything this section
describes) is in `zensim`'s `default` feature list as of 2026-09-01.** Before
this, `ZensimProfile::D`'s `fast_by_default` wiring (`Zensim::new` sets
`fold_engine = true, skip_unread_pools = true` for `D` unconditionally) was a
no-op on a plain `cargo add zensim` build — the fields it sets are only read
inside `#[cfg(feature = "feature-regime-v2")]` blocks, and that feature
wasn't default, so `D` silently fell back to the buffered (`B`-class-cost)
walk on any build that didn't explicitly opt in. That was the W7 gap
("reachable by a default build") multiple campaigns registered and didn't
close. `--no-default-features` still removes the whole module family; every
profile including `D` still scores correctly there, just without the fold's
speed. Adds zero dependencies (`feature-regime-v2 = []`) — the cost is
default-build compile time (the ~18k-line `feature_v2.rs` plus siblings now
always compiles), not a runtime behavior change for any profile other than
`D`. A genuine module split (carving the v1-only fold subset out of the
v2-bounded machinery it's interleaved with, so a default build pays only for
what `D` needs) remains a larger, separately-scoped refactor, named but not
attempted here.

New gate: `fold_engine::skip_policy_tests::
default_build_profile_d_matches_feature_gated_off_buffered_walk` — proves
score/`raw_distance`/`mean_offset` and the entire SCORED `f0..228` prefix
bit-identical between default-build `D` and a forced-`Buffered` proxy, while
explicitly asserting the two arms' `f228..372` (masked/IW) LEGITIMATELY
differ — the skip optimization zeroes it, the buffered walk always computes
it, and both are correct. An earlier draft asserted full-372-vector equality
and correctly failed at `f228`; see the test's own doc comment (and the
benchmarks doc §1.4) for the corrected claim and why it's the honest one.

**The free-set raw-moments accumulator (`V1FreeExtras::RawMoments`,
`fused.rs`) was hand-duplicated at 6 vector SIMD sites + 4 scalar-tail sites
(one per tier: `_v4`/`_v4x` native-16-wide main loops + their 8-wide
`token.v3()` remainder loops, `_v3`'s native 8-wide main loop, the
`#[magetypes(neon, wasm128, scalar)]`-generated function's 8-wide main
loop) — now two generic helper pairs
(`raw_moments_accumulate{8,16}`/`raw_moments_finish{8,16}<T: F32x{8,16}Backend
+ Copy>`) plus one scalar pair, `#[inline(always)]` (verified with `nm` —
zero `raw_moments_*` symbols survive in a release binary; see
`dense_block_kernel_generic`'s own comment for why `#[inline(always)]` and
not something weaker is mandatory here — a 5.3× regression from an
un-inlined generic SIMD helper, already measured once in this file).
**`#[rite]` does not apply**: it resolves `#[target_feature]` from a
concrete token parameter or explicit tier-name arguments
(`archmage-macros/src/rite.rs`), and these helpers are generic over a
*backend trait*, not a concrete token — there is no single tier for `#[rite]`
to attach to a shared body. This is also the "add a new free slot at ~zero
marginal EFFORT" extension point the free-features doc's own hand-off asked
for: a future `V1FreeExtras` variant's per-row step is one function to add,
not up to 10 hand-copies to keep in sync.

**Tier-naming correction, made from archmage's own source, spread across
this repo's own docs and one bit of shipped code:** `X64V3Token` =
`"x86-64-v3"` = **AVX2+FMA+BMI1/2** (Haswell 2013 / Zen 1 2017), NOT SSE4.2 —
there is no dedicated SSE4.2-only tier in the `v4x/v4/v3/neon/wasm128/scalar`
ladder. `X64V4Token` = AVX-512 baseline; `X64V4xToken` = AVX-512 +
VBMI2/GFNI/VNNI/etc. `zensim/src/feature_v2.rs::harness_active_tier` reported
`"v4 (AVX2)"` / `"v3 (SSE4.2)"` — both one tier off — fixed to `"v4
(AVX-512)"` / `"v3 (AVX2+FMA)"`. Several 2026-08-28/30/31 campaign docs
(`benchmarks/balance_campaign_2026-08-28.md`,
`benchmarks/era2_perf_break_2026-08-31.md`,
`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`) carry the
same "v3 = SSE4.2" mislabeling describing fleet hardware (e.g. "i134 …
v3 (SSE4.2)") — flagged here, **not edited**: correcting historical
measurement-hardware labels in someone else's campaign record needs
verifying what CPU tier that box actually ran on, which is out of this
lane's scope.

**Both tiers are now first-class in the named speed instrument.**
`zensim-bench/benches/ssim2_speed_bar.rs` gained `ZEN_S2_CAP_V3` (disables
`X64V4Token` process-wide via `testable_dispatch`, cascading to `X64V4xToken`
— confirmed from archmage source, not assumed — leaving `X64V3Token`/AVX2 as
the ceiling) and `ZEN_S2_EXTRACT_ONLY` (adds `add156_extract_only`/
`free156_extract_only` arms — same entry point as the real arms, minus the
`Predictor` forward pass, to separate extraction cost from forward-pass
cost without a second, potentially-mismatched instrument). See the
benchmarks doc for the measured 1T/8T/16T × 576²/1152²/2304² × both-tier
table and the W4 1152²@8T diagnosis (reproduced; not primarily a forward-pass
effect; does not reproduce as an isolated single-arm cost, implicating
zenbench's round-robin rayon-thread-pool interleaving over a fixed defect in
the walk's own code — diagnosed, not silently patched with an unverified
fix).

## SEED SPREAD IS NOT SUBSET COVERAGE — and `--seed` used to fuse two streams (2026-09-04)

Full record: [`benchmarks/subset_quality_study_2026-09-04.md`](benchmarks/subset_quality_study_2026-09-04.md).
Registry: [`benchmarks/good_subsets_registry.json`](benchmarks/good_subsets_registry.json).

**The trainer runs TWO independent `SplitMix64` streams**, and always has:
init from `SplitMix64::new(seed)`, sampling from
`sampling::sample_stream_seed(seed)` (`seed*0x9E3779B97F4A7C15 + 0xDEADBEEFCAFEBABE`;
the per-sample-α entry point adds `0x0123456789ABCDEF` instead). The split is
deliberate — it exists so a 228-vs-372 A/B sees the same pair draws even though
init consumes a different number of normals. **The CLI did not expose it until
2026-09-04, so every bake trained before then confounds "different subset" with
"different init" and NO board bake can separate them.** `--init-seed` /
`--sample-seed` now override each stream; omitting both is byte-identical.

**A drawn training subset is reconstructible from a bake's `zentrain.repro`
alone.** The drawn multiset is a pure function of `(seed, [train_w], [rows],
epochs, pairs_per_epoch, boosts, within_ref)` — not of the feature matrix, the
architecture, or the loss — and `repro.inputs[].rows` IS the `n` the sampler
draws modulo. `subset_sim --fulleval <x>.fulleval.json --seeds N` replays it
through the SAME owner the trainer uses, reading only the target + ref columns
(`parquet_loader::load_scores_and_refs`), no features. Faithfulness is provable,
not asserted: `ZENSIM_SAMPLE_DIGEST=1` makes a real run print the same
sample-sequence hash, verified equal on the uniform, within-ref and q-boost
paths.

**`mlp_train::sampling` is THE owner of the pair-draw step** (four training loops
+ the replay). Its RNG consumption is a WIRE CONTRACT: 1 value for the group,
then 2 (uniform / per-row-CDF) or 3 (within-ref), with an early return after the
group value when the group has <2 rows. Changing it re-rolls every model ever
trained — an era break, which is why `simulate_digest_is_pinned` says so in its
own failure message.

**MEASURED, and it kills a tempting research direction: subset coverage does NOT
explain seed-to-seed spread.** At production settings (120 x 50,000 draws over
~700k rows = ~17 hits/row) coverage is SATURATED: over 66 seed-sibling arms the
between-seed *relative* spread of pooled row coverage is 1.37e-4 (median), of
reference coverage 1.31e-5, of row-multiplicity entropy 3.85e-6 — 12 of 29
descriptors never exceed 1 % on any arm — while the targets move 28x to 940x
more (within-image CID22 3.9e-3, KonJND 1.3e-1). **Do not tune the pair sampler
hoping to cut seed variance at these settings.** The untried levers that act on
the trajectory rather than the multiset are `--ema-decay` (its own doc claims
"seed-variance reduction"; NO board bake sets it) and seed averaging.

**METHOD LESSON — carry a pure-luck control descriptor.** The study's strongest
correlate of within-image CID22 across the whole board was its CONTROL: the rate
at which the RNG drew the same row index twice (rho +0.3707, n=119), which
carries no information about the training data at all, and whose own permutation
null called it significant at p=0.002. Real coverage descriptors topped out at
0.3380. Mechanism, MEASURED: the 202 board cells hold only **132 distinct subsets over 51
seeds**, and 88 cells share a subset with another cell — one subset is the
training subset of 22 different cells. Arms sharing a group structure and a seed
draw the IDENTICAL pairs in the identical order, so a seed-pure descriptor takes
one value across all of them and permuting *within* arm leaves the cross-arm
dependence intact — the null is under-dispersed and every p-value is optimistic. **A within-arm-rank correlation pooled over arms
that share seeds needs a control that cannot matter; without one, |rho| ~ 0.3 at
"p<0.01" is not evidence.**

**Seeds are not transferably good.** Ranking seeds within arm and testing
per-seed mean-rank consistency (2,000 permutations, 31 arms): p = 0.205 / 0.035 /
0.446 / 0.587 on CID22-per-ref / CID22 / KonJND / AIC-3 — one nominal hit out of
four, with orderings that disagree across targets (seed 4006 is 2nd best on CID22
and 3rd worst on AIC-3). A "lucky seed" is arm-specific. Best-in-arm seeds are
also NOT better covered (max mean z 0.467, with the pure-luck control third at
0.428).

**A WHOLE KNOB FAMILY was silently discarded off the per-sample-α path.**
`--ema-decay`, `--hard-pair-frac`, `--dro-eta`, `--listwise-weight` are read ONLY
inside `train_mlp_per_sample_alpha_head`; on the standard path (all 312 board
bakes) they were accepted and thrown away, producing a bake byte-identical to a
run that never set them. `train_mlp_strategy` already guards `--monotonicity-reg`
/ `--mse-weight` / `--triplet-weight` against exactly this; these four were never
added. They now **fail loud** (2026-09-04). NOT wired through — an EMA/DRO/
listwise step in the other three loops is a real optimizer change. **So the
obvious seed-variance lever is currently UNAVAILABLE on the standard path**: if
you want EMA there, wiring it is the work. No board bake sets any of the four.

**`--stratified-bands` was a SILENT NO-OP on the default path until 2026-09-04.**
`strat_bands` was built in one of four training loops; the other three —
including the standard path all 312 board bakes trained through — passed an empty
table, so the flag was accepted and discarded (bands 0 vs 8: same digest
`127b831bed8a3873`). Fixed, default byte-identical, failing-first test. No board
bake sets it, so nothing published is affected. **Whether stratified sampling
HELPS is unmeasured** — it is merely reachable now.

## LATENCY + TOKEN DISCIPLINE — idle waiting is re-charged, not cached (2026-08-04)

**MEASURED, `benchmarks/rnd_cycle_audit_2026-08-04.md`.** Over the 2026-08-03/04
campaign (34.3 h, 11 waves): **14.80 h of whole-session idle, 6.77 h of it dead**
(nothing computing, or finished work sitting unharvested), and **$395.24 —
13.9 % of the $2,837.34 session — spent re-creating prompt cache that idle
waiting had expired.** The mechanism: cache entries are `ephemeral_5m`, read at
0.1× and written at 1.25×, so **any gap over 5 minutes converts the whole
prefix from a 0.1× read to a 1.25× write — a 12.5× multiplier** on 500–800 k
tokens. 138 turns (3.7 % of all turns) carried 22.9 M re-created tokens =
**55.7 % of every cache-write token spent that day**. One agent idled 7.92 h of
its 11.69 h span and alone burned 12.15 M cache-write tokens; its worst single
turn re-created **779,717 tokens after a 141-minute gap**.

Rules, all load-bearing:

- **Do NOT park on short-interval polls.** Polling is not what costs — the polls
  read a warm cache (57 wait-turns = 626 k write vs 21.2 M read). The **wake-up
  after a long silence** is what costs. So the fix is never "poll less often";
  it is "do not be idle-attached at all".
- **Arm ONE terminal condition, then go do other work.** `Monitor` a file that
  appears exactly once — `scripts/await_artifacts.sh --heartbeat X` writes
  `X.done` on **every** exit path (COMPLETE / TIMEOUT / SIGNAL + rc). Never
  `Monitor` a `tail -f` (loses the file on rotation) and never hand-roll a
  `while sleep` waiter: the two worst events of the day, **125.6 min** and
  **80.6 min** of dead wall-clock, were both a bespoke waiter that stopped
  without leaving evidence.
- **Supervisors must not idle-wait on delegated work.** A supervisor watching a
  subagent ages *two* prefixes, so one event re-charges both. Do independent
  work; let artifacts be the channel.
- **Make a late wake-up free.** Harvest on completion — `scripts/harvest_bakes.sh`
  verdicts + fullevals each bake as it lands, so results are already on disk
  when anyone next looks. A post-bake hook MUST fail loud: the coherence wave's
  hook exited 2 nine times into an unread log and silently voided a 3 h 24 min
  lane (21 verdicts re-run by hand, 804 s).
- **Batch status into one report.** Do not emit a turn per artifact; one report
  per terminal event.

Wave skeleton + the priced anti-pattern table: [`docs/WAVE_PLAYBOOK.md`](docs/WAVE_PLAYBOOK.md).

**Not a cause, measured and rejected:** per-agent `CARGO_TARGET_DIR` cold
rebuilds. Total `cargo` wall-clock across every agent all day was **23.0 min**
(91 builds; cold `bake_verdict` = 72 s / 221 crates), while a second concurrent
`cargo` on a *shared* target dir **blocks 31.8 s** on the build lock. Keep
per-agent target dirs; agents that only consume binaries should build nothing
and use the `ZL_BV` / `ZL_TRAIN` / `CARGO_TARGET_DIR` pointers the drivers
already honour. The real target-dir cost is disk — 28 dirs, 113.6 GB, root at
95 % — so delete yours when a wave closes.

## PERF MEASUREMENT: the noise floor at 2304² is 10 %, and it is ASLR (2026-08-31)

**MEASURED, `benchmarks/era2_perf_break_2026-08-31.md` §22.5.** The 944 walk at
2304²/1T, same binary, same environment, CCD-pinned, min of 11 walks per
process:

| | 8 process starts | spread |
|---|---|---:|
| **ASLR off** (`setarch -R`) | 363.22 363.39 363.53 363.22 363.16 363.49 362.98 363.80 | **±0.13 %** |
| **ASLR on** | 335.48 335.11 357.60 340.53 328.81 361.47 361.97 357.51 | **10.1 %** |

The distribution is **bimodal** (~334 or ~360, rarely between), not noisy. The
~13 strip planes are each `2304 × 148 × 4 B` = exactly 333 pages at a fixed
relative stride, so the mmap base decides whether the streams conflict. Ruled
out by measurement: THP (`madvise` mode, `AnonHugePages: 0 kB` in **both**
states), a 0–512-page heap-base shift (327.29–328.27 ms flat), per-plane
staggering from 64 B to 64 KiB (326.95–328.54 ms flat), and CCD placement.

**Consequences — treat these as rules, not advice:**

- **Any single-process 2304² perf number is ±10 %.** That includes numbers
  already published in this repo's perf docs. Interleaved multi-process
  comparisons survive; single before/after pairs do not.
- **A before/after across two BUILDS cannot be trusted at all below ~10 %** —
  any edit reshuffles the binary's own layout by about that much. Put the arms
  behind a **runtime** flag in ONE binary.
- **The environment block is a layout input.** Adding an env var that provably
  does nothing flipped one build from 359 → 328 ms. Keep env values the same
  BYTE LENGTH across arms (`ZENSIM_X=032`, not `=32`); never select an arm by
  the *presence* of a variable.
- **The protocol:** one binary + runtime arms → identical-length env → arms
  interleaved → **min of N walks in a process** (kills interference) → **min
  over ≥15 process starts with ASLR on** (kills layout). Carry a
  **bit-identical control arm** when one exists; if your estimator reports the
  control as faster than the thing it is identical to, the estimator is not
  sound yet. `setarch -R` is a fast second opinion on one arbitrary layout,
  never the primary.

Instrument: `zensim/examples/foldapp_stream_bigpair.rs` —
`ZENSIM_BIGPAIR_TOGGLES=944full|924|372`, `ZENSIM_BIGPAIR_ITERS=N` (median +
min + `smaps_rollup` Rss/AnonHugePages), `ZENSIM_BIGPAIR_PARALLEL=1`,
`ZENSIM_BIGPAIR_DUMP=<path>` (every feature with its `to_bits()` — the
bit-identity control).

### A SECOND, LARGER 2304² noise source: zenbench itself can degenerate under a tight wall-time budget, and own-process CPU contention is real even niced (2026-09-01)

**MEASURED, `benchmarks/profile_d_notax_2026-09-01.md` §4** (the Profile-D
no-tax refactor's own both-tier sweep). The ASLR noise documented above
(±10%, bimodal, ~334/360ms) is a property of the WALK. This is a DIFFERENT,
much larger-magnitude problem in the MEASUREMENT HARNESS: a `zenbench`
`compare` group given too little `max_wall_time` for the arms/sizes it's
asked to run can silently report a **spuriously near-zero mean for every
arm in the group simultaneously** — not a crash, not a warning, a plausible-
looking small number that is off by 100-1000×. Found because `fast_ssim2`
(a single-threaded C++ arm whose cost has nothing to do with zensim's own
threading) read exact `0.0` or ~0.2-1.6 ms in the bad rows, against
380-900+ ms in neighboring good rows at the same cell (2304², where a
single round's slowest arm alone costs 400-900 ms). At `ZEN_S2_WALL_S=8`
(sized for 576²/1152², both fine), a 6-arm group at 2304² blew the wall
budget before completing `min_rounds` for every arm, and the reported mean
degenerated rather than gracefully returning a partial-but-honest number.
**Consequence for anyone reusing the min-over-starts protocol above: min()
is only safe against noise that is one-directional (contention can only
ADD time). A harness that can spuriously report LOW is a different failure
mode entirely — min() will happily select the corrupted reading as "the
best one" and hide the defect.** Fix: scale `max_wall_time`/`ZEN_S2_WALL_S`
with image size (this repo's own re-run used 60s at 2304² after 8s produced
a 0-of-9-valid cell), and **validate every reading at collection time**
(a stable reference arm like `fast_ssim2` reading below a plausible floor
for its size is a free, cheap sanity check) rather than trusting a
zenbench summary line to always be honest under a tight budget.

**Own-process CPU contention corrupted the SAME tier's data independently
of the above.** During the flagged sweep's `v4x`-tier window, several
`cargo build`/`cargo test` invocations ran concurrently via
`~/work/zen/scripts/run-heavy` (`nice -19`, job-count-capped) while the
`taskset`-pinned sweep was running on the same physical cores — `nice`
lowers scheduling PRIORITY, it does not GUARANTEE isolation from a
`taskset`-pinned process sharing the same core set, especially under a
job-count cap that doesn't itself account for cores another process has
pinned. Result: even in cells with no zenbench-degeneration corruption,
`fast_ssim2` (which should be stable and thread-count-independent) swung
**128.9-633.6 ms within one 9-start cell** — whole-machine-load variance
riding along with every arm together, clustering into a visibly bimodal
"contended" / "clean" split. `min()` over enough starts still recovers the
clean cluster IF the clean cluster is well-represented and not itself
corrupted by the harness issue above — but the honest fix is **do nothing
else on the machine for the duration of a taskset-pinned sweep**, full
stop, regardless of how "cheap" the concurrent work seems (a single 5-10s
`cargo build` is enough to contaminate several nearby sweep invocations at
this box's per-invocation cadence).

## PERF: on the strip walk, TILE MEANS PACK — restricting a loop range does nothing (2026-08-31)

**MEASURED, `benchmarks/era2_perf_break_2026-08-31.md` §23.** The phase-A
fused H blur's cost is a function of **image WIDTH, not pixel count**: at a
fixed 5.31 MP and 1T it costs **104.99 ms at width 2304 and 34.58 ms at width
1152** (3.4×), flat below that. It holds 16 rows × 6 planes, which is 884 KiB
at 2304 against a 1 MiB L2. Column-tiling it with **packing** — stage the tile
plus a `±BLUR_RADIUS` halo into a compact `rows × (tile + 2R)` buffer, blur
there, copy the interior back — is worth **1.15× at 5 MP and 1.73× at 21 MP on
the whole 944 walk** at 1T (`blur_h` itself 1532.7 → 179.6 ms, **8.5×**), and
1.23×/1.11× at 4608² on 8/16 threads.

**The load-bearing lesson: the win is the PACKING, not the tiling.** Threading
an output column range `x0..x1` through all 16 H-blur bodies — so the kernel
tiles *in place* on the full-width planes with no copies — was built,
byte-neutral (369/0 including the v1 golden gates), and **bought nothing**
(1.06× at 1T/2304², **0.96× at 1T/4608²**) against the packed form's 1.26×/1.71×
in the same binary. Restricting `x` does not change which cache lines are
walked: the planes are still full-width, so a 16-row group at tile width is
sixteen contiguous runs `width × 4 B` apart and the prefetchers see the same
six strided streams. Locality comes from the **layout**, not the loop bounds —
the packed-GEMM result. The x-range refactor was deleted rather than parked.
**Reach for "pack the tile", never "restrict the loop".**

Two corollaries, both measured the hard way:

- **The other axis does not work.** Row banding fails on the same pipeline
  because the phase-A halo closure is `±2·BLUR_RADIUS` (activity =
  `blur(|src − blur(src)|)`), i.e. **20 rows out of a 32-row band = 62 %
  redundancy**; the column closure is `±BLUR_RADIUS`, so a 1536-wide tile
  re-blurs **0.6 %**. Band-local phase A measured +15.6 % / +5.5 % (§22).
- **Carry an identical-code-path control.** The tile does nothing when
  `width <= tile`, so 576²/1152² cells run the *same code* in both arms and
  must read 1.000×. What they actually read is the measurement's noise floor:
  **±0.3 % at 1T, ±1.8 % at 16T, up to 6.5 % at 8T.** Any threaded cell inside
  that band is unestablished, not a result — a 0.891 × "regression" published
  earlier in the same session evaporated under it.

## ⇒ POST-COMPACT / NEW SESSION: read [`SESSION-RESUME.md`](SESSION-RESUME.md) FIRST

Then return here. `SESSION-RESUME.md` is the canonical entry point —
it points at every other doc + lists the current critical-path
tasks. Reading order on resume:

1. [`SESSION-RESUME.md`](SESSION-RESUME.md) — current state, ~1 min
2. [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md) — **the validated
   science + exact reproduction of the top models + the new-model loop + the
   pitfall list.** THE entry point for any model work (2026-07-18).
3. [`docs/MODEL_SELECTION_SCORECARD.md`](docs/MODEL_SELECTION_SCORECARD.md) —
   the five-gate exam (RANK/DIAL/STEER/RD/TARGET) every ship candidate takes.
4. This doc (`CLAUDE.md`) — methodology + workflow + gotchas. **NOTE: the V0_x /
   PreviewV0_5-era historical sections were excised 2026-07-19 to
   [`docs/HISTORY-2026-05-v0x-era.md`](docs/HISTORY-2026-05-v0x-era.md); the
   cookbook supersedes them for current state.**
5. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — **the traceability
   spine**: how a number chains back to bytes (verdict → bake sha → manifest →
   input shas → trainer commit), which gate enforces each link, and the honest
   list of gaps. Read before making or citing any measurement.
6. [`RESEARCH.md`](RESEARCH.md) — corpus map + workflow recipes
7. [`benchmarks/INDEX.md`](benchmarks/INDEX.md) — find prior experiments
8. Run `TaskList` and work on the lowest unblocked task

(CONTEXT-HANDOFF.md is DELETED — handoff files are banned; durable facts live in
the docs above. The IQA literature corpus is `~/work/zen/zenpapers` — search it
before designing features or metrics.)

## RECURRING PRIORITIES + ASSETS — do not re-forget (consolidated 2026-07-16)

The user has had to repeat these across the last week. They are load-bearing;
re-search + honor them EVERY session. This section exists because they kept
getting lost (wrong dashboard picked twice, HF parquets forgotten, bigcodec
metrics assumed absent, negative-value + diffmap-coherence requirements dropped).

### The product = a CONSISTENT DIAL (not just a ranker)

Users type a target zensim; the codec tunes to hit it, using the diffmap to close
the loop. Every metric decision serves this:
- **Monotone in codec quality** (so target-hitting converges) + **bounded [0,100]**.
- **NEGATIVE zensim values MUST work** — inputs worse than the worst codec output
  score BELOW 0 (do NOT clamp at 0; the lower spline extrapolation + profile
  `extrapolate_score` carry it). Negative-tail training data:
  `canonical-2026-07-15/train/kadis_negrich.parquet` (negative-rich).
- **The diffmap MUST match the scalar** — `DiffmapResult.diffmap()` must reflect
  the SAME model as `.score()`, so the per-block "where to adjust" signal drives
  the closed loop. Currently INCOHERENT (diffmap uses per-scale SSIM weights,
  scalar uses the 372-feat model) — this is the #1 closed-loop blocker.

### Evaluation north stars (priority order)

- **ssim2 is the best north star for NON-PHOTO content** (user directive).
  `imazen26` (real-codec ssim2) + `nonphoto` (non-photo ssim2) are FIRST-CLASS
  gates in bake_verdict (G-IM26, G-NP). Eval every ship-grade bake on them.
- **CID22** = gold human-MOS holdout (validation only). CID22 trades are user-gated.
- **HF near-lossless is the metric's WEAK ZONE** — high-fidelity / near-lossless
  (B8/B9, q75-100) is where compression product decisions live AND where every
  learning metric is weakest. Always eval AND train it.

### Data assets that keep getting forgotten

- **HF near-lossless parquets**: `canonical-2026-07-15/train/hf_nearlossless_{train,val}.parquet`
  (900 + 300 rows × 372 feat). The `hf_nearlossless` corpus in bake_verdict.
  INCLUDE in training + eval — **but it is an ssim2 SELF-TARGET, not human data.**
  MEASURED 2026-09-01: `human_score` **is** `ssim2_gpu / 100`, exactly, in float
  equality, on **1200/1200 rows** — they are ONE column, not two targets, and this
  bullet said otherwise for six weeks. So a score on this axis is AGREEMENT WITH
  ssim2, never a win over it (`peer_ssim2` reads pooled SROCC **1.0000**, per-ref
  mean 1.0000); it belongs with `nonphoto`/`imazen26`/`hfnlproxy` in the
  circularity-excluded set. It is also **372-only and NOT re-extractable**: the
  1,200 distorted JXL bitstreams were never persisted (`encoded_filename` blank on
  1200/1200 `pareto.tsv` rows; both `refit/distorted/` mirrors empty), so no wider
  regime can ever read it. For a NON-circular near-lossless read use
  `hfnl_cid22band` — the top MOS band of CID22 (n=1425/49 refs), where ssim2 scores
  0.5058 pooled / 0.7099 within-image. Record:
  `benchmarks/ssim2_replacement_bar_2026-08-31.md` APPENDIX A.
- **Negative-rich data**: `canonical-2026-07-15/train/kadis_negrich.parquet` — the
  negative-dial-tail training corpus.
- **bigcodec's cvvdp/iwssim ARE backfilled** (the depth-iter
  `bigcodec_train_120k_stride.parquet` is ssim2-only, but the metrics exist
  elsewhere). CONCRETE (audited 2026-07-16): bigcodec cells WITH `score_cvvdp` +
  `score_iwssim` (99.6%/100% non-null, feature-space, f0..f371) live at
  **`/mnt/v/output/zensim-multicodec-probe/bigcodec_mm6_traindigits_2026-07-02.parquet`**
  (1.56M rows; 6 codecs + jxl-hqfill, NO avif — "mm6"). Authoritative per-encode
  metric sidecar (all 6 codecs, 4.18M rows, key=`encoded_filename`/`encode_sha`):
  `/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_patched_2026-07-02.parquet`
  + JXL near-lossless top-up `hqfill_7metric_sidecar_2026-07-02.parquet`. The
  120k-stride file dropped `encoded_filename`, so the fallback fingerprint to
  rejoin it is `(ref_basename, round(human_score,9), round(f0..f2,9))` (per
  `scripts/v_next/linear_projections_2026-07-03.py:1064`). Column names differ:
  mm6 uses `score_cvvdp`/`score_iwssim`, canonical uses `cvvdp_score`/`iwssim`.
- **CVVDP/IW-SSIM mix training targets**: safesyn/kadid/tid/cid22_train carry
  `iwssim`, `cvvdp_score`, and `mix_cv{25,50,75}_iw{75,50,25}` (all positive-
  direction, [0,100]). Use them (`train_minmax --synth-target`, or the trainer's
  `--target-column`) to recover CID22 from the ssim2-shaping bias. bigcodec/kadis
  need the sidecar join first. NOTE: pure-cvvdp-SCALAR target is a known dead-end
  ([[feedback_cvvdp_scalar_target_dead_end]]); a MIX / IW-SSIM is the ask.
- **High-quality-zone HUMAN data (untapped, local)**: JPEG-AI-SDR25
  (`/mnt/v/datasets/jpeg-ai-sdr25/`, 95k triplets, q75-100), AIC-3 raw triplets
  (`/mnt/v/datasets/aic3-btc-ptc/`, 420k), AIC-HDR2025 (`/mnt/v/datasets/aic-hdr2025/`).

### THE dashboard (the "pretty one" the user means)

`scripts/v_next/bandwise_dashboard.py` — the every-graph dashboard. THIS is the
combined dashboard; EXTEND it, don't rebuild a thinner one. Three modes:
- `--bakes label:path.bin,...` — compare bakes directly (shipped B auto-prepended;
  ssim2/cvvdp/butteraugli refs auto-added).
- `--from-search /mnt/v/output/zensim/reports/blend/blend_results_r7_2026-07-15.json`
  — the blend-candidate view.
- **`--fulleval-dir /mnt/v/output/zensim/reports/fulleval` — the INTERACTIVE
  summer-gauntlet** (2026-07-26). Reads pre-computed per-bake `*.fulleval.json`
  (schema + fixtures: `scripts/v_next/make_stub_fulleval.py`; ordering from
  `best_per_day.json`) and emits ONE self-contained, **OFFLINE** HTML
  (`--out …/summer_gauntlet.html`) via `scripts/v_next/gauntlet.py`: bake-toggle
  checkboxes (stable per-bake color; all/none/top-6), a **sortable scoreboard**
  (click any header — CID22/KonJND/dial-mono/M3/corruption/composite/…), a
  cross-corpus SROCC heatmap, the CID22-vs-{nonphoto,KonJND} trade map, and the
  **correlation SCATTER MATRIX** — predicted vs each reference (MOS/JND/ssim2/
  butteraugli/cvvdp), one clean faceted scatter per (bake × corpus) with OLS fit +
  canonical SROCC/PLCC. Hand-rolled inline SVG+JS (no CDN/plotly — opens offline),
  theme-aware (light/dark), dataviz-validated palette. **Stats are NEVER
  hand-rolled**: SROCC/PLCC come from the fulleval JSON's `scatter` block (eval
  agent → canonical `panel`) or, if omitted, `scripts/lib/zen_stats.panel` at build.
  Plus (2026-08-01) the **JXL loop-targeting panel**: `2shot/3shot ±2` scoreboard
  columns (emit-best, mapped bakes only via `gauntlet.LOOP_BAKE_MAP`) + its own
  section table (all loop models incl. emit-last, the outer arms and ssim2, which
  are not bakes), fed by the jxl-encoder sweep summary via `--loop-targeting`
  (default = the committed
  `~/work/zen/jxl-encoder/benchmarks/zensim_loop_23shot_summary_2026-08-05.json`,
  which carries every 2026-08-01 entry probe-verified PLUS the sota944 candidate
  `W10L9_base` — campaign appendix M; section omitted with a loud note when
  absent). Counts/medians are READ from that JSON, never re-derived (the
  jxl-encoder analyze script is the owner).
  Plus (2026-08-03, **REBUILT 2026-08-06 by appendix V**) the **cross-bake
  per-band SROCC table** under the Mohammadi panel's band bars. Bands are now
  cut by `zensim_validate::bands` (scheme `merged-decile-2026-08-06`) — fixed
  deciles merged until every band holds **n ≥ 1000 pairs spanning ≥ 0.08** — so
  CID22 shows THREE bands (`B0-B6` / `B7` / `B8-B9`), TID two, and CSIQ + LIVE
  one NOT-MEASURED band each because they are too small to band at all. Values
  are **SIGNED** (`srocc_signed`): a negative band draws below the axis and
  renders red, instead of being hidden by an absolute value that used to make a
  more deeply inverted band score HIGHER. Headers carry each band's `n` AND its
  `span`; a NOT-MEASURED band renders as an em-dash with its reason on hover —
  never as a zero. Cells cut on the pre-2026-08-06 fixed deciles are EXCLUDED
  with a visible count (their `B9` is a different quantity), and cells whose
  band edges disagree are refused. Read down a column, never across one: band
  SROCC is range-restricted. All values are read from `rank.<corpus>.bands[]`;
  nothing is recomputed.
  Plus (2026-08-30) **ERA ROWS**: a board name ending in `gauntlet.ERA372_CUR_SUFFIX`
  (**`@cur372`**) is the SAME BAKE as its unsuffixed sibling, read on the
  current-extractor 372 root instead of the 2026-05-15 one — same stem so the pair
  sorts together, `@` in no other board name so the suffix test is unambiguous.
  Promoted by `scripts/promote_era372_board.py` (a caller of `promote_fulleval.py`,
  with a never-overwrite gate on the stored halves); `family_of()` returns
  `"@cur372 (current extractor)"` for them, checked FIRST. **`family_of()` is not
  only a toggle label** — `build_html`'s knob-end gate scopes its peers/HDR
  exemption on it, so any rule that judges the MODEL rather than the ruler must go
  through `gauntlet.era_base_name()`. Read a pair DOWN, not across: only shipped
  B's pair is a clean era A/B (7 of the 9 historical "stored-era" rows are
  `--regime 720` ext720 reads — `benchmarks/board_era_rows_2026-08-30.md` §2,
  registry `board372-row-read-on-ext720-root-2026-08-30`).
  Plus (2026-08-04) **ENSEMBLE rows**: a fulleval JSON stamped `model.kind:"ensemble"`
  by `scripts/promote_fulleval.py --members` (the generalized promoter — it publishes
  ANY verdict, single-bake or ensemble, onto the board and recomputes NOTHING — every
  stat block is asserted byte-identical to the source verdict, whose path+sha256 land
  in `source_verdict`; `--strip-per-pair` implements the registered board-size rule,
  `--graft-into` folds a `*_corrjoint` verdict's corruption_head under the plain name)
  renders an `ens×k` marker wherever the bake is named, and its Model-details card
  leads with a warning that the architecture/repro shown is the ANCHOR member only;
  `m3_coherence`/`m3a_coherence` are **null** because the coherence instrument loads
  one ZNPR, and null renders as an em-dash (NOT MEASURED — never shaded or displayed
  as a measured zero). Wave-5's six arms are promoted by `scripts/wave5_ensemble.sh
  promote`, which reuses the same frozen membership arrays it scored with.
  Plus (2026-08-04) **FULL-GRID COVERAGE + curation + zoom/pan**: every sota944
  campaign verdict cell is on the board (162 bakes; `scripts/promote_sota944_board.py`
  — a caller of the generalized `promote_fulleval.py` — with a COVERAGE GATE: every
  non-excluded verdict must map to a board file; mapping index
  `fulleval/_sota944_board_map.tsv`; excluded = LOO/REPROCHK/XBUILDCHK/GATE/recheck/
  SMOKE instrument duplicates; `*_corrjoint` fold under the plain name as sha-gated
  corruption_head grafts). Presentation is two-tier so the grid doesn't drown a fresh
  reader: `gauntlet.py CURATED_BOARD` (29 names — era flagships + every arm-candidate/
  named leader + the ensembles) is the default-visible set AND the only set with
  embedded per-pair scatter (registered size rule; grid-interior cells keep every
  scalar stat, their full per-pair stays in the source verdict, recorded via
  `source_verdict` + `per_pair_stripped`); family toggles (arm A/B/C-seeds,
  coherence/W4, near-top, distilled, ensembles, era bridge, pre-944 era) + a
  collapsible per-bake picker + 'curated' preset replace the flat 160-chip bar; the
  scoreboard still lists every cell (dimmed = hidden, click to toggle).
  `model.feature_transforms` embeds are capped at the 48 chips the card can show
  (`n_feature_transforms` keeps the true count).
  Plus (2026-08-04, dashboard-rebuild session) **ECharts SEMANTIC zoom + real
  sortability + regime truth**: the five heavyweight panels (scatter-matrix cells,
  per-codec dial curves, 10-band bars, cross-corpus heatmap, trade maps) are **Apache
  ECharts 5.6.0** (canvas renderer) — dataZoom rescales the AXES and re-plots while
  marks/strokes/labels stay constant size; trade-map labels `labelLayout.hideOverlap`
  (hidden at 1x, REAPPEAR zooming in); dial tooltips show p25/p50/p75 at the hovered
  q; heatmap has a calculable visualMap; band bars show negative bands (old view
  clamped to 0); double-click = reset. The predecessor `makeZoomable` viewBox zoom
  (geometric — overlaps stayed overlapped at every level) is DELETED. **Vendoring:
  the bundle is >30 KB so it is NEVER in git** — bytes at
  `/mnt/v/zen/vendor/echarts/echarts-<ver>.min.js`, described + sha256-pinned by the
  committed `scripts/v_next/vendor/echarts.pointer.md`; `build_html` verifies the
  sha and fails LOUD with download instructions (env `ZEN_ECHARTS_JS` overrides the
  path); the bundle rides its own `<script id=vendor-echarts>` ahead of the app
  script. Charts ink from `THEME_VARS` — ONE Python dict generating both the CSS
  custom properties and `DATA.chartThemes` (light+dark) — and rebuild on
  prefers-color-scheme flips AND on the artifact viewer's `data-theme` attribute
  (MutationObserver, typeof-guarded). echarts.init is guarded on a real canvas 2d
  context + lazy on first viewport intersection, so the DOM-shim harness (no canvas)
  still renders the page. Same session fixed the **scoreboard sort regression**
  (th.onclick called renderTable() — which RETURNS a detached wrapper — instead of
  mountTable(); sorted tables were built and thrown away since 62404415) and made
  EVERY stat table sortable (`makeSortable`: Mohammadi, band, gates, loop). The
  scoreboard/chips **regime** now shows the model's TRUE input width from `n_inputs`
  (372/720/924/944-class) — the stored campaign flag string reads "720" cosmetically
  on all 166 board JSONs. 166-bake board = ~10.3 MB (cap 12 MB), gates PASS.
  Plus (2026-08-04, board-integrity pass) **the annotations-registry + dominance +
  block-usage layer**: `benchmarks/eval_annotations.json` = ONE committed
  machine-readable registry of invalidated/annotated/absent-not-failed numbers
  (schema in its `_schema` header; append-only), consumed by `freeze_check
  --annotations` (absent axes print `— (absent)` DISTINCT from measured fails;
  n/8 keeps the registered absent=not-passed rule AND the n/m-measured form is
  stated; TSV +n_measured/absent/annotations/blocks/dominated_by) and by the
  board (⚠ badges + hover reasons, caption line, `DATA.annRegistry`).
  `dominated_by` marks = strict same-class Pareto (`scripts/sota944_dominance.py`
  rule strict-pareto-2026-08-04 + `promote_fulleval.py --mark-dominated`;
  17 trims, all 944-single) — dominated cells render dimmed + default-off behind
  a "dominated" chip, files NEVER deleted. Scoreboard gained **HF-NL/ref**
  (hfnlproxy per-ref mean; the era incumbents were FILLED by
  `derive_hfnlproxy_372.py` — exact-key join, identity-gated vs the 944 slice —
  and grafted sha-gated via `promote_fulleval.py --graft-rank`; era per-ref
  0.64-0.83 ≫ the 944 frontier's 0.13-0.42 — ⚠ 2026-08-05 appendix O: **80
  pre-orientation-pin board cells were per-ref SIGN-FLIPPED and are repaired**
  (`--repair-rank-orientation`); corrected 944 singles reach 0.73-0.80
  (nt223/W10L9/KFG75), axis LSD ≈0.04, sparsity-TRAINED cells (GL/pilot-λ,
  era-additive) 0.70-0.85 vs mid-MLP ~0.09 — see
  `benchmarks/hfnl_axis_report_2026-08-05.md` + the gauntlet HF-NL panel).
  **Feature-block-usage filter**:
  `bake_block_profile` (via `zenpredict::Model`; no new wire code) →
  `block_profile` injected into all fullevals (`--set-block-profile`, sha-gated);
  "uses f156-371" chip + per-family used/total on the Model-details card —
  f156-371 were ZEROED by the folded regimes (slots preserved per the
  append-only discipline, not removed); 944 MLPs zero them exactly (216/216),
  B uses 49/216. §8's falsified pool-wide census was corrected in place
  (campaign doc CORRECTIONS + BOARD-INTEGRITY PASS sections).
  **Regen gates (MANDATORY, run on every emitted HTML):**
  `scripts/v_next/gauntlet_gates.sh <html>` = `node --check` on EVERY extracted
  `<script>` block + the DOM-shim render harness (`gauntlet_render_check.js`) —
  committed 2026-08-01 (previously ad-hoc; the raw-Python-string `\'` escape class
  blanked the page once, e7f929ca). The harness now ALSO dispatches real header
  clicks and asserts the ATTACHED tables reorder (the sort-regression test), checks
  ECharts mounts + built options + both chart themes, and SSR-renders one option per
  panel kind through the real echarts (svg SSR, no canvas) so a malformed option
  fails the gate instead of blanking the page.
The first two modes' plots: per-bake scatter+trend, grouped 10-band SROCC bars,
calibration curve, residual, candlestick, SROCC heatmap, 2-panel Pareto trade
(CID22 vs nonphoto / KonJND), composite ranking bar, per-codec dial plots +
dial-mono %, full Mohammadi stat panel (incl. low-tail/high-tail SROCC), 10-band
table, honesty/provenance panels. Run from `scripts/v_next/` (imports `blend_lib`/
`gauntlet` from cwd). `bake_report.py` adds the 2×4 8-corpus scatter grid with 4PL
fit + PWRC (reports/); `bake_verdict --html` is the single-bake Rust report.

> **Historical (May-2026 V0_x / PreviewV0_5 era):** the training goals + three-trail SOTA + shipping/experiment-rigor policies, 2026-05-1x eval mandates, V_20/V39 learnings, canonical-2026-05-18 corpus archaeology, the interactive-site spec (since shipped: `site/compare.html`), V_X experiment workflow, V0_1-era weight status, and V0_7 e1 fill were moved verbatim to [`docs/HISTORY-2026-05-v0x-era.md`](docs/HISTORY-2026-05-v0x-era.md) on 2026-07-19. Current guidance: [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md).

## G-ADDR — the dial ADDRESSABILITY gate, and what it measured on the SHIPPED dial (2026-09-04)

USER RULE (2026-09-04): **"floor and ceiling dial addressability is crucial … any model that
limits dial range cannot ship."** `bake_verdict`'s DIAL panel now emits a **G-ADDR** section
on every run and carries `dial.addressability` (+ `dial.min` / `dial.max`) in `--full-json`.
Owner: `zensim-validate/src/dial_addressability.rs`; append-only registry
`benchmarks/dial_addressability_floor_2026-09-04.json`. Full record:
[`benchmarks/dial_addressability_gate_2026-09-04.md`](benchmarks/dial_addressability_gate_2026-09-04.md).

**Two tiers, never merged.** REGRESSION (`A1`–`A9`) bars against a REFERENCE scorer's own
value on the SAME instrument; CONTRACT (`C1`–`C6`) are absolute product bars. Separate
because **the shipped dial fails four contract rows**. Absent probe = NOT MEASURED, never a
pass; an unregistered dial grid = NOT MEASURABLE. Run it with `--negtail-probe` +
`--identity-probe` (both pinned under `/mnt/v/output/zensim/dialgate-2026-09-04/`) or those
axes read `—`.

**⛔ THE REGRESSION BARS ARE `peer_ssim2`, NOT SHIPPED B (re-pinned 2026-09-04).** USER
DECISION, verbatim: *"I don't think we should pin to B, ssim2 seems a better mentor."* The
gate's first run measured its own shipped-B pins to be defective — A1/A3/A6 sat ABOVE what
the reference metric reaches on the same grid, and A4 was met by B only through a −23-point
low-band bias — so the gate was barring candidates for being CLOSER to the truth than the
incumbent. Registry rows are now keyed **`(instrument, reference)`**; the shipped-B set is
retained and printed as **`incumbent`**, never a bar. **Every G-ADDR number published before
2026-09-04 — including §0–§11 of the gate record and the two bullets further down this
section that name A1/A3/A4/A6 fails — is graded on the RETIRED pins.** Reproduce that
grading deliberately with `bake_verdict --gaddr-reference shipped_b`; never quote it as a
current bar. Re-pin record: gate doc **§14**.

- **The bars** (ssim2's own, canonical grid + both pinned probes, full f64):
  `max` **98.376644** · `min` **−55.354544** · `p95` **95.45929935** · `p5` **10.26332105** ·
  `reach` **153.731188** · `DR` **85.1959783** · negtail `min` **−770.619744** /
  `p1` **−187.13142579** / `frac<0` **1.0000** (definitional — the probe was selected on
  ssim2 < 0) · identity **100.000000** on all 38 refs · **0** of 4,424 cells above identity.
- **It is NOT a relaxation — it moved the difficulty to the FLOOR.** Re-grading all 17
  candidates under both pin sets flips **70 cells PASS→FAIL against 9 FAIL→PASS**; shipped B
  goes from **0** regression fails (it *was* the bar) to **6**.
- **ssim2 PASSES the whole CONTRACT tier (6/6)** — mono **0.9924** (more monotone than the
  shipped dial's 0.9792, same zero tied rate), negtail `frac<0` 1.0000, identity MEASURED at
  exactly **100.000000** via `zenmetrics batch --metric ssim2` (not assumed from
  SSIMULACRA2's definition), 0 cells above identity. So the four contract rows B fails are
  not unreachable in principle.
- **Still no `B dial-era v2`.** Every era-corrected arm fails, but on DIFFERENT axes than
  before: **A2 A4 A5 A6 A7 A8 A9 + C2** (floor and spread) instead of A4/A6/C2. The best arm
  is 45.96 short on A2, 44.34 on A5, 758.94 on A7, and reaches 6.6 % of the negative tail
  where the mentor reaches 100 %.
- **Honest limit:** A1/A3 are `≥`, so a dial that OVERSHOOTS the truth passes (B's `max`
  99.98 clears A1 while sitting 1.6 above the reference metric). G-ADDR is an
  **addressability** gate, not a calibration gate; the calibration-referenced reading
  (`|dial end − truth end| ≤ δ`, or MAE vs the reference metric) remains an unimplemented
  user option.
- **Flags that make all of this measurable:** `--negtail-peer-scores` /
  `--identity-peer-scores` (a reference metric has no bake, so its floor axes were previously
  unmeasurable; peer mode is now **all-or-nothing per axis** and never fills a probe from
  `--bake`), `--gaddr-json` (the G-ADDR block alone at full f64, stamped with which scorer it
  describes — peer-safe where `--full-json`/`--fulleval` are refused), `--gaddr-reference`.
  Re-grade an existing bake with `scripts/dialgate_arms.sh score <label> <bake.bin> [regime]`.
- **BOARD COVERAGE (2026-09-04, gate doc §15):** all **97** fair-board cells are graded under
  both pin sets and 96 carry the verdict; the board draws a **NOT SHIPPABLE** badge off a
  MEASURED contract FAIL (**47** cells; C3+C4 dominate at 39 each — dials that never go below
  zero on an all-negative probe — then C2 at 23). The **only** contract-clean cell is
  `v47_strict_QAT_native` (6/6). A probe is scored only when its column count equals the
  bake's caller width, so the 372 probes reach 3 cells; two **944** negative-tail probes were
  cut in-era by `scripts/cut_gaddr_negtail_probe.py` (the committed owner of that cut; its
  control reproduces the stored 372 probe EXACTLY) and are deliberately **NOT registered** —
  no reference has been measured on them, so A7–A9 stay NOT MEASURED there while the
  absolute-bar C3/C4 read normally. Graft with `promote_fulleval.py --graft-gaddr`.

**Facts that are now measured and must not be re-derived:**

- **The identity feature vector is the ZERO vector, for every image — AT 372 ONLY** (38/38
  dial-grid refs, byte-identical), so at that width the identity dial is a SCALAR property of
  a bake, `dial(0⃗)`: **v47-QAT 97.6893, shipped B 96.2412, ADD156 96.1157.** The
  `[97.5, 100]` identity band is a **v47-era** property; both shipped LINEAR dials miss it.
  **⚠ It does NOT extend to 944 (MEASURED 2026-09-04, gate doc §15.3):** extracting the same
  38 `ref == dist` pairs through `sdr944_extract` gives **190 of 944 slots non-zero**, varying
  per image (row-to-row spread 0.594). A 944 identity read is an extractor-era-dependent
  measurement, not an algebraic constant — so C5/C6 need a real in-era 38-pair extraction per
  944 era, and none exists yet (94 of 97 fair board cells read NOT MEASURED on those rows).
- **Shipped B ranks 266 of 4,424 dial-grid cells (6.01 %) ABOVE a perfect copy** in RAW
  space. Consequence, proved in §10.3: **C2 ⊻ C6** — pin identity at 100 and those cells cap
  (tied 0.0567 > 0.05); leave it below and they out-score identity. **No monotone output
  spline can satisfy both. It is a weights defect.**
- **Shipped B's negative tail does not exist**: on a 2,000-row probe whose every row's ssim2
  truth is negative (−770 … −0.33) it emits `frac_below_zero = 0.0000`, min **+2.5167**. The
  cause is the dial anchor's `target_score = max(ssim2, 0)`: 147 of 2,000 rows with genuinely
  negative ssim2 (to −64.16) are stored as 0 and `fit_spline_knots` collapses that run to one
  bottom knot. **The lever for the floor is the CLAMP, not the anchor.**
- **The ssim2 TRUTH on the canonical dial grid** (`/mnt/v/output/zensim/ssim2-bar-2026-08-31/
  dialcells_ssim2_qv2grid.tsv`): min **−55.35**, p5 **10.26**, p95 **95.46**, max **98.38**,
  dynamic range **85.20**, 3.48 % of cells negative. Shipped B bottoms out at **+3.13**.
  **A1 / A3 / A6's bars all sit ABOVE the truth** (99.98 / 99.72 / 86.08 vs 98.38 / 95.46 /
  85.20) — a dial calibrated exactly to the truth fails all three, and **both other shipped
  profiles do**. And **A4's `p5 ≤ 13.645` is unattainable by any monotone dial ON B's
  ORDERING** — an ORACLE arm (the eval grid as its own anchor, truth as target) reads p5
  **21.5–22.8** across n_edges 12→120, worse than every real candidate; shipped B clears A4
  by mapping the low band BELOW its conditional median. The bar IS reachable by other
  weights: **Profile D (ADD156) reads p5 9.52.**
- **Gate readings for the other shipped profiles** (same grid, same probes): **Profile A
  (v47-QAT) is the only bake that passes the ENTIRE CONTRACT tier** — identity 97.6893 in
  band, 0 above identity, negative tail 55.75 % below zero (min −93.90). **Profile D
  (ADD156)** fails contract on **C5 only**, with `p5` 9.52 and 85.80 % of the negative probe
  below zero. The shipped SDR dial (B) is the outlier on every contract row. *(The regression
  fails once listed here — A's A1/A3/A4/A6 and D's A1/A3/A6 — are RETIRED-PIN grading. Under
  the ssim2 bars A fails all nine and D fails A1/A2/A3/A5/A7/A8/A9, i.e. D's A6 flips to PASS
  and both lose the five floor axes B's shallow bars had been granting them.)*
- **WHICH DIAL SHOULD BE THE DEFAULT is an OPEN USER DECISION** (gate doc §14.6), not a
  settled fact: **A** is the only contract-passing dial that exists today (6/6 contract, 0/9
  regression); **D** fails contract on C5 alone (identity 96.1157, 1.384 below the band),
  passes 2/9 regression axes (the only shipped bake to pass any), and beats A on 4 of 5 rank
  corpora. **MEASURED and decisive for D:** D's grid `max` (96.049) is strictly BELOW its own
  identity (96.1157), so every cell already scores under a perfect copy and **the 266-cell
  C2 ⊻ C6 either/or that blocks the B lineage provably cannot arise for D** — building
  `D-id100` is REGISTERED, NOT RUN (a real build: ADD156 lasso lineage, its own scaler and
  anchor, not a re-spline). Shipped B remains the CID22 rank leader (0.88212); the dial and
  the ranker are not the same decision.
- **720/944 bakes read `NOT MEASURABLE` by design**: their default dial grid is unregistered
  and the 372-wide probes refuse to score against them (both refusals print loudly).
  Registering a 944 grid needs a SHIPPED 944-class reference dial to measure a floor from,
  and none exists.
- **ssim2 targets are reusable across decoder eras PER CODEC, not in aggregate.** Re-scored
  today through imazen decoders + `fast-ssim2`: `zenwebp` **0.0000 on 1,000/1,000** rows
  (bit-exact — which also proves bigcodec's stored `score_ssim2` is the CPU path, no GPU
  confound), zenjpeg/mozjpeg/zenjxl median ≤ 0.061, but **XYB JPEG (`zenjpeg-420-xyb-e2`)
  median 0.637 with 58 % past the 0.5-pt dial materiality**, and AVIF carries a 6 % tail
  (max 8.37). Rank is untouched everywhere (SROCC ≥ 0.9998) — it is the ABSOLUTE value that
  moves. When the bytes are on disk, re-score (6,000 pairs = 171 s).
- **`serde_json`'s default float parser is not correctly rounded** and made the reference
  bake fail its own bar by one ULP (`99.98330778475787` parsed back as `…788`).
  `zensim-validate` now enables `float_roundtrip`. **Every float bar `freeze_check` reads out
  of a fulleval JSON had the same hazard.**
- **imazen-26 is the wrong anchor corpus for the FLOOR** — every imazen-26 arm regresses grid
  `min` to +11.9 and the negative-tail probe to +11.6. Its deficit is low-band coverage in
  PREDICTION space; top-densification could never move `p5` (the imazen-26 record found the
  same and this lane localises it).
- **Neither dial anchor is contaminated by CID22.** `check_holdout_overlap --threshold 10`
  over both anchors' references vs the 49 CID22 validation refs: **0 hits at d ≤ 10** for
  both (imazen-26 1,224 refs, closest d=12, 9 origins in the d ≤ 16 screening tier reported
  never quarantined; safesyn 1,495 refs, 0 even at d ≤ 16, closest d=17).

**Build a candidate arm with `scripts/dialgate_arms.sh <label> <anchor.parquet> [n_edges]`**
(shared-anchor → add-winsor → extend-top → gated verdict). Its chain control reproduces the
imazen-26 lane's `B_safesyn_curera` BYTE-IDENTICALLY (`c414b3f91da83e69…`).

## TWO-PANEL EVAL MANDATORY — rank + dial, every ship-grade bake (added 2026-05-29)

**`bake_verdict` runs BOTH panels natively (Rust) on every invocation** —
the DIAL panel is built in, so any time you compute a bake's SROCCs you
also get its dial metrics. Never accept a rank-only verdict:

1. **RANK panel** (`bake_verdict`) — full Mohammadi 2025 stats on the 6
   canonical val parquets. Held-out corpora are CID22 + AIC-3 + AIC-4
   (+ KonJND semi); **KADID/TID are 100% train==val pair-overlap** so
   their numbers reward memorization, not skill — treat as integrity
   guards, not ranking signal.
2. **DIAL panel** (`qsweep_eval` on the densified multi-codec grid) —
   monotonicity + tied-rate + per-q dial span across codec configs
   (G1 dynamic range, G3 monotonicity ≥93% / tied ≤5%, G4 reach). The
   grid is densified where dial precision matters: **q0 + step-1
   q90→q100 + JND-zone (q70→90 step2) + JXL-in-butteraugli-distance**,
   4 codec families, 372 features.

A bake can win the rank panel and be a broken dial (V0_5 Balanced:
panel-best by meanG3, 60% tied / collapses to 0 above q50). A bake can
pass a coarse dial and fail near-lossless step-1 (Cell5: 0.8% tied on
the 16-q grid → 13.1% on the densified grid). **Single-panel verdicts
are a regression — do not accept them.**

**Stored feature sets live on R2** (`s3://zentrain/eval-grids/`:
`dial_grid_372col_2026-05-29.parquet`,
`corruption_grid_372col_2026-05-28.parquet`). `bake_verdict` reads the
dial grid directly (default path or `--dial-grid` / `ZENSIM_DIAL_GRID`)
and forwards the bake over the stored 372-feature vectors —
**rescore any model with no re-encode/re-extract.** If the grid isn't
local, `bake_verdict` emits a loud SKIPPED note; fetch it once with
`aws s3 cp s3://zentrain/eval-grids/...`. Full spec + gates + refresh:
`docs/EVAL_PANEL_REQUIREMENT.md`. Pointer:
`benchmarks/eval_grids_2026-05-29.pointer.md`.

## JSON pipeline mandate for ZNPR v3 bakes (2026-05-15)

**Ad-hoc Python emitters for ZNPR v3 wire format are BANNED.** All
new bake-side serialization goes through the
`zenpredict-bake <input.json> <output.bin>` CLI (binary at
`~/work/zen/zenanalyze/target/release/zenpredict-bake` after a
`cargo build --release -p zenpredict-bake`).

The JSON format is documented in `zenpredict-bake/src/json.rs`:
`BakeRequestJson` with fields `schema_hash, flags, scaler_mean,
scaler_scale, layers[], feature_bounds[], metadata[],
output_specs[], sparse_overrides[]`. Per-bake metadata entries
declare `key: String, type: utf8/bytes/numeric, value: ...`.

Use `scripts/v_next/v0_20b/bake_znpr_v3.py` as a template — emits
JSON, shells to `zenpredict-bake`, exits.

**Why**: the wire format is small but easy to get wrong (alignment,
section ordering, header layout). zenpredict-bake is the canonical
serializer; trusting it keeps wire-format invariants in one place.
Ad-hoc emitters drift, get out of sync with v3.x extensions, and
ship wrong-shape bakes that load but score garbage.

## CID22 is VALIDATION-ONLY (added 2026-05-15)

**CID22 human MOS is sacred validation across the entire zensim
project. NEVER use CID22 human MOS as a training target.** This rule
is load-bearing — every documented contamination cleanup
(2026-05-12 perceptual-overlap purge, 2026-05-14 dHash audits) exists
to defend this gate.

### What "validation only" means in practice

- **NO** `--group cid22:...` argument in any `zensim_mlp_train`
  invocation that loads CID22 human MCOS as the `human_score`
  column. CID22 human MOS appears only at the END of an experiment
  via `dataset_metric_baseline --cid22 /mnt/v/dataset/cid22/...`.
- **NO** "CID22-train-fold" or "CID22-train-subset" carved out of
  the validation set for fine-tuning a head. The 49-reference
  held-out set is the WHOLE CID22 (4,292 pairs across the 49 refs).
  There is no "training-fold half" to peel off.
- **NO** indirect leakage: training-source perceptual-near-duplicates
  of CID22 references count as contamination too. The
  `check_holdout_overlap` audit (dHash d≤10 + user-eye verification
  per the 2026-05-14 revert) is mandatory before any new training
  corpus lands.

### What IS permitted

- CID22 ssim2 or CVVDP metric scores on the **training-only subset
  of the broader CID22 image library** (i.e., images that exist in
  the CID22 source pool but are NOT part of the 49-reference
  validation set + their distorted pairs). The training-only subset
  must be extracted from a different source than the validation set
  on disk — typically the unfiltered CID22 image library at the
  upstream source, NOT `/mnt/v/dataset/cid22/CID22_validation_set/`.
- Metric-anchored training signal on that training-only subset uses
  ssim2 (fast-ssim2 / GPU ssim2) or CVVDP as the target column —
  never human MOS.
- Whoever extracts the training-only-subset metric-anchored CSV
  MUST document the cut clearly (`_MANIFEST.md` entry: "CID22
  training-only subset, ssim2-anchored, N pairs, source images
  NOT in the 49-ref validation set, verified by basename diff").

### What's currently extracted

`/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.csv`
is **validation only** (4292 pairs from the 49-ref held-out set,
`human_score` = MCOS / 100). It exists for end-of-experiment full-
panel evaluation, NOT training input. The file's `_MANIFEST.md`
spells this out.

The historical V_18/V_19/V_20a/V_20b training pipelines have NEVER
included CID22 as a `--group` to the trainer — confirmed by
inspecting every methodology doc at `benchmarks/v0_1*_methodology*.md`
and `benchmarks/v0_19_REVERTED_2026-05-14.md`. The training command
loads `safesyn + kadid + tid + konjnd` only.

### Why this rule is absolute

CID22 (Sneyers / Ben Baruch / Vaxman 2023, JPEG WG1 `wg1m99012`)
is the only large human-MOS dataset that exercises **codec-output
distortions** specifically (KADID + TID are ~95 % non-compression
synthetic distortions). It is the **single gold-standard
generalization holdout** for compression-targeted metrics. If we
train on any part of its human-MOS labels — even a "train fold"
carved from the same 49 references — we lose the only honest
generalization check we have.

Past CID22-contamination incidents (V0_8 perceptual-near-duplicate
leak, V0_19 indirect KADID-overlap inflation) cost the recovery
cycle weeks of wasted training. The "no CID22 human MOS as training
target" rule prevents the next such incident. Re-read this section
whenever drafting a new training corpus or fine-tune fold.

## ZNPR v2 PROHIBITED (added 2026-05-15)

**Producing ZNPR v2 bakes is BANNED. Period.** Every new bake MUST
be v3 (header byte 4 = `0x03`). Tools producing v2 are bugs that
need fixing on contact — not "legacy support" or "compatibility
shims."

### Why

The current zensim runtime loads v3 bakes only. The 2026-05-15
falsification re-evaluation exposed ~150 pre-existing v2 bakes
across `benchmarks/rust_*`, `benchmarks/h*x*`, and
a scratch bakes dir (artifact was under /tmp — wiped; re-derive if needed) that are **structurally unevaluable** by
the current runtime — every recovery-cycle falsified hypothesis
(cycles 7–14) is locked behind this wire-format gap. Producing
more v2 makes the gap worse and creates "ghost bakes" that look
like data but can't be re-tested.

### How to comply

- **Bake-emitting code** uses `zenpredict::bake(&BakeRequest{...})`
  (the v3 path). NEVER call `zenpredict::bake::bake_v2`.
- **Read the bake's header byte 4** as a smoke test in any tool
  that produces a bake: assert it's `0x03` before writing the file.
- **Function names + docs** that say "v2" but emit v3 are
  misleading — rename + correct comments on contact (e.g. zensim's
  `bake_two_layer_znpr_v2` was renamed to `bake_two_layer_znpr_v3`
  on 2026-05-15; the function had been emitting v3 internally for
  weeks).
- **Tests that lock in v2** (`assert_eq!(version, 2)`) are wrong —
  fix them to assert v3.

### Audit list (as of 2026-05-15)

Existing `bake_v2` callers in this repo:

- `zensim-train-core/src/mlp.rs` — REMOVE v2 path; only emit v3.
- `zensim-bench/examples/quant_compare.rs` — same.
- `zenpredict::bake::bake_v2` is still EXPORTED from the sibling
  `zenanalyze/zenpredict` crate, but it MUST NOT be imported into
  zensim crates. If you see `use zenpredict::bake::{..., bake_v2}`,
  fix the import to `bake` only.

### Re-bake old v2 bakes when possible

If a falsification's bake is v2 and the hypothesis is worth
re-testing: **retrain** under the current trainer (which emits v3
through `bake()`). Don't write a v2→v3 upgrade tool — the right
fix is "retrain, evaluate on full Mohammadi panel" per the
principled experiment workflow. Bakes are cheap; ghost data isn't.

## NO DUPLICATE IMPLEMENTATIONS — one owner per task, extend it or don't do it (2026-07-15, user directive)

**Every task below has exactly ONE canonical implementation. Re-implementing
any of them — in Python, in a second Rust site, in a script, anywhere — is
PROHIBITED. Not discouraged: prohibited.** If the owner can't do what you
need, **extend the owner**. That is always the move.

| Task | THE owner | Never |
|---|---|---|
| Load a feature parquet | `zensim-validate/src/parquet_loader.rs` (`load_parquet`) | `pq.read_table` in a script that then trains/evals |
| IQA stats (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE) | **`zenstats`** (`zenmetrics/crates/zenstats/src/panel.rs`), reached via `zensim_validate::panel` (an 82-line `pub use` re-export — the shim is NOT the owner) or the `panel` bin; Python shells it or uses `scripts/lib/zen_stats.py` | `scipy.stats.spearmanr`, a hand-rolled `_srocc`, any private stat math |
| Per-reference SROCC | `bake_verdict` (first-class panel row since 2026-07-15) | reducing `--per-pair-output` in a script |
| Train a model / bake | `zensim-validate/src/bin/zensim_mlp_train.rs` | a torch MLP in a script |
| Evaluate a bake | `bake_verdict` (rank + dial panels) | ad-hoc scoring loops |
| Edit bake bytes (spline / winsor / gate) | `bake_dial_refit` | numpy PCHIP, `struct.pack` |
| Serialize / inspect / repack a bake | `zenpredict` CLI (`bake`/`inspect`/`repack`) | any other ZNPR emitter |
| Build a canonical corpus parquet | `scripts/canonical_corpus/` + `join_safety` | a bespoke join in a probe script |
| Train/val/test split | `zenmetrics/scripts/picker/origin_split.py` (`split_of()`) | a seeded shuffle (per-rendition → scale leakage) |

**Python is not banned — DUPLICATION is.** Python is correct where it IS the
owner: canonical-corpus building, plotting/HTML dashboards, R2 sync. It is
prohibited where a Rust owner exists. The test is not "what language" but
"does this already have an owner". A second **Rust** site is just as much a
duplicate as a Python one — zensim currently carries ~10 private Rust copies of
`spearman` across probe binaries plus a separate impl in
`zensim-train-core/src/stats.rs`.

### The ONE exception: a gated mirror

A second implementation is legitimate **only** when it exists for a measured
engineering reason AND a test holds it bit-exact against the owner. Two real
ones, both keep-don't-delete:

- `zenpicker-train/src/picker_eval.rs` — `pwrc_sa_st_auc_lowmem` (O(n²)→O(1)
  memory), gated by `pwrc_lowmem_matches_canonical_exactly`.
- `zensim-validate/src/panel.rs` — `compute_light_panel_subsampled`, which
  fixed a 307 GB OOM.

"Mirror" means: the owner is still the source of truth, the mirror exists to
solve a *specific measured* problem the owner can't, and a test fails the build
the moment they diverge. Without that test it is not a mirror, it is a fork
with a good story.

### Why the old narrow rule failed

A rule already said "Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in Python"
(the 14-fork consolidation, `benchmarks/iqa_stats_consolidation_2026-05-26.md`).
On 2026-07-15 an audit of `scripts/v_next/` found **30 of 134 scripts
hand-rolling IQA stats anyway**, 69 loading parquet in Python, 11 running
parallel torch trainers, 33 editing bake bytes after that work was migrated to
`bake_dial_refit`. The rule named one *symptom* (srocc in Python) instead of
the *principle* (one owner per task), so every new script re-derived the
forbidden thing under a slightly different name. This section states the
principle. It covers tasks, not function names, and it covers Rust too.

### What duplication actually costs — measured, not theoretical

- **It hides capability gaps.** `blend_lib.py` grew a within-ref RankNet term
  in Python because nobody checked whether the Rust trainer could do it. It
  couldn't — `zensim_mlp_train` drew every pair uniformly across a group
  (cross-image). The gap sat invisible behind a working Python script until
  2026-07-15. Extending the owner surfaced it in an hour and fixed it for
  every future recipe; the duplicate would have hidden it forever.
- **It diverges silently, and you find out months later.** `bake_verdict` once
  had its own inline copy of every stat. When `panel.rs`'s OR + PWRC were
  rewritten to the paper-correct ITU-T P.1401 / Mohammadi SA-ST forms
  (`83e7ff70`), the copy wasn't. **Every bake_verdict output before that fix
  reported the wrong OR + PWRC** while the `panel` binary reported correct ones
  on the same fixture. Nothing failed; the numbers were just wrong. That is the
  characteristic damage: not a crash, a quietly wrong number in a shipped
  report. (The 2026-05-26 consolidation found the same shape three more times —
  PWRC argument order off by ~0.2, an OR definition off by 0.375, and one
  script whose "pwrc" was Spearman-as-Pearson and not PWRC at all.)
- **It re-pays the same debugging.** `blend_lib._load` OOM'd on a 5.3 GB
  parquet (one `read_table`, ~2x peak). `parquet_loader.rs` had never had that
  bug. The duplicate bought a fresh copy of a solved problem.
- **Extraction is not migration.** The 2026-05-26 consolidation succeeded
  architecturally — `zenstats` shipped, both siblings consume it, the parity
  gate passes at ~5e-11. It still failed behaviorally, because the old call
  sites were never migrated and new ones kept appearing. In zenanalyze,
  `load_features_raw` adoption went from 7-of-25 to ~15-of-35: the lib exists,
  the forks kept coming. **Landing the owner is half the job; deleting the
  callers is the other half, and it is the half that gets skipped.**

### NEVER hardcode a sibling-worktree path in a committed script

**A worktree is ephemeral; the repo is not.** Writing
`/home/lilith/work/zen/zensim--my-experiment/target/release/foo` into a script
you commit guarantees that script dies the moment the worktree is cleaned up —
which the mandatory cleanup rule says MUST happen. The worktree rule and the
script outlive each other badly: cleanup works, and silently leaves fossils.

MEASURED 2026-07-15: **25 of 130 scripts** in `scripts/v_next/` pointed into
`zensim--cross-codec-metric`, `--cross-codec-v7/v8/v9`, `--v10`,
`--v10-human-eval`, `--eval-accel`, `--picker-train`, `--exp-tuner-v2`,
`--cli-per-codec-calibration`. Every one had been unrunnable for weeks. **None
of them needed to be**: a worktree is a *copy of this repo*, so every one of
those binaries exists here — the fix was `zensim--whatever` → `zensim`, one
sed, zero deletions.

- **Reference the main repo**, or better, a repo-relative path / an env var
  with a repo-relative default (`ZM_BIN`, `SCORE_BIN`).
- **`just lint-scripts`** fails on a dead worktree ref or a binary with no
  source. Run it before committing a script that shells out.
- A missing artifact whose **source still exists** is just unbuilt — that is a
  `cargo build`, not a fossil. The linter distinguishes these; only 2 of the
  original 25 were genuinely dead (their target had no source anywhere).

Related failure from the same audit: `metric_compare_report.py` had not
**parsed** since a bulk sed (`731cf0eb`) inserted an unescaped
`<meta charset="utf-8">` into a Python string literal. A bulk edit across 293
files broke one and nothing caught it, because nothing ever asked "does this
still run?". `just lint-scripts` asks.

### The rule in practice

1. **Before writing a script that loads/trains/evals/bakes: check the table.**
   If an owner exists, use it. If it can't do the thing, go to 2.
2. **Extend the owner.** Add the flag/mode to the Rust binary, with a test.
   Prove you didn't break existing callers (build the binary at the parent
   commit, run an identical recipe, diff the bake bytes — that's how the
   `:withinref` change proved byte-identity, md5 `346c5a6d…` from both).
3. **A probe/experiment is not an exemption.** "It's just an experiment" is
   how all 134 scripts started. If the experiment needs a capability, the
   capability belongs in the owner; the experiment is then three lines of
   shell.
4. **Delete on sight.** A duplicate found is a duplicate removed, same commit
   if it's dead, next commit if something still calls it. Do not "queue for
   removal" — queueing IS the bug (ML Data Pipeline Discipline §6).

Companion rule in `~/work/zen/zenanalyze/CLAUDE.md` — the other trainer-owning
repo (4,347-line `zentrain/tools/train_hybrid.py`, `zenpicker_train.rs`, ~15
Python torch trainers). zenmetrics deliberately does NOT carry this rule: it is
the *supplier*, not the patient — it owns `zenstats` (which both siblings
consume), has zero MLP trainers, and already enforces single-source rules of
its own (`origin_split.py` hard-errors rather than allow a leaky fallback —
that hard-error is the enforcement pattern to copy).

Audit of record: `benchmarks/duplication_audit_2026-07-15.md`. Prior art:
`benchmarks/iqa_stats_consolidation_2026-05-26.md`,
`benchmarks/cross_repo_duplication_audit_2026-05-26.md`.

## Canonical bake / eval / training tool inventory (added 2026-05-17)

**When you need to do X, use this tool — don't write a new one.**

### Primary `zenpredict` CLI (bake / inspect / repack)
**`zenpredict` binary** at
`/home/lilith/work/zen/zenanalyze/zenpredict-bake/src/bin/zenpredict.rs`.
Build with `cargo build --release --bin zenpredict -p zenpredict-bake`.

The single canonical CLI with three subcommands:

```sh
# Convert BakeRequestJson to ZNPR v3 bin
zenpredict bake <input.json> <output.bin>

# Inspect a ZNPR v3 bake's structure + metadata + weight stats
zenpredict inspect <bake.bin>

# Re-bake an existing v3 with different dtype/compression
zenpredict repack <input.bin> <output.bin> \
    [--dtype f32|f16|i8] [--zerobias <tau>] [--compress] [--optimize]
```

`repack` preserves `feature_transforms`, `output_specs`,
`discrete_sets`, `sparse_overrides`, and all metadata entries.
Verified 2026-05-17 on V_22-IW v2 PreviewV0_5 (200,984 → 14,065 bytes,
7.0% of input, CID22 SROCC delta 0.0003).

The legacy `zenpredict-bake` and `zenpredict-inspect` binaries still
ship but are thin shims that call the same `cli::run_*` functions —
they're deprecated-in-favor-of subcommands. Per zenanalyze CLAUDE.md,
binaries are not part of the semver surface; future passes may remove
the legacy aliases.

**DO NOT USE** `zensim-bench/examples/quant_compare.rs` — it drops
metadata, causing catastrophic SROCC collapse (0.88 → 0.53 on the mix
champion). It is a diagnostic-only weight-magnitude reporter; for any
actual rebake, use `zenpredict repack`.

The JSON pipeline is still mandated for any new bake-producing tool
(per "JSON pipeline mandate" section below). See template at
`zensim/scripts/v_next/v0_20b/bake_znpr_v3.py`.

### STANDARD bake packing — QAT-native (2026-05-27)

**The trainer emits the packed + calibrated bake NATIVELY — no Python
post-step.** Use `--qat-fine-tune-epochs N` (recipe field `qat_fine_tune_epochs`)
+ `--out-dtype f16`: the last N epochs train quantization-aware (f16+zerobias
straight-through estimator), the post-training dial spline is fit on the
PROJECTED+QUANTIZED (shipped) net, and the 2-layer bake stores f16
(encoder) + compressed. One `zensim_mlp_train --manifest v47_strict_qat.toml`
pass → a ~27 KB bake, identity 97.7 (exact), 0 above-identity, correct dial.
VERIFIED 2026-05-27 (`benchmarks/qat_fine_tune_2026-05-27.md`): CID22 0.8657
(> the non-QAT recal 0.8564), Z-RMSE 0.512.

**Load-bearing rule (BOTH paths): fit the dial spline on the SHIPPED net —
projected (encoder≥0, rank_w≤0, α≡1) AND quantized.** Fitting on the
un-projected/f32 net inverts the pred↔target correlation → the spline picks
the wrong direction (blur scored UP to 2184) or identity drops (97.8→93.4).
QUANTIZE-then-CALIBRATE, on the projected net.

**QAT trade (intrinsic, not tau-tunable):** QAT improves CID22 + Z-RMSE but
regresses KonJND (0.485→0.418 — f16 removes the fine-weight precision PJND
discrimination needs; both fail G5's 0.70 floor regardless). So QAT is kept
OPT-IN (`qat_fine_tune_epochs` default 0): the codec-dial ship recipe opts in
(CID22 + native packing win); an HF/PJND-focused bake stays non-QAT.

**Non-QAT fallback** (existing f32 bakes, or HF-focused bakes that can't take
the KonJND trade): `bake_dial_refit pack --in IN.bin --out OUT.bin --neg-tail`
(defaults `--dtype f16 --zerobias-bulk 0.005`) — pack-then-calibrate as a
post-step (strip → zerobias+dtype → refit spline on packed → re-inject).
_(Was `scripts/v_next/pack_and_calibrate.py` — DELETED 2026-07-29 after the
Rust port reproduced the shipped `v47_strict_recal_negtail_packed30k` artifact
BYTE-IDENTICALLY, sha256 `302c9154…`, triple-matched vs a fresh Python run.)_

#### `bake_dial_refit pack` (non-QAT post-step) details

**Load-bearing rule: QUANTIZE, then CALIBRATE.** zerobias/f16/i8 preserve
RANK (signs intact) but SHIFT the network's raw outputs, so a spline fit on
the f32 net maps the PACKED net's identity output to the wrong dial value →
identity drops (97.8 → 93.4 observed). `bake_dial_refit pack` refits the
output spline ON THE PACKED network (strip → zerobias+dtype → refit spline on
packed → re-inject), which re-anchors identity exactly. SROCC is rank-invariant
under the monotone spline. This makes plain GLOBAL zerobias safe — `repack`'s
naive global `--zerobias` (calibrate-then-quantize order) drops identity; do
NOT use it for a spline-bearing bake.

Result on the per-sample-α arch: f32 198 KB → **30 KB**, identity 97.5 (exact),
CID22 0.8564 (≈ f32), 0 above-identity — 6.6× smaller, below the old 41–54 KB
convention, zero quality cost. Per-layer (`--protect-last`) is available but
USUALLY UNNECESSARY (refit recovers identity even with the last layer 98%
zerobiased). Full method + numbers: `benchmarks/standard_bake_packing_2026-05-27.md`.

The V39-era workflow regressed on packing (V39 ships raw F32 257 KB); re-pack
existing `zensim/weights/` F32 bakes through this path when rotating each
profile (SROCC-neutral by construction).

#### DEAD-COLUMN PRUNING — automatic, on by default (2026-08-04)

`pack` now drops layer-0 inputs that **cannot** change a prediction, in the
same pass as zerobias + dtype + spline refit. Order is **zerobias → PRUNE →
quantize → spline** (zerobias is what creates most dead columns; the spline
still lands last on the final packed net, so QUANTIZE-then-CALIBRATE holds).

**The caller's feature width never changes.** A pruned bake declares
`FeatureTransform::Drop` (zenpredict, landed 2026-08-04) on the dead raw lines,
so it still takes 944 features and internally forwards 667. Consequently
`Model::n_inputs()` (667) ≠ `Model::caller_input_width()` (944) — **size every
feature vector by `caller_input_width()`.** Mis-sizing fails loud
(`FeatureLenMismatch`); it never scores a prefix.

**Three classes of "dead", only two prunable** — the whole correctness story,
enforced in `zensim-validate/src/prune.rs` + `tests/prune_classes.rs`:

| class | test | prunable |
|---|---|---|
| 1 weight-dead | `W0[k,:]` exactly zero | **yes — BIT-identical** |
| 2 transform-forced-constant | the bake's OWN transform pins input `k` (winsor family, `lo >= hi`) | **yes** — contribution folded into `b0`; exact in real arithmetic, not bit-identical (the fold reorders one f32 sum) |
| 3 inert on a corpus | `bake_contrib` says mean\|Δ\|≈0 but the weight is live and no transform pins it | **NO** — the corpus merely never exercised it |

Class 3 is the trap: it is indistinguishable from class 1 in any corpus report
and is *not* mathematically dead. `prune::plan()` takes **no corpus statistic
as input**, which makes class 3 structurally unreachable rather than merely
discouraged. Class 2 is refused outright on an i8 layer 0 (removing a nonzero
row can move the per-output max-abs quantization scale).

**Identity gate runs on every pack.** Pre- vs post-prune scores over the anchor
corpus must be bit-identical when only class 1 fired, else within
`--prune-identity-tol` (default 1e-4). Fails loud and refuses to write.

Flags: `--no-prune` (off; restores byte-exact reproduction of pre-2026-08-04
bakes), `--no-prune-constants` (class 1 only ⇒ bit-identical for every input
including NaN), `--prune-identity-tol`.

MEASURED on the three sota944 ship candidates
(`benchmarks/dead_column_pruning_2026-08-04.md`): **944 → 667 layer-0 inputs,
all 277 class 1, identity gate bit-identical on 2035 anchor rows, verdicts
byte-identical.** File size barely moves (−382 B, 0.2%) — LZ4 was already
squeezing the zero rows — so **the win is inference and decompressed
footprint, not bytes**: 29.3% fewer layer-0 rows, a zenbench-measured
**−25.4% forward time** (71.6 → 53.4 ms / 256 rows, 95% CI [−29.6%, −19.1%];
4-round result on a busy box), and −73,128 B resident. The
`bake_contrib` "73 KB = 44% of the packed encoder" figure was a *decompressed*
measurement; do not quote it as a file-size saving.

**`--no-prune` is required to reproduce a historical bake byte-for-byte** —
verified: `pack --no-prune` on `C_em944_s31_dial.bin` reproduces the shipped
`C_em944_s31_packed.bin` sha256 `5870046d…` exactly.

**SPARSE ADDITIVE bakes: pass `--zerobias-bulk 0`.** The 0.005 default is
calibrated for 100-500 KB MLPs, where 0.5 % of the bulk weight magnitude is
noise. On a lasso-sparse additive head every surviving coefficient is signal, and
the default measurably costs rank: MEASURED 2026-08-06 on ADD156
(28 coefficients) **−0.0069 CID22** and on the appendix-T `T_b_lam1e-3` cell
(40 coefficients) **−0.0083**, to buy 192 / 334 bytes. With
`--zerobias-bulk 0` the pack is **rank-EXACT** (0.8634 and 0.8695, unchanged
to 4 dp) and dead-column pruning alone still gives 3,575 → **837 B** (4.3×,
372 → 28 layer-0 inputs, identity gate BIT-identical on all 2,035 anchor rows).
Record: `benchmarks/sota944_campaign_2026-08-03.md` APPENDIX T.R11.

### Bake evaluation (per-bake instant verdict from parquet sidecars)
**`bake_verdict` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_verdict.rs`.
Build with `cargo build --release --bin bake_verdict -p zensim-validate`.

```sh
./target/release/bake_verdict --bake <bake.bin> \
    [--corpora cid22,kadid,tid,konjnd,aic3] \
    [--features-root /mnt/v/zen/zensim-training/2026-08-30-full-features-372] \
    [--output verdict.md]
```
**The `--features-root` default flipped to the current-extractor root on
2026-08-30** (owner: `zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372`).
Every run prints `bake_verdict: features-root era — <label> :: <path>`, so a
verdict is self-describing; pass
`--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features` for a
deliberate STORED-ERA read (the note relabels itself). A number read on one era
cannot be corrected into the other — the shift is model-specific.
Loads pre-extracted 372-feature parquets per validation corpus + bake
bytes, scores MLP via `Predictor::predict_transformed`, emits full
Mohammadi panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) aggregate +
10-band per corpus. **~3.5 sec for all 5 corpora.** Replaces the older
`dataset_metric_baseline` which re-decodes images (~15-20 min per bake).

**Freeze-decision surface (2026-07-31):** `bake_verdict --corruption-head
<head.bin>` adds the companion corruption-head joint report (the shipping
design's corruption owner — the 924 dial's own ordering is broken by design;
dial-alone numbers kept for honesty; `corruption_head` block in
`--full-json`). **`freeze_check`** (same crate) turns one bake's fulleval
JSON into the freeze-bar PASS/FAIL table:
`freeze_check --fulleval <bake.fulleval.json> [--bar csiq=X --bar live=X]` —
externally-owned rows (UPIQ/Korshunov/perf/LOO/corruption-ORDERING) print as
explicit ATTACH rows, exit 1 on any FAIL; it computes NO stats, only compares
what the owning tools produced. Bars = zenpapers final-metric plan §5; owner
map = `benchmarks/decision_surface_audit_2026-07-31.md`. The Korshunov row's
owner (audit gap 3) is now **`scripts/external_reads/run_external_reads.py`**
— the committed seven-domain external-read runner (UPIQ hdr-dmean / SI-HDR /
HDR-VDC / AVT / CHUG / Rousselot / BANDVIS+CSFW LOO): `--from-stored`
rescores the stored feature tables in ~11 s and gate-checks the recorded
numbers (Korshunov 0.9346, Narwaria 0.7688, AVT pooled 0.7742, …);
`--scorer bake:<final.bin>` is the Phase-4 final-bake mode; as-run
provenance copies live in `scripts/external_reads/asrun/`. See its README.

### Replay a training run's pair sampler / describe a drawn subset
**`subset_sim` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/subset_sim.rs`.
Build with `cargo build --release -p zensim-validate --bin subset_sim`.

```sh
subset_sim --fulleval <bake>.fulleval.json --seeds 4004,4005 --out cov.json
subset_sim --group a:x.parquet:1.0:1.0 --seeds 42 --epochs 3 \
           --pairs-per-epoch 2000 --expect-digest <hex>
```
Reconstructs which training pairs a run drew, from its embedded repro block,
with NO feature columns read and no model built. Routes through
`mlp_train::sampling::draw_pair` — the same owner the four training loops use —
so it is a replay, not a re-implementation. `--expect-digest` checks it against
a real run's `ZENSIM_SAMPLE_DIGEST=1` output.

### IQA statistical panel on arbitrary (predicted, target) pairs
**`panel` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/panel.rs`.
Build with `cargo build --release -p zensim-validate --bin panel`.

THE canonical entry point for the full Mohammadi 2025 panel (SROCC +
PLCC + KROCC + OR + PWRC + Z-RMSE + per-sample Z-RMSE + 4-param
logistic) on an arbitrary table — the NON-bake case. (For a bake on a
canonical corpus use `bake_verdict`.) Reads TSV or Parquet with columns
`predicted`, `target`, optional `sigma`, optional `band`:

```sh
panel --input scores.tsv [--json]            # aggregate panel
panel --input eval.parquet --json            # + per-band when `band` present
panel --batch jobs.tsv --stats srocc         # N (x,y) pairs -> N rows, ONE process
```

Wraps `zensim_validate::panel::{compute_panel, z_rmse_per_sample,
rescale_logistic}` directly — zero new stat math. Verified equivalent
to scipy to <= 1e-9 by `scripts/verify_panel_parity.py` +
`tests/panel_parity.rs`. Python pipelines that can't shell directly use
the thin `scripts/lib/zen_stats.py` shim (`from scripts.lib.zen_stats
import panel`). **Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in
Python** — that re-creates the 14-fork divergence this consolidates
(see `benchmarks/iqa_stats_consolidation_2026-05-26.md`).

**Batch mode (2026-07-31, audit gap 4):** `panel --batch <FILE|->`
takes a manifest of many (x, y) vector pairs — explicit rows, or
`#def`'d base vectors + index-set resamples (the paired-bootstrap
shape; the caller keeps the RNG) — and emits one TSV stat row per pair
in ONE process, so a 10k-resample bootstrap is a single invocation.
`--stats srocc` fast path; full mode adds `srocc_signed` (pre-abs
midrank) + `plcc_raw` (un-rescaled Pearson). Python:
`zen_stats.panel_batch` / `panel_batch_indexed`. Gate:
`scripts/verify_panel_batch_parity.py` (<=1e-12 vs scipy midrank incl.
tie-heavy; determinism) + `tests/panel_parity.rs --ignored`.
**scipy-in-a-bootstrap-loop is the banned pattern this replaces** —
`scripts/hdr/upiq_panel.py` is the migrated exemplar (byte-identical
recorded outputs, 3× faster).

### Bake training (MLP supervised learning)
**`zensim_mlp_train` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/zensim_mlp_train.rs`.
Supports `--group <name>:<path.parquet>:train_w:val_w` (auto-detects
.csv vs .parquet by extension), `--target-column NAME`,
`--feature-set extended_iw|extended|standard`, full PWRC+NiN flags,
auto-transforms via `--auto-transforms <screen.tsv>`.

### Eval matrix comparison (N bakes side-by-side)
**`scripts/cvvdp_matrix_compare.sh`** at
`/home/lilith/work/zen/zensim/scripts/cvvdp_matrix_compare.sh`. Runs
bake_verdict on every `*.bin` in a dir + emits a per-corpus
SROCC/Z-RMSE/PWRC table for ship-decision review.

### Bake dial / spline refit + tail gate (bake_dial_refit)
**`bake_dial_refit` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_dial_refit.rs`.
Build with `cargo build --release -p zensim-validate --bin bake_dial_refit`.

THE canonical Rust home for editing a ZNPR bake's output-calibration
spline / feature-winsor guard — replaces the `scripts/v_next/*.py` that
hand-edited bake bytes in numpy (2026-07-05 migration). Reuses the shared
serializer (`zenpredict_bake::bake`), spline eval (`output_calibration_spline`),
and stats (`zenstats::panel`) — never re-serializes or re-implements PCHIP.
Subcommands:

```sh
# extend the spline TOP by the training-fitted concave saturation (THE
# shipped-B producer; reproduces b_sdr_linear_cid80_dense_dial BYTE-IDENTICALLY)
bake_dial_refit extend-top --in <winsor.bin> --out <out.bin> \
    --anchor <multiband_anchor.parquet> --target-col target_score
# whole-spline refit to a shared anchor (percentile-edge fit_spline_knots)
bake_dial_refit shared-anchor --in <bake> --out <out> --anchor <parquet> \
    --target-col <col> [--target-scale 100]
# prepend a (floor_raw, 0.0) bottom knot   (BYTE-IDENTICAL to Python)
bake_dial_refit bottom-extend --in <bake> --out <out> --floor-raw 0.0
# add 372 winsor_p99 guards from a fit corpus (functionally identical)
bake_dial_refit add-winsor --in <raw.bin> --out <out.bin> --fit-corpus <parquet>
# G-RANGE tail gate (below/above-knot raw-pred fraction) + Z-RMSE/OR/SROCC,
# NO PWRC (OOM-safe). The 3rd eval panel SROCC is blind to.
bake_dial_refit gate --bake <bin> --corpus <parquet> [--ref-col human_score]
# STANDARD non-QAT packing: per-layer zerobias + DEAD-COLUMN PRUNING + dtype,
# spline refit ON THE PACKED net. Pruning is ON by default (944 -> 667 on the
# sota944 bakes, bit-identical); --no-prune restores the pre-2026-08-04
# BYTE-IDENTICAL reproduction of pack_and_calibrate.py + the shipped packed30k
bake_dial_refit pack --in <f32.bin> --out <out.bin> [--neg-tail] \
    [--dtype f16] [--zerobias-bulk 0.005] [--protect-last] \
    [--no-prune] [--no-prune-constants] [--prune-identity-tol 1e-4]
# drop one metadata entry, rest verbatim (BYTE-IDENTICAL to the deleted
# strip_spline_metadata.py on both MLP and linear fixtures)
bake_dial_refit strip --in <bake> --out <out> [--key zentrain.output_calibration_spline]
# lasso-CD fit on a FROZEN gram npz + f16 pack + anchor spline + bake — THE
# shipped-BHdr producer, pure Rust (task #68; lasso w/bias/mu/sd f64 BIT-EXACT
# vs the Python fit via --parity-fit; whole file sha 7d7f2123… BYTE-IDENTICAL;
# reproduce_bhdr.sh now runs zero Python between fit and bake)
bake_dial_refit fit-lasso --gram <grams/hdr_v3mix.npz> --space shaped \
    --target human_score --lam 0.0003 --anchor <val/anchor.npz> \
    --transforms-tsv <screen.tsv> --out <bake.bin> [--parity-fit <fits/*.npz>] \
    [--tau 0] [--expect-sha256 <hex>]
```

Method + measured byte-parity: `benchmarks/bake_refit_rust_migration_2026-07-05.md`
(+ `benchmarks/pack_rust_migration_2026-07-29.md` for `pack`,
`benchmarks/key_bake_repro_verification_2026-07-29.md` for `fit-lasso`).
`fit-lasso` support modules: `zensim_validate::gram_lasso` (bit-exact
MixGram+lasso port, single-rounding f64→f16, CPython float-repr) and
`zensim_validate::npz` (minimal stored+deflate npz reader via zenflate).

### Affine calibration of an existing bake
**`affine_calibrate` binary** at `zensim-validate/src/bin/affine_calibrate.rs`.
Applies `y' = α + β·y` (`W' = β·W`, `b' = β·b + α`) to a ZNPR **v2 or v3** bake
— v2 and v3 share the layer-table layout for the first 96 header bytes, and
v3.1's reserved fields are zero for the F32 bakes we calibrate.

_(Corrected 2026-07-15: this section previously read "Missing v3 equivalent for
the affine op — build a v3 affine tool when needed". That was false; the Rust
port has existed since 2026-06-18. A stale "missing tool" claim is worse than
no claim — it tells the next session to rebuild something that ships, and it
excused `scripts/v_next/affine_calibrate_bake.py` as filling a gap that was
never open. Per the no-duplication rule, that script is a duplicate.)_

Output-**spline** / dial refits are NOT affine — those live in
`bake_dial_refit` above.

### Per-corpus baseline metric extraction
**Missing v3 equivalent.** Older `score_unified_with_bake.py` was
v2-only (DEPRECATED, refuses). Use `zenmetrics batch` (from
`/home/lilith/work/zen/zenmetrics/target/release/zenmetrics`) for
metric scoring on (ref, dist) pairs, then merge into per-corpus
parquet sidecars analogous to T11.7 safesyn CVVDP backfill.

### CVVDP scoring
**`zenmetrics`** at
`/home/lilith/work/zen/zenmetrics/target/release/zenmetrics`.
Build with `cargo build --release --bin zenmetrics --features 'gpu-cvvdp,gpu-cuda' -p zenmetrics-cli`.

```sh
zenmetrics batch --metric cvvdp --gpu-runtime cuda \
    --pairs <pairs.tsv> --output <scores.tsv>
```
Pairs TSV must have `ref_path` + `dist_path` columns. Note: rejects
16-bit RGB and 8-bit RGBA inputs (decoder widening pending). For TID's
`.BMP` images, convert to PNG first (see T11.10b notes).

### Migration tools
- **`zenanalyze/zentrain/tools/migrate_znpr_v2_to_v3.py`** — converts
  an old v2 bake to v3. Use this exactly once per archived bake; the
  trainer + bake_verdict + zenmetrics all produce v3 natively now.

### Deprecated / DO NOT USE
- `zensim-bench/examples/quant_compare.rs` — drops metadata, catastrophic SROCC loss.
- `dataset_metric_baseline` (zensim-bench example) — slow (15-20 min)
  AND silently drops KADID rows on image-decode failures. Use
  `bake_verdict` instead.
- **DELETED 2026-07-15** (superseded; git history preserves them —
  "kept for provenance" was redundant with version control, and a
  deprecated file left in tree is a file the next session copies):
  `dense_dial_refit_b.py` → `bake_dial_refit extend-top`,
  `bhdr_bottom_extend.py` → `bottom-extend`, `winsorize_bake.py` →
  `add-winsor`, `w11_webp_ood_refit_2026-07-05.py` (falsified campaign).
  The first three were proven **byte-identical** to their Rust
  replacements before deletion. Already deleted earlier:
  `affine_calibrate_znpr_v2.py`, `score_unified_with_bake.py`,
  `soft_iso_smooth.py` — this list claimed they were "deprecated but
  present" long after they were gone.
- **DELETED 2026-07-29**: `pack_and_calibrate.py` → `bake_dial_refit pack`
  — proven byte-identical THREE ways (fresh Python run == Rust ==
  the shipped `v47_strict_recal_negtail_packed30k_2026-05-27.bin`,
  sha256 `302c9154…`; `benchmarks/pack_rust_migration_2026-07-29.md`).
  Also deleted same day: `bake_outlier_gate.py` → `bake_dial_refit gate`
  (its one importer `xmetric_consensus.py` now shells the canonical
  `predict_features_with_bake` forward + `zen_stats.srocc` — smoke-verified
  on a kadis-gpu slice); `shared_anchor_refit.py` → `shared-anchor`
  (the claimed `hdr_anchor_dense_refit.py` importer was STALE — it imports
  `linear_projections`, the mention was docstring-only);
  `strip_spline_metadata.py` → `bake_dial_refit strip` (byte-identical on
  the v47 MLP `7c65814e…` AND shipped-B `5ec68b1f…`; live caller
  `recal_v47_dial.py` migrated); `bake_to_znpr.py` (DEAD: emitted banned
  v2, trainer gone); `affine_calibrate_bake.py` (duplicate of the Rust
  `affine_calibrate` bin per the affine section above).
- `hdr_anchor_dense_refit.py` is PARTIALLY migrated: its base whole-spline
  refit is `bake_dial_refit shared-anchor`; only the 28-bin densify + Q-Q
  top-end knots remain as experiment logic. Its bake primitives live in the
  Rust bin — don't resurrect the numpy PCHIP/serialize code.

## zenpredict crate dependency policy (added 2026-05-15)

**Use path or git refs to the local `zenanalyze/zenpredict` repo,
NEVER the published crates.io version.** zenpredict 0.1.0 on
crates.io is v2-only; v3 lives unpublished on the local sibling.
Pinning the published version would silently ship a runtime that
can't load any current bake.

### Default: path ref (sibling worktrees)

In the zensim workspace `Cargo.toml`:

```toml
[workspace.dependencies]
zenpredict = { path = "../zenanalyze/zenpredict" }
zenpredict-bake = { path = "../zenanalyze/zenpredict-bake" }
```

This works when the user's machine has both repos checked out as
siblings under `~/work/zen/` — which is the standard layout for
zen-org work. Path is preferred because it makes cross-repo edits
inspectable in `cargo build` output and avoids stale lockfiles.

### Fallback: git ref (CI, fresh clones)

For CI or environments without the sibling worktree, use git refs
pinned to a specific commit:

```toml
zenpredict = { git = "https://github.com/imazen/zenanalyze", rev = "<commit-sha>" }
zenpredict-bake = { git = "https://github.com/imazen/zenanalyze", rev = "<commit-sha>" }
```

Update the `rev` deliberately when a v3 feature lands that zensim
needs. Do NOT use a branch ref (`branch = "main"`) — that causes
silent breakage when zenanalyze's main moves.

### Audit

When adding a new zen-internal dependency (zencodec, zenresize,
etc.), check the workspace `Cargo.toml` for the right pattern. If
a sibling exists under `~/work/zen/`, use path. Never copy a
published-crate version from crates.io into a workspace dep.

## Shell scripting gotchas (added 2026-05-15)

### Bash readonly variables: GROUPS, PIPESTATUS, EUID, UID, ...

Assigning to these in a bash script may silently fail to take effect
— the result `$VAR` resolves to the builtin value, not yours. The
trap that bit a Phase 3 retrain script today:

```bash
GROUPS="--group safesyn:... --group kadid:..."   # silently overridden
zensim_mlp_train $GROUPS ...                     # bash sees $GROUPS = "1000"
# error: unexpected argument '1000' found
```

`$GROUPS` is bash's primary-group ID (e.g., `1000` on most Linux
boxes). Reading from it gives the readonly builtin; writing to it
in `bash` works in interactive sessions but is unreliable in scripts
(depends on `set -u`, shell mode, etc.).

**Avoid these names in scripts**: `BASH`, `BASHOPTS`, `BASHPID`,
`BASH_*`, `COMP_*`, `DIRSTACK`, `EUID`, `FUNCNAME`, `GROUPS`,
`HISTCMD`, `HOSTNAME`, `HOSTTYPE`, `LINENO`, `MACHTYPE`, `OSTYPE`,
`PIPESTATUS`, `PPID`, `RANDOM`, `SECONDS`, `SHELLOPTS`, `UID`.
Pick descriptive prefixed names instead (`DSET_GROUPS`, `TRAIN_GROUPS`,
`PIPE_STATUS`).

When debugging a script that produces unexpected positional args:

```bash
# This trick reveals readonly-builtin collisions:
GROUPS="hello"; echo "[$GROUPS]"   # might print "[1000]" not "[hello]"
```

If you see "unexpected argument 'NNNN' found" from a CLI tool and
NNNN is a small integer (often 1000, 65534, 0), suspect a readonly
collision before suspecting the CLI.

### `set -u` masks the readonly collision

With `set -u` on, writing to a readonly variable produces no error;
the read silently uses the readonly value. Without `set -u`, the
same script may still appear to work in some shells. Make the
diagnostic explicit by renaming.

## Release Process

`zensim` and `zensim-regress` are released **independently** with **separate semver**. A bump to zensim does not require a bump to zensim-regress, and vice versa. Tag format:

- `zensim-v0.2.0` for the zensim library crate
- `zensim-regress-v0.1.1` for the regression testing crate

`zensim-validate` is internal tooling — not published.

### Before any release

1. Run `cargo semver-checks` against the previous published version:
   ```bash
   cargo semver-checks --manifest-path zensim/Cargo.toml
   cargo semver-checks --manifest-path zensim-regress/Cargo.toml
   ```
   Fix any semver violations before bumping. If the API change is intentional, bump the appropriate semver component (minor for additions, major for breaking changes).

2. Run the full test suite: `cargo test --workspace`

3. Run clippy clean: `cargo clippy --workspace --all-targets`

4. Verify README.md is accurate — ask user to confirm before publishing.

### Release steps (per crate)

1. Bump version in `<crate>/Cargo.toml`
2. Run `cargo update -w` to update workspace lockfile
3. Run `cargo semver-checks --manifest-path <crate>/Cargo.toml`
4. Commit: `release: <crate> v<version>`
5. Tag: `git tag <crate>-v<version>`
6. Push tag: `git push origin <crate>-v<version>`
7. Publish: `cargo publish --manifest-path <crate>/Cargo.toml`

Never publish without a matching pushed tag. Never tag without passing semver-checks.

## Weight Training & Dataset Contamination

### dHash threshold (2026-05-14, after revert)

`check_holdout_overlap` uses dHash-64. The literature thresholds:

| Hamming distance | Label | Use for contamination? |
|---|---|---|
| d = 0      | identical (bit-perfect)                 | yes |
| d ≤ 5      | near-identical (recompression / resize) | yes |
| d ≤ 10     | "very likely the same image"            | **yes, but require user-eye verification** |
| d ≤ 16     | "possibly the same image" (screening)   | **NO** — too many false positives in our content domain |

**The d ≤ 16 default in `check_holdout_overlap.rs` is a screening
threshold for HUMAN review, NOT an automatic contamination cutoff.**
A 2026-05-14 cleanup based on d ≤ 16 produced a 149-basename blocklist
that user review proved was mostly false positives (UI screenshots
matching by flat-region dHash; "blue sky" overlap mistaken for content
overlap). The cleanup was REVERTED — see
`benchmarks/dhash_threshold_revert_2026-05-14.md`.

**Ship policy for any future contamination claim**:
1. Run `check_holdout_overlap --threshold 10`.
2. Build side-by-side montages for every flagged pair.
3. Get user sign-off entry by entry before adding to any blocklist.
4. Never auto-quarantine based on dHash alone.

**Where dHash IS the right tool — WITHIN-corpus resample-variant clustering
(issue #33, landed 2026-08-28):** `corpus_content_clusters` (owner module
`zensim_validate::content_clusters`, also the one `dhash_64` both
`check_holdout_overlap*` bins use) clusters a corpus at the STRICT d ≤ 3
threshold so `<hex>_512sq` / `_769x513` / `_1024sq` variants of one source
form one cluster, then emits per-content weights (`1/cluster_size`; for the
trainer, `--reweight-dir` writes one `--group` per cluster size with
`train_w ∝ n_rows/k`), a one-variant-per-cluster cull (`--cull-csv`), and a
content-stratified split (`--split-dir`). It refuses `--max-dist > 10`.
`--montage-dir` renders the mandatory eyeball pass (montage PNG + `index.html`
per proposed group, with max intra-group distance and every member's
dhash/pixels/canonical/split; `--montage-all` for every multi-member cluster) —
the review is the precondition for acting on the clusters, per the policy above.
The reviewer sign-off and the retrain comparison (equal-weight vs reweighted vs
culled) have NOT been run — they need `/mnt/v/input/zensim/sources/` + a
training box.

### Safe synthetic dataset (V0_18 ship corpus)

- File: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv`
  (218,089 pairs; sha256
  `659982b3ce8d26184eca835a85f8d66c8550d945d659559c499b5670cd5d8589`).
- R2 mirror (2026-05-22):
  `s3://zentrain/synthetic-v2/training_safe_synthetic.csv`.
  Encoded distortion **bitstreams** under `/mnt/v/input/zensim/images/`
  (`<ref>/<codec_dir>/q<X>.{jpg,webp,avif,jxl}`) are mirrored to
  `s3://codec-corpus/synthetic-v2/<ref>/<codec_dir>/q<X>.<ext>` —
  **bitstreams only (~38 GiB, 729,703 objects, R2-verified 2026-06-22)**.
  The earlier claim of a `q<X>.png` mirror was WRONG: R2 never held the
  decoded PNGs.
- **⚠ 2026-06-22: the `q<X>.png` decode-cache was DELETED** (~402 GiB;
  `images/` went 440.80 GiB → bitstream-only ~38 GiB; freed because both
  `/mnt/v` and `/` were ~94-99% full). Each `q<X>.png` was a lossless
  decode of the adjacent `q<X>.<bitstream>` that `extract_features_372col`
  consumed via the CSV `decoded_path`. **So the CSV `decoded_path` PNGs no
  longer exist — re-extraction must DECODE the bitstream first** (unified
  zencodec API; reference: `zensim-bench/examples/verify_bitstream_decode.rs`,
  `--features verify-decode[,verify-avif,verify-jxl,verify-webp]`). The
  canonical feature parquets are unaffected (frozen + mirrored R2/Tower);
  only the regenerable cache is gone. Decoder-drift caveat measured
  2026-06-22: zencodec re-decode is **byte-exact** for the May-gen
  `zenjpeg-420-e1` run, but March-gen JPEG runs drift (zenjpeg decoder
  evolved: max_abs ≤ 5; XYB ≤ 42) and JXL differs (zencodec uses
  `zenjxl-decoder`; the generator used `jxl-oxide`) — so re-decoded pixels
  will NOT byte-match the canonical parquets for those codecs. If exactness
  matters, re-extract ALL corpora through one decoder rather than mixing.
- **⛔ CORRECTED + QUANTIFIED 2026-09-04 (`docs/DATASET_HISTORY.md` §3.32,
  `benchmarks/b_reextract_wave_2026-09-04.md` §3): safesyn is NOT
  re-extractable, and "decode the bitstream first" is not a fix.** Two CSVs
  exist and are **row-identical except in one column**: the actual 372-col
  extraction input
  `/mnt/v/zen/zensim-training/2026-05-16/safesyn_with_iwssim.csv` points
  `decoded_path` at the **`q<X>.png` decode cache** (measured survival
  **0/3000 sampled rows, all six codec families**), while
  `synthetic-v2/training_safe_synthetic.csv` points at the **bitstreams**
  (present). The bullet above is true only of the first file; conflating
  them is what makes safesyn look re-extractable.
  **⚠ THE MAGNITUDE HERE WAS RETRACTED 2026-09-04 — §3.34,
  `benchmarks/safesyn_zencodec_probe_2026-09-04.md`.** ~~a HEAD extraction
  moves the basic block on 240/240 rows, 69 % of cells over tolerance,
  worst `0.659 → 2875.0`, ~10⁴× the correction~~ — that probe decoded with
  the third-party `image` crate, which reads an XYB JPEG as an ordinary
  JPEG and has no AVIF/JXL decoder, so it *manufactured* its own headline.
  **Re-measured with OUR codecs** (zencodec detect + zenjpeg/zenpng/zenwebp/
  zenavif/zenjxl; 360 rows, 60 per family, all six families, alignment gate
  360/360): basic worst cell **5.481e+1 (52× smaller)**, the XYB family's
  own worst **29.84 → 60.16 (95× smaller)**, and `|Δ| > 1.0` on **14 cells
  = 0.025 %, 6 of 360 rows**, all in `zenjpeg-420-xyb-e2` / `zenjxl-e7`.
  Those land INSIDE the 2026-06-22 drift bounds in the bullet above (plain
  JPEG ≤ 5, XYB ≤ 42, JXL differs by lineage); the retracted 2,875 was
  outside all of them.
  **safesyn is still NOT re-extractable, for a measured reason instead:**
  re-decoding shifts shipped B's dial by **mean −3.658 points (median
  −3.181, sd 2.589, 94.4 % of rows > 0.5)** against an era defect of
  −4.98/−5.86 — **decoder era is 73 % of extractor era**, i.e. a confound of
  comparable size, not of four orders. **This is true of EVERY stored-pixel
  corpus, imazen-26 included**: pick a deliberate decoder era and record it
  per format, do not hunt for a corpus that escaped drift.
  **FIXED same day**: `extract_features_372col` no longer decodes via
  `image::open()` + `None`. It uses the one owner
  `zensim-bench/examples/shared/zen_decode.rs` (magic-byte detection via
  `zencodec`, five imazen codecs, `Rgb16` through
  `zenpixels_convert::RowConverter`) and returns `Result`; a row that cannot
  be decoded ABORTS unless the caller passes `--allow-failures N` (default
  0). Build it with `--features training,zen-decode`. Gate:
  `zensim-bench/tests/zen_decode_formats.rs`, 13 tests.
  The same blocker owns `multiband_anchor_dial100.parquet`, a 2,000-row
  safesyn subset (joins **2000/2000** on `(ref_basename, f0)`) that is
  shipped-Profile-B's **entire dial calibration anchor**.
- Tower mirror:
  `/mnt/tower/output/zensim-archive-2026-05-20/synthetic-v2-{tables,images}/`.
- Created from `training_concordant.csv` minus all 49 CID22 validation
  image sources.
- 475 CID22-contaminated pairs removed.
- Always use this CSV for V_X training; never `training_with_dssim.csv`
  or `training_concordant.csv`.
- Feature cache: `training_safe_synthetic.csv.features.*.bin`.
- Also valid: the 2026-05-12 post-CID22-purge variant at
  144,791 rows (artifact was under /tmp — wiped; re-derive if needed),
  produced after Phase-1 CID22 d ≤ 16 purge — **also** at the
  loose threshold; subject to the same false-positive caveat.

### Dataset contamination rules (2026-05-14, post-revert)

- **CID22**: 49 validation images. Original 2026-05-12 Phase-1 purge
  removed 361 sources at d ≤ 16 from CID22 refs. Those flags were at
  the loose threshold and need re-audit at d ≤ 10. CID22 ↔ KADID and
  CID22 ↔ TID cross-corpus audits at d ≤ 10 BOTH find **zero matches**
  — CID22 is perceptually disjoint from both holdouts.
- **KADIK10k**: Uses I01-I81 reference images. At d ≤ 10, **6 training
  sources** match KADID refs (4 `gmessages_*` variants near I18,
  `e7a01ec14bcca684_769x513.png` near I18 d=7, `2232979_512sq.png`
  near I25 d=10). Several of those are flat / UI / screen-content
  images where dHash is unreliable (large zero blocks dHash to zero
  regardless of content). User review pending in
  `/mnt/v/output/zensim/contamination_review_2026-05-14/d10_kadid_matches/`.
- **TID2013**: 25 reference images. At d ≤ 10, **1 training source**
  matches TID I12 (`b5cd470348ef0609_769x513.png` d=10). User review
  pending.
- **The file-name "no overlap" check is insufficient**. Hex-hashed
  training source names don't collide with KADID's I01..I81 or TID's
  I01..I25 namespace, but content can still overlap. Use
  `check_holdout_overlap` (dHash-64 at d ≤ 10) PLUS user review of
  side-by-side montages before declaring contamination.
- **Synthetic training sources**: Hex-hashed tiles from CLIC 2025 +
  CID22 collections.
- **dssim co-training is FALSIFIED** (cycle-7 verdict, commit
  `4ed499e`): all 5 dssim-weighted variants regressed CID22 by 0.04–0.07
  vs V0_16 baseline. Don't retry without a fundamentally different
  mechanism. The identified next lever for B0/B1 SROCC is direct
  JPEG-AI training-corpus acquisition (not started).
- **AIC-3 / AIC-4 are HOLDOUT-ONLY**. Never train on them.

### Contamination guard status

The `scrub_csv_or_die` runtime guard (in
`zensim-validate/src/contamination_guard.rs`) is still present but
its 149-basename embedded blocklist is **stale and over-aggressive**
(loose-threshold false positives). Don't rely on the guard's
embedded blocklist; regenerate it at d ≤ 10 + user verification
before reactivating as a ship gate.

### Available human datasets for training/evaluation
Three independent human datasets: **KADIK10k** (10,125 pairs), **CID22** (4,292 pairs), **TID2013** (3,000 pairs).
- Train on synthetic + 1-2 human sets, validate on remaining holdout(s)
- Use `--also type:path` and `--dataset-weights name:weight` flags
- Human datasets should be weighted to exceed synthetic (e.g., 1.0:2.0)

## KADIS-700k dataset (zensim 2026-06-30; GPU-metrics 2026-07-01)

700,000 distorted-image cells — 140k KADIS pristine references × 1 `dist_type_1` × 5 severity
levels, each with its 372-D zensim feature vector. **The zensim score and the 372-D `feat_*`
vectors are produced by THIS crate** (`Zensim::compute_extended_features`, `with-iw` regime);
the pure-CPU path (no GPU dep) is what made the cheap-fleet zensim sweep reliable, and the
GPU-metrics variant additionally runs zensim's GPU backend as `score_zensim_gpu`. Two canonical
variants (same 700k cells, same `source_id` split key):

- **★ GPU-metrics canonical (2026-07-01) — current, richest.**
  `s3://zentrain/kadis-700k-gpu/canonical/kadis700k_canonical_gpu_2026-07-01.parquet`
  (700k×387, ~936 MB zstd, 0 nulls; sha256 `c9a6fd56…`). **7 perceptual scores** —
  `score_{zensim,ssim2,butteraugli_max,butteraugli_pnorm3,iwssim,dssim}_gpu` + `score_cvvdp_cpu_imazen_v0_1_0`
  — plus `distorted_url` (a persisted distorted PNG per cell → rescore-from-links), on top of the
  372-D `feat_*` + shared keys. Sidecars `s3://zentrain/kadis-700k-gpu/{omni,zensim_features,pairs}/`
  + `distorted/<chunk>/*.png`.
- **zensim-only canonical (2026-06-30) — earlier variant.**
  `s3://zentrain/kadis-700k/canonical/kadis700k_canonical_2026-06-30.parquet` (700k×380, ~906 MB
  zstd, 0 nulls; sha256 `b57e4b3f…`). `score_zensim` + `feat_0..feat_371`. Sidecars
  `s3://zentrain/kadis-700k/{omni,zensim_features,source_features}/` (350 each).
- **Shared keys (both):** `source_id` (stable split key 0..139999 — split on this, never on row,
  for leak-free train/val/test), `source_filename`, `dist_type`, `dist_name`, `severity_level`,
  `dist_param` (signed for types 7/18/25 → U-shaped scores by design).
- **Mirrors:** `/mnt/v/datasets/kadis700k/canonical/`, `/mnt/tower/output/kadis700k/canonical/`.
- **Full README + schema:** `s3://zentrain/kadis-700k-gpu/README.md` + `s3://zentrain/kadis-700k/README.md`
  (and `~/work/kadis-distort/docs/DATASET.md`).
- **Credit:** reference images + distortion design © VQA Group, Universität Konstanz (Lin, Hosu,
  Saupe) — KADID-10k / KADIS-700k, https://database.mmsp-kn.de/kadid-10k-database.html ("freely
  available to the research community"). Cite KADID-10k (QoMEX 2019) + DeepFL-IQA (arXiv:2001.08113).
