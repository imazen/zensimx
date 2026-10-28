# Open decisions — 2026-09-06 (coordinator memo, written for the user)

Every ruling that is the user's to make, with the measured evidence, the options and their consequences, the coordinator's recommendation, and the exact next action each ruling triggers. Links are to records on `main` (GitHub blob URLs) and to the LAN gallery. Nothing here is installed or flipped until ruled.

Board (fair tier, ladder ruler): http://localhost:3300/zensim/reports/summer_gauntlet_fair.html
Four-way compare (shipped D · constrained MLP · guarded D · fast-class MLP): http://localhost:3300/zensim/reports/summer_gauntlet_fair.html#compare=d_id100_negrich@did100lane,BOA_F_nonneg32_s4004,D_guard12_p999@dguard2,fc2_372_S228_H128_s4004

---

## D1. The new model: spend once more on floors, or split ranker and dial?

**Evidence.** The constrained MLP (228 servable slots, bias-free non-negative distance head, ladder hinge) holds the dial CONTRACT on every seed (above-identity 1,642 → 0, ties → 0) and ranks inside the 944 leaders' band (CID22 0.884 vs D 0.863, +0.020 CI-clean; CSIQ +0.054; AIC-3 +0.016; LIVE tie) at 0.85–1.01× D's speed with the tree corruption head reading 100 %. It fails the ship rule on two independent clauses: **per-codec floors** (5/5 codecs; on the ladder ruler every MLP arm collapses on `avif-rav1e` at 0.13–0.21 vs the bar 0.641 — the D lineage is the ONLY bake family on the board passing all five) and **KonJND** (−0.062 vs D, CI-clean). The floor null is narrow, not open: architecture, loss and hinge weight never moved floors, but the matched A/B that adds floor-reaching anchor ladders moved **4 of 5** codec floors, and the sole non-mover is the codec with no anchor data. Constraint cost −0.0091 CID22 pooled / −0.0052 per-ref.
- record: https://github.com/imazen/zensimx/blob/main/benchmarks/best_of_all_2026-09-06.md (§5.21 ship table, §6 decision)
- plan: https://github.com/imazen/zensimx/blob/main/docs/PLAN_BEST_OF_ALL_2026-09-06.md
- ladder ruler on the board: https://github.com/imazen/zensimx/blob/main/benchmarks/board_ladder_ruler_2026-09-06.md
- the fast-class antecedent: https://github.com/imazen/zensimx/blob/main/benchmarks/fastclass2_campaign_2026-09-05.md

**Options.** (a) ONE more floor-data iteration: extend the anchor ladders to all five codec families at the encoders' true lowest settings through the persisted ladder pipeline (≈30 min fleet encodes + ≈half a day of lanes), densify the bottom window, refit nonneg+hinge at k=3, re-gate. (b) SPLIT: the MLP ships as the ranker (board/offline), D stays the dial. (c) Both.
**Sub-ruling D1b.** Is KonJND a blocking axis for the D slot? It is |SROCC| on 504 pairs, variance-dominated (seed spread 0.13), ssim2 itself reads 0.527; nine mechanisms failed to move it; the only untried input is the squintly human near-threshold study.
**Recommendation.** (a) once — the mechanism is measured and rav1e is where anchor coverage is thinnest; if floors reach the bar, D1b decides whether the model ships or waits for human data.
**Next action on ruling.** Launch the anchor-extension wave (pre-registered), or promote the split on the board and cookbook.

---

## D2. The contrast bound (F17) and `SHIPPED_REVISION` for Profile D

**Evidence.** Revision 2 = five inert eras behind one switch: F4 clamp (bit-identical on every local pixel), F5 route fix (944 bakes only), F17 `contrast_inc` bound (12 basic slots; the ONLY unbounded feature that fires: 36,466 on safesyn, 122 LIVE rows > 100), F18 deterministic roots, F19 deterministic pow. On the shipped sparse D chain EVERY bounded F17 form costs CID22 (log1p −0.0009 … satexcess −0.0046, CI-clean) and drops the `avif-rav1e` floor below rev1's; the registry keeps `SaturatingExcess` (best for the wide lasso class, worst for the sparse one — one feature definition, no per-class forms). The model-side alternative — winsor guard on D's TWO exposed slots + spline refit — is READY: rank tie to the digit (CID22 CI [0,0]), floors equal on all five codecs, contract 6/6, 0 CID22 rows clamped, no era break; honest limit: a rank no-op, and −0.0099 outlier ordering on the 17 LIVE rows it fires on.
- D-class arms: https://github.com/imazen/zensimx/blob/main/benchmarks/rev2_d_arms_2026-09-06.md (§10 trade, §11 guard-refit, §12 shipped-bake candidate)
- refit at rev2: https://github.com/imazen/zensimx/blob/main/benchmarks/rev2_refit_2026-09-06.md
- F17 decision: https://github.com/imazen/zensimx/blob/main/benchmarks/feature_rev2_2026-09-05.md (§11)
- F4 arm decision: https://github.com/imazen/zensimx/blob/main/benchmarks/f4_arm_decision_2026-09-05.md
- staged candidate: `/mnt/v/output/zensim/rev2-d-arms-2026-09-06/guard/shipped/d_sdr_add156_id100_negrich_guard12_2026-09-06.bin` (1,523 B, sha256 `c2db9e29ac97ad6f…`, `_MANIFEST.json` beside it; install recipe in the record §12.11 — it re-points `dense_bake_flip_gate`, never weakens it)

**Options.** (a) INSTALL the guarded D now: bounded inputs, no era break, rank/floors unchanged. (b) Ship revision 2 to D: era break + recalculation are done, but −0.005 CID22 and a rav1e floor below the bar — not shippable under the floor rule unless accepted. (c) Leave D exactly as it is; revision 2 stays the registered definition for new fits of the wide class (per-bake revision, phase 5).
**Recommendation.** (a), and keep (c)'s per-bake rule: `SHIPPED_REVISION` flips per bake only when a rev2-fit bake ships (a winning D1 model would be rev2).
**Next action on ruling.** Install lane with the flip discipline (weights + manifest + profile.rs + tests + CHANGELOG + board), or nothing.

---

## D3. Profile C / CHdr serving toggle (`append2_dst_activity`)

**Closed September 7:** serve canonical activity OFF, declare C/CHdr IDs,
and retire both old plan derivations. [Evidence](../benchmarks/feature_plan_cleanup_2026-09-07.md).
The options below are the earlier decision record.

**Evidence.** `from_block_profile`'s `everything` fallback serves C/CHdr with `append2_dst_activity: true`; their training extraction ran it OFF (production verdict 2026-08-02). Measured on one CID22 pair: C 0.866 and CHdr 0.311 zensim points of train/serve skew. Pre-existing; exposed by densify. It blocks densifying C/CHdr and deleting the last two dead paths (`from_block_profile`, `wide_bake_v2_read`). C/CHdr are candidate profiles behind a default-on feature.
- inventory: https://github.com/imazen/zensimx/blob/main/benchmarks/cruft_inventory_2026-09-06.md
- plan: https://github.com/imazen/zensimx/blob/main/docs/PLAN_CRUFT_PURGE_2026-09-06.md
- the toggle's adjudication: https://github.com/imazen/zensimx/blob/main/benchmarks/bandvis_dst_activity_2026-08-02.md

**Options.** (a) Serve OFF to match training (one line; moves C/CHdr scores by the amounts above). (b) Keep as-is, documented. (c) Retrain C/CHdr with the toggle ON.
**Recommendation.** (a): a runtime that disagrees with a model's own training is a defect. Then densify C/CHdr, delete the two dead paths, re-verdict both on the board.

---

## D4. The corruption head: public API and wiring

**September 7 update:** complete corruption composition now serves through
`BakeScorer`, including tree and linear heads, and candidate evaluation uses
that surface. [Implementation record](PLAN_CRUFT_PURGE_2026-09-06.md).
The older named-profile proposal below is not the implemented API contract.

**Evidence.** The right head class is a gradient-boosted tree on D's 156+peaks features: 98.9 % detection / 1.23 % honest FP / 2.38 % near-lossless FP at T = 0.9, no dial guard needed (the logistic: 86.0 / 11.4 / 50.0); leaving the top chromatic families out of training buys −4.3 pt FP / +12.9 pt detection at matched FP. Servable: `ZCTH` v1 format, Rust evaluator bit-exact vs sklearn (0 ulp decision function on 35,607 rows), 659 ns/compare (0.63× D's own forward), `bake_verdict --corruption-head` reproduces the gate end-to-end (671/672). Wired into evaluation; NOT into the runtime (doc-hidden behind feature `corruption-head`, public-API delta zero).
- serving: https://github.com/imazen/zensimx/blob/main/benchmarks/corruption_head_serving_2026-09-06.md · plan §3 (signatures): https://github.com/imazen/zensimx/blob/main/docs/PLAN_CORRHEAD_SERVING_2026-09-06.md
- theories: https://github.com/imazen/zensimx/blob/main/benchmarks/corruption_head_theories_2026-09-06.md
- rev1 head: https://github.com/imazen/zensimx/blob/main/benchmarks/corruption_head_d_2026-09-05.md

Proposed public surface (0.3.0 batch, feature `corruption-head`): `CorruptionHead::{from_bytes, declared_feature_ids, caller_input_width, n_trees, schema_hash, deadband, deadband_score, probability, probability_f64, score, score_f64, decision_function, verdict, check_servable_by}`, `CorruptionVerdict {probability, head_score, fired, perceptual_score, gated_score}`, `CorruptionHeadError` (11 variants), `fn gate_score(perceptual, head_score, deadband_score) -> f64`, `Zensim::{with_corruption_head, corruption_head, corruption_verdict}`.
**Options.** (a) Approve as proposed (feature-gated, never on the default compute path, head ids must be ⊆ the profile's plan). (b) Approve with changes (name them). (c) Evaluation-only for now.
**Sub-rulings.** Default deadband T = 0.9 (0.95 also measured); adopt the chromatic leave-out variant as the shipped head after a k-seed check.
**Recommendation.** (a) + ship the tree head as D's companion, default detached.

---

## D5. Deterministic transforms in `zenpredict` (sibling repo)

**Evidence.** The product path is libm-free end to end on the opt-in arm (features: 21 → 0 cross-libc differences; score: 1 → 0), and the validate side now calls the metric's own arithmetic. The LAST exposure is inside the frozen predictor crate: `zenpredict::feature_transform` (`signed_cbrt`, `signed_pow`, `soft_clip`, `yeo_johnson`, the `log`/`log1p` family, `Sinusoidal`) calls `cbrt`/`powf`/`ln`/`ln_1p`/`sin`/`cos` — live in Profiles A, BHdr and C; B and D are clean (winsor only).
- owner table: https://github.com/imazen/zensimx/blob/main/zensim/src/det_math.rs (module doc)
- consolidation: https://github.com/imazen/zensimx/blob/main/benchmarks/score_owner_consolidation_2026-09-06.md

**Options.** (a) Yes: the same deterministic treatment inside zenpredict (internal, no API change; zenpredict v3 is unpublished on `main`), cross-libc gate extended to A/BHdr/C. (b) No: register A/BHdr/C as carrying cross-libc score variance.
**Recommendation.** (a) — it is a sibling-repo edit, so it needs the explicit yes.

---

## D6. The pinned-score gate's tolerance (0.16) vs per-architecture pins

**Evidence.** The gate that caught every-profile mis-scoring under `default-features = false` (B 48.2 → 13.5; D 48.4 → −213) had its bar derived on one architecture (x86-64 tiers agree to 2e-5). Measured across the four CI arms there are two arithmetic classes: x86-64 vs the i686-scalar and wasm32 tiers differ by up to **2.8e-2** on near-identical pairs (i686 and wasm bit-identical to each other) because those tiers' `mul_add` is unfused; the smallest mis-serve on the gate's own cells is **0.945**. Bar re-derived to **0.16** (5.7× above noise, 5.9× below the defect); within one architecture the serving matrix still demands bit-exactness. Kernel-side fusing was tried and rejected (shifts variance terms, breaks cross-platform equivalence); the root cause is upstream in magetypes.
- record: https://github.com/imazen/zensimx/blob/main/benchmarks/dense_serving_ungate_2026-09-06.md (§2d)
- commit: https://github.com/imazen/zensimx/commit/0c6307a74d37

**Options.** (a) Keep the single 0.16 bar. (b) Per-class pins (x86-64 tiers at a 1e-4-class bar; i686/wasm their own pins). (c) Fix upstream (fused `mul_add` in magetypes' scalar/wasm tiers), then tighten.
**Recommendation.** (a) now, (c) as an archmage/magetypes item.

---

## Backlog rulings (unchanged, listed once)

- **Seven stale sibling workspaces** found in the wild (`zensim--avifgen`, `--era2-flip`, `--freefeats`, `--gaddrinst`, `--sparsehf`, `--v47pin`, `--waver4`): left alone per the rule; say the word to `jj workspace forget` + remove after a clean-status check.
- **Publishes (all user-gated):** zenanalyze-api 0.1.1 (additive); zenavif-parse 0.7.0 (breaking, from head); zensim 0.3.0 — QUEUED BREAKING CHANGES list in https://github.com/imazen/zensimx/blob/main/CHANGELOG.md, still blocked on https://github.com/imazen/zensimx/issues/46. Note the one public field added this week: `V2NewFeatureToggles::formula_revision` (struct not `#[non_exhaustive]`).
- **Squintly near-threshold study** — the only untried input for KonJND (D1b); needs the user's hours.
- **AVIF hold** — lifted only for the ladder/anchor encodes with the new zenav1-svt (measured 1.50×, not 2×); tuning waves stay parked.
- **Rotate the OPENAI key** that a lane printed into a transcript on 2026-09-04 (recommended then; not confirmed).

Related pages: JXL floor ladders http://localhost:3300/zensim/dpeaks372-2026-09-05/jxlfloor/ladders/index.html · D's worst inversions (encoder-confirmed vs dial-only) http://localhost:3300/zensim/ladder-2026-09-05/inversions/index.html
