> Historical snapshot archived September 7, 2026. Current workflow: [WAVE_PLAYBOOK](../WAVE_PLAYBOOK.md). Later rulings and current code supersede these dated commands and statuses. Exact original bytes remain in Git at `9879bbfd53b8`; relative links here are adjusted.

# zensim

**Current entry point:** [`SESSION-RESUME.md`](../../SESSION-RESUME.md). For the
one user-facing score and current profile/bake mapping, read
[`docs/CODEC_TARGET_METRIC.md`](../CODEC_TARGET_METRIC.md). Dated findings
below may be superseded; check the later correction and the implementation
before treating an old measurement or instruction as current.

**User ruling 2026-09-07:** new models must execute/serve entirely in Rust,
be wired to a zensim surface API, and be evaluated through that API, including
all heads, corruption gates and splines. Python remains useful for invention
and independent references. No consumers have calibrated to B/C/D; all may be
improved/replaced. Check `../zenpapers` and dated summaries before fundamental
research changes. Feature ablation is low priority when it saves no compute.
The user authorized the listed cruft removals/changes; retain their evidence
and caller checks. Canonical status alone does not prove correctness;
see [cleanup plan §0](../PLAN_CRUFT_PURGE_2026-09-06.md#0-establish-correctness-before-choosing-what-survives).

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

**Feature-gap map (read before feature work):**
`~/work/zen/zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` — the
2026-07-26 audit of the 720-feature set (v1-vs-v2 iw/masked naming trap,
ranked weaknesses, fast-CPU candidates with evidence, regime-inversion
finding, don't-build list). The f720+ append block
(`benchmarks/v2_append_block_2026-07-26.md`, `FeatureRegime::Folded720Append`)
implements its A1-A5/A9 candidates.

## ⛔ PUSH ONLY THROUGH `scripts/safe_push.sh` — a bare `jj git push` clobbers other lanes (2026-09-05)

**Do NOT run `jj bookmark set main -r @ && jj git push --bookmark main` in this
repo. Run `scripts/safe_push.sh` instead.** (`-r <rev>` for an explicit target,
`-b <bookmark>` for a bookmark other than `main`, `--dry-run` to check without
pushing, `--self-test` to prove the guard still refuses a sideways target.)

**Why it is a hard rule, not a preference.** With five-plus lanes pushing to
`main` concurrently, `jj bookmark set` + `jj git push` performs a **non-fast-forward
push with no prompt and no warning** whenever `@` does not descend from
`main@origin`: the bookmark moves SIDEWAYS and every commit that was only
reachable from the old tip becomes unreachable from the new one. The objects
survive; nothing points at them. The pushing lane's own `jj log` looks correct,
so the loss is invisible until somebody asks
`git merge-base --is-ancestor <sha> origin/main`.

**MEASURED 2026-09-04:** `main@origin` moved sideways **twice in one afternoon**
(16:58:53 and 17:08:29 MDT, jj ops `db7c8ca86b69` and `0edf97e28a91`). The second
move dropped **nine commits from six lanes**, including `d3a948ca` — the G-ADDR
board-coverage feature, +555/−23 across six files. The boards on `/mnt/v` had
been generated WITH that code, so the next regen from `main` would have silently
un-drawn **46 NOT-SHIPPABLE badges** and dropped `promote_fulleval.py
--graft-gaddr` and `cut_gaddr_negtail_probe.py` from the tree. Nothing failed;
the numbers would just have been wrong. Full record:
[`benchmarks/push_clobber_2026-09-05.md`](../../benchmarks/push_clobber_2026-09-05.md).

`safe_push.sh` does: **fetch → assert `<bookmark>@origin` is an ANCESTOR of the
target → bookmark set → push → verify the target landed.** On a sideways target it
**exits 3, prints every commit the push would drop, and does not touch the
bookmark.** There is no `--force`; the fix is `jj git fetch && jj rebase -d
main@origin`, resolve keeping BOTH lanes' hunks, then re-run. **Equally prohibited: `jj bookmark set --allow-backwards` (and any `--allow-*` flag that lets a bookmark move sideways), `jj git push --force`, and `git push --force*`.** The 2026-09-06 transcript audit (`benchmarks/agent_transcript_audit_2026-09-06.md`) found that BOTH 2026-09-04 sideways moves were produced by exactly that flag after jj had refused the move twice — the refusal IS the guard; overriding it is the incident. If a push is refused, the only correct action is the rebase above; there is no flag that makes a sideways move safe. The post-push verify
exists because a push can report success and not land (2026-05-29 orphaned-bookmark
incident). Guard self-test: 4 cases including the negative control (sideways
REFUSED, remote provably unmoved) — `scripts/safe_push.sh --self-test`.

**A SECOND GATE runs in the same wrapper: `hygiene: address/identifier check`.**
It scans the lines the push would ADD for the identifier classes that must not
enter a public repo (private-range addresses, hardware addresses, household
framing), and **exits 7** — a different code from the sideways refusal's 3, so a
caller can tell them apart. There is no bypass flag, for the same reason there is
no `--force`. The patterns have ONE owner,
[`scripts/lib/hygiene_patterns.txt`](../../scripts/lib/hygiene_patterns.txt), shared
with `scripts/lint_scripts.py` (which runs the same classes over every tracked
file, so an INHERITED hit is found and not only a newly added one) and pinned by
7 tests in `scripts/test_lint_scripts.py`; both run in CI. Site-specific
additions (names, labels) are NOT in this repo — they come from
`hygiene_patterns` in the private homefleet config (`HOMEFLEET_NODES`, default
`~/work/zen/homefleet/zenmetrics/fleet/nodes.toml`), because a public pattern
list that spelled out the values it protects would leak exactly what it exists to
keep out. **MEASURED on the trees it was built against: 0 false positives across
2,832 tracked zensim files and 2,040 zenmetrics files, and 13 true positives** —
all one class, the stale WSL-gateway form of a gallery URL, which the memory
record had already flagged as going stale and which nothing had ever checked.
Guard self-test: `scripts/safe_push.sh --self-test` is 6 cases, including the
negative control (CASE 6: the canonical `localhost` form of the SAME line is
ACCEPTED and lands — without it, a gate that refused everything would also pass).

Two jj traps the script already handles, which hand-rolled one-liners keep
re-learning: a successful push makes `@` immutable and jj creates a **fresh empty
`@` on top**, so `-r @` one command later targets the wrong commit; and a jj
**workspace has no `.git`**, so read-only `git` verification must run as
`git -C ~/work/zen/zensimx …` against the primary checkout.

## Known Bugs

- **⛔ THE SCORE, NOT ONLY THE FEATURES, IS LIBC-DEPENDENT ON REVISION 1 (F19,
  measured 2026-09-06; OWNER + ERA LANDED, FLIP NOT TAKEN).** `metric.rs` calls
  `powf` at exponents that are not powers of two — `score_mapping_b` is **0.7
  on every shipped profile**, plus `0.5979`/`1.2244`/`0.6130` in the `approx_*`
  public helpers and a bake's p-norm `p` in both head runtimes — and `exp` in
  the bounded squash / soft clamp / tanh pin / both α gates, and `log2` in the
  four `--mlp-size-axes` MLP **inputs**. None of those is correctly rounded, so
  **a SCORE is a function of which libc the binary linked against.** MEASURED
  on the 2×2 cross-libc gate (one commit, glibc vs static-musl, 220 procedural
  cells): revision 1 differs on **1 of 220** scores — and **so does
  `root=sqrt, pow=libm`**, i.e. **F18's fix left the score exactly as
  libc-dependent as it found it.** Revision 2 (`root=sqrt, pow=pure`) reads
  **0/81,840 features and 0/220 scores**. Owner
  `zensim::det_math::PowForm`/`DetPow`, era **`scorepow`** on
  `FormulaRevision::Rev2`, override `ZENSIM_POW_FORM=libm|pure`; **INERT BY
  DEFAULT** (`PowForm::default()` is `LibmPowf`, `SHIPPED_REVISION` is `Rev1`),
  so no shipped byte moves. Unlike F18 **the arm is CHOSEN, not derived** —
  `x^0.7` has no correctly-rounded closed form the way `x^(1/4)` does, so the
  only purchasable property is that every target run the same source
  (`libm::{pow, exp, log2}`, already in the dep graph, no `fma`; the only arch
  `select_implementation!` among the three is on `exp` and is gated
  `x86_no_sse`, which no shipped target selects). Error bound over 6,611 rows of the score's
  own domain against a 60-digit reference: platform libm **1 ULP**, the port
  **1 ULP**, `magetypes::nostd_math::powf_f64` **7.2e12 ULP**, a *perfectly
  rounded* f32 pow **1.4e10 ULP** — so `log2_midp_precise`/`exp2_midp_precise`
  cannot be reused (f32-and-SIMD-only; even a perfect f32 route underflows the
  head p-norm tail to `0.0` at `p=6, x=1e-12`). **Same correction as F18: NOT
  more accurate** — the two arms disagree on 7.911 % of rows and glibc is
  nearer the truth on 520 of 523. **STILL OPEN, registered not fixed:** (a)
  `zenpredict::feature_transform`'s `cbrt`/`powf`/`ln`/`ln_1p`/`sin`/`cos` are
  on the **PRODUCT** path via `predict_transformed` and are LIVE in Profiles
  **A, BHdr and C** (**B — the default — and D declare only `winsor_p99`, a
  clamp, and are clean**); it is a sibling repo this lane must not edit.
  **(b) IS FIXED 2026-09-06 — the validate-side mirror is DELETED, not routed.**
  It read: *"`zensim-validate::bake_runtime` and its `bake_compare` fork
  re-implement both head runtimes, document themselves bit-exact with
  `metric.rs`, do not follow `PowForm`, and no test holds them together — a
  BLOCKER on flipping `SHIPPED_REVISION`."* Every word of that was right, and
  the prediction was **MEASURED before the fix**: `bake_verdict --full-json` on
  six shipped/board bakes × `cid22,kadid,tid,konjnd,aic3` was **byte-identical
  under `ZENSIM_POW_FORM=libm` and `=pure` on all six** — the evaluation tooling
  could not see the form the product obeys. The arithmetic now has ONE owner,
  `zensim::score_math` (`#[doc(hidden)] pub`, zero supported-API delta), reached
  by both `zensim::metric` and the validate runtime; `zensim::det_math` is
  `#[doc(hidden)] pub` too, which is the surface F19 named as the fix and put out
  of its own scope. **No number moved** — all six default-arm verdicts are
  byte-identical to the pre-change binary's — and after the fix the two
  per-sample-α bakes move across the arms (430 leaves each) while the four
  head-less ones correctly do not. Gates:
  `zensim-validate/tests/{score_owner_parity,no_score_path_libm}.rs`, both
  mutation-verified. The output-calibration spline is **clean of `PowForm`**
  (`powi` only) but its validate-side copy had a DIFFERENT divergence, also
  closed the same day — see the entry below. Record:
  `benchmarks/score_path_libc_determinism_2026-09-06.md` +
  `benchmarks/score_owner_consolidation_2026-09-06.md`; ledger §3.52 + §3.54.

- **⛔ THE FEATURE EXTRACTOR IS LIBC-DEPENDENT ON REVISION 1 — a musl build and
  a glibc build of the SAME COMMIT disagree (found 2026-09-06; owner + era
  landed same day, the FLIP is NOT taken, so revision 1 is still exposed).**
  `powf` is not correctly rounded and no standard makes two libcs agree on it,
  so `(Σx⁴/n).powf(0.25)` and `(Σx⁸/n).powf(0.125)` make a feature a function
  of **which libm the binary linked against**. MEASURED: static-musl vs glibc,
  same source, **77/322,152 csiq cells (0.0239 %)** and **328/1,116,000 tid
  cells**, every delta exactly one ULP at f64; a libm-free probe closed the
  mechanism (`x ** 0.25` over 400,000 doubles: 276 disagreements glibc↔musl,
  **0** between two glibc versions). **This is why the fleet's Feature executor
  had to be rebuilt against glibc (`exec-featrev2glibc-88477e38`).**
  - **What is in tree:** one owner `zensim/src/det_math.rs`
    (`RootForm::{LibmPowf, NestedSqrt}` + the `DetRoots` trait), defect
    `DEFECT_F18`, era **`v1detroot`** on `FormulaRevision::Rev2`, measurement
    override **`ZENSIM_ROOT_FORM=libm|sqrt`** (same byte length, deliberately).
    `RootForm`'s default is `LibmPowf` and `SHIPPED_REVISION` is `Rev1`, so
    **no shipped byte moves**.
  - **The era is 156 slots, NOT the 144 the discovery record priced** — the
    `(M4/n)^0.25` form is also the **v2 `ssim_dev4`** slot, in three more
    finalizers that record's v1-block table did not reach. Registry answer is
    derived from each slot's own `Statistic` (`L4`/`L8`), never a second list.
  - **⚠ Flipping rev2 now invalidates any `ZENSIM_FORMULA_REV=2` table
    extracted BEFORE 2026-09-06** (the R6b lane has some).
    `ZENSIM_ROOT_FORM=libm` reproduces those exactly from a new binary.
  - **The rest of the SDR feature path is ALREADY libc-free, by reading the
    sites** (stronger than the corpus inference the discovery record used):
    sRGB→linear is `linear_srgb`'s LUT, the opsin cube root is
    `color::cbrtf_fast` (bit-trick + Halley, `mul_add` only) and magetypes'
    `cbrt_midp`, the SIMD PU-XYB path is magetypes'
    `log2_midp_precise`/`exp2_midp_precise`, and every other v1 pool is `sqrt`
    (IEEE-required correctly rounded). `powi` never reaches libm.
  - **STILL EXPOSED AND UNFIXED: the SCORE.** `metric.rs`'s raw-distance →
    score mapping calls `powf` at `0.5979 / 1.2244 / 0.6130 / b` — none a power
    of two — so **the dial value is libc-dependent on every profile**. No
    `sqrt` composition exists for those exponents. The cross-libc dump
    instrument emits the score beside the features so it stays measurable.
  - **⚠ CORRECTION to the discovery record:** it says the composition is
    "*more* accurate than one `pow` call". **FALSE** — `sqrt∘sqrt` rounds
    twice. Over 4,000 log-uniform doubles vs a 60-digit reference the two agree
    on 3,455; of the 545 that differ (always exactly 1 ULP) **glibc's `pow` is
    nearer the truth in 544 and the composition in 1**. The case for the fix is
    **determinism and a bounded error**, not accuracy.
  - **THE GATE PASSES, MEASURED 2026-09-06** (`just check-cross-libc` /
    `scripts/verify_cross_libc_features.sh`): one commit built for `-gnu`
    (dynamically linked `libm.so.6`) and `-musl` (`static-pie`), 220 cells ×
    372 features = 81,840 values over the 20-cell parity matrix + a 200-cell
    procedural distortion ladder. **Revision 1: 21 differing (0.0257 %, inside
    the fleet corpora's own 0.0239-0.0294 % band). Deterministic arm: 0.**
    The script FAILS if revision 1 shows no difference — a zero proves nothing
    from an instrument that is not demonstrably sensitive. Toggling the arm on
    ONE binary moves 4,097/81,840 rows over **exactly 144 slots**, and all 20
    libc-divergent slots are inside them, so the era covers the exposure
    exactly. **The score differs on 1 of 220 cells in BOTH arms** — the
    unfixed `metric.rs` exposure, now measured rather than suspected. Pinning
    test: `det_math::tests` (bit-exact values, independently re-derived with
    60-digit `Decimal.sqrt`). Records:
    [`benchmarks/libc_determinism_2026-09-06.md`](../../benchmarks/libc_determinism_2026-09-06.md)
    (landing + audit + gate),
    [`benchmarks/libm_pow_nondeterminism_2026-09-06.md`](../../benchmarks/libm_pow_nondeterminism_2026-09-06.md)
    (discovery).

- **★ FEATURE REVISION 2 IS LANDED AND INERT — `ssim_form::SHIPPED_REVISION`
  is still `Rev1`, and flipping it is gated (2026-09-05).** Both of the
  audit's arithmetic defects now have fixes in tree behind
  `FormulaRevision::Rev2` (`ZENSIM_FORMULA_REV=1|2` pins an era from one
  binary). Nothing shipped has moved: the 22,397-row `to_bits()` dump is
  byte-identical, sha256 `940c82dd0499d3ad…`. Read
  [`benchmarks/feature_rev2_2026-09-05.md`](../../benchmarks/feature_rev2_2026-09-05.md)
  before citing any F4/F5 number. Four things there supersede what is written
  elsewhere in this file and in the audit:
  - **F4's blast radius is 132 slots, not 72.** The audit scopes it to masked
    + IW; a bounded-form re-extraction MEASURES basic 36 + peaks 24 + masked
    36 + IW 36. The peaks pair (`ssim_max`, `ssim_l8`) is an every-third-slot
    comb no per-block reading predicts. The registry assertion was WIDENED.
  - **The no-C1 form is genuinely ssimulacra2's** — verified against
    `fast-ssim2`'s own kernels, not assumed from `lib.rs`. So a bounded form
    is a deliberate deviation from that lineage. (`fast-ssim2` carries the
    same unbounded term; different repo, NOT touched, reported only.)
  - **F4's arm is NOT chosen, and the ladder instrument cannot choose it** —
    all four arms give identical monotonicity on both control-passing ladders
    (132/132 noise, 116/132 quantize), because the ladders never reach the
    regime where the arms differ. The decision needs a real monotone-linear
    fit on real corpora (plan R6). Do not infer an arm from a ladder result.
  - **⛔ "F5 is free" is RETIRED.** Fixing F5 requires changing the APPEND
    route, and `bake_block_profile` MEASURES all three shipped 944 bakes
    (`c_sdr_mlp944_corrmix` = Profile **C**, `c_hdr_l1t1944` = **CHdr**,
    `c_sdr_purity944`) reading the full `GLOBAL_*` set — 11 `DMEAN` + 11
    `CGAIN` + 11 `CLOSS` each. `candidate-profiles` is default-on and C's bake
    ships to crates.io. Flipping rev2 moves 22 of those 33 per bake (`DMEAN`
    is untouched, by construction and by test), so C and CHdr need
    re-verdicting. Phase 2b's G2b.2 precondition is FALSE.
  F5 itself is FIXED behind the flag: paired free-vs-append disagreement past
  the 2e-5 bar goes **8.30 % → 0.18 %** (93 → 2 of 1,120 cells), worst
  2.31e-4 → 2.49e-5. Three plausible fixes were measured and FAILED first
  (per-row granularity; Kahan on the free route; Kahan on both) — the
  registry's own proposed "compensated accumulation" is among them. The
  working fix is reassociation of BOTH terms of `gvar2 − gvar1` into per-pixel
  differences; doing only the second moment leaves `GLOBAL_CGAIN` flat.


- **⛔ 92 % OF THIS REPO'S BAKES CANNOT BE SERVED BY `Zensim::compute`, AND TWO
  SHIPPED PROFILES ARE AMONG THEM (measured 2026-09-05, OPEN — the architecture
  lane owns the fix).** Census through the production entry on the committed
  golden pair: **400 of 433 board bakes, 3 of 11 `zensim/weights/*.bin`, 11 of
  14 registered feature sets, and 2 of 10 selectable profiles are REFUSED**,
  every one with the identical `ModelForwardFailed { "bake declares more input
  features than the caller supplied" }`. ONE cause: `Zensim::compute` emits a
  **372-layout** vector with `free_extras: Off`
  (`compute_folded_v1_372_streaming_impl` builds toggles with
  `..Default::default()`; `compute_fold_backed` truncates to ≤372;
  `wide_bake_v2_read`, which would pick the wider read, is `allow(dead_code)`),
  so the rule is exactly **`caller_input_width() <= 372` serves**. Declared
  widths on the board: 4×156, 28×372 (the whole SERVED set), 1×504, 8×720,
  2×924, **389×944**. **`ZensimProfile::C` and `CHdr` are the sharp end** —
  `candidate-profiles` is **default-ON** and `c_sdr_mlp944_corrmix_2026-08-05.bin`
  is in the crates.io `include` list, so a consumer writing
  `Zensim::new(ZensimProfile::C)` gets a hard error on every image. **The
  identity short-circuit hides it at exactly the input a smoke test uses**: both
  refused profiles still return `IDENTITY (ref vs ref) score = 100.000000`,
  because `mark_identical` fires before the model. Every SERVED case is also
  **served-but-MISMATCHED**, with no new mechanism — the runtime is
  self-consistent (bit-exact, measured) while the 372 roots are one extraction
  era behind (see the §3.37 entry). **When the fix lands, wire the feature-set-id
  match check into the SERVING path, not only the verdict path** — otherwise it
  converts a loud refusal into a silent wrong number. Record:
  `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md` §4A; ledger §3.42; registry
  `profile-c-chdr-unservable-2026-09-05`.

- **⚠ THE 372 IDENTITY ALL-ZERO FEATURE VECTOR IS FABRICATED, NOT MEASURED
  (2026-09-05, OPEN as a claim-scoping issue).** Both product-facing SDR
  entries short-circuit `source == distorted` before any walk and synthesise
  `(score = 100, raw_distance = 0, vec![0.0; width])` —
  `metric.rs::identical_result` behind every `Zensim::compute*`, and the free
  function `compute_zensim_with_config` behind **both** v1-372 extractors. So
  `zensim-validate`'s `dial_addressability` constant — *"ref == dist yields
  all-zero features for every image"* — is at 372 a property of the
  short-circuit and is **unfalsifiable by construction at that width**.
  COMPUTED on the same pixels, the v1 block populates **144 of 372** slots (max
  |v| **1.12e-3**), and the 944 walk **286 of 944**, in exactly three classes:
  **15 reference-only** (`GRAD_SRC_MEAN`, `LUMA_MEAN_REF` — correct,
  `∂f/∂dist ≡ 0`), **12 `PJND_FRAGILITY`** (0.395 full-walk / exactly **1.0**
  v1-only — a formula artifact both ways), **259 fp residue**. Consequence for
  the dial: `Zensim::compute` returns exactly **100.000000000** at identity but
  **96.2296** when ONE byte in ONE channel of ONE pixel out of 90,000 changes —
  a **3.77-point step at zero distortion**, which IS the mechanism behind
  G-ADDR's "shipped B ranks 266 of 4,424 cells above a perfect copy" and its
  `C2 ⊻ C6` either/or. Gate:
  `zensim/tests/feature_invariants.rs::identity_is_fabricated_by_the_short_circuit_and_differs_from_the_computed_vector`.
  Registry: `identity-score-cliff-fabricated-2026-09-05`.

- **⚠ `V1FreeExtras` IS SILENTLY INERT UNLESS `append_block` IS ALSO DECLARED
  (2026-09-05, gated as a contract).** `append_block` does double duty: it
  declares the LAYOUT (720 → 924, with `append2_block` → 944) **and** enables
  the append COMPUTE. Every raw-moment slot lives at `f720+`. So a `v1_only`
  walk requesting `V1FreeExtras::RawMoments` **without** `append_block` emits a
  **720-wide** vector in which those slots do not exist, with a populated-slot
  count **identical to `V1FreeExtras::Off` (228 vs 228)** — no error, no
  warning. Same failure shape as the fixed `==`-vs-`!=` emission-gate defect,
  reached by a different route; a training table full of structural zeros with
  nothing failing. Class C is only PARTLY affected — its twelve v2-348 `MSE`
  cells are inside the 720 layout and survive (228 → 240); its twelve
  `LUM_*_ERR` append cells do not. The correct "156+free" shape is
  `v1_only: true` **plus** `append_block: true, append2_block: true`. Gate:
  `free_extras_are_silently_inert_without_the_append_block_declaration`.

- **⚠ `docs/FEATURE_SET_IDS.md` §1 failure #9 IS AN ERA ARTIFACT, NOT A CODE
  CLAIM (corrected 2026-09-05).** That row reads *"the v1-372 `f0..155` is NOT
  the 944 fold's `f0..155` — 156 of 156 slots differ, max abs 1.0214"*, which
  compared two **stored instruments built in different extractor eras**. In one
  process at one commit on the same pixels they do not differ at all: **372 of
  372 slots bit-identical at 11 geometries** (tight, non-tight, odd, sub-64,
  past `H_TILE_WIDTH`) **and at both SIMD tiers**, through the public free
  function both v1-372 extractors call. The row is still a true warning about
  the INSTRUMENTS; read as a claim about the CODE it sends a reader hunting a
  divergence option C (`56bbcda2`) already closed. Gate:
  `extractor_entry_is_bit_exact_to_the_fold_v1_block`.

- **⚠ ~95 FEATURE SLOTS ARE NON-MONOTONE UNDER A CONTROL-VALIDATED DISTORTION
  LADDER, AND MOSTLY BY DESIGN (2026-09-05).** Over 944 slots × 12 images on
  two ladders whose own MSE control is monotone 12/12 (additive noise 4..48,
  quantization step 4..64), **40 and 55 slots respectively are persistently
  (≥9/12 images) and amplitude-really non-monotone**, concentrated in the v2 and
  append blocks. Most are correct: **62 of the violating series contain an exact
  `0.0` beside non-zero values** — the signature of a rectified ONE-SIDED
  feature (the `GLOBAL_CGAIN`/`GLOBAL_CLOSS` pair; heavy quantization with a
  `+step/2` reconstruction offset *increases* contrast, so contrast-LOSS
  correctly collapses to zero). **Anyone fitting a monotone head, a dial, or a
  per-slot sign constraint must exclude the persistent set by NAME** (it is in
  `/mnt/v/output/zensim/feature-audit-2026-09-05/ladder3.tsv`, with each slot's
  raw series), not by guess; the already-registered per-slot findings (`f162`'s
  local bump, `f161`'s sign disagreement, `f93`'s share of D's jpeg-floor
  inversions) are individual instances of a property the whole set has.
  **METHOD, load-bearing:** a THIRD ladder had to be discarded — repeated
  radius-1 box blur is itself **non-monotone on 12 of 12 images**
  (`29.13 → 26.02 → 29.11 → 31.62 → 34.53 → 37.66`) and produced **176 false
  violations** before the control was added. **Never accept a monotonicity
  result with no stimulus control.** Registry:
  `nonmonotone-feature-slots-are-by-design-2026-09-05`.

- **⚠ THE ONE LIVE ARITHMETIC DEFECT IN A SHIPPED FEATURE IS THE UNBOUNDED SSIM
  `d` (F4; re-surfaced by the 2026-09-05 audit, OPEN BY DECISION).** v1's
  per-pixel SSIM dissimilarity has a `.max(0)` floor and **no upper cap**, and
  `num_m = 1 − (mu1−mu2)²` carries **no `C1`**, so on high-magnitude chroma `d`
  reaches **5.8e6** — `f313` (`iw_ssim_4th` s0 ch2) = **5,814,302** across 2.3 M
  scanned rows, against a photographic p99.9 of **0.48**. The shipped winsor
  guard clamps the symptom, which is why B ships; the consequence to carry is
  that **144 features hold a clamped-outlier value on the weakest content
  class**, so any model fit WITHOUT that guard is mis-specified there. The
  denominator-cancellation hypothesis was tested and FALSIFIED (worst effect
  1.2×). Every OTHER audited engine property is clean at HEAD — determinism
  (incl. 28 threads), engine parity (33/33 bit-exact), cross-tier (0 cells over
  the golden tolerance policy), degenerate inputs (0 NaN / 0 Inf), width
  independence. Full inventory of all 27 defects with status and gates:
  `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`.

- **⛔ `contrast_inc` (v1 basic local 12, twelve slots) IS UNBOUNDED, and it is
  the unbounded value that ACTUALLY OCCURS — not F4 (found + measured
  2026-09-06, registered as F17, fix PROPOSED not landed).**
  `max(0, var_dst/var_src − 1)` divides by the SOURCE term, so unlike its two
  siblings — `var_loss` and `tex_loss`, whose numerators their own denominators
  bound at exactly **1.000000** — its numerator is not bounded by anything. The
  `var_src > 1e-10` guard is a threshold, not a stabiliser. MEASURED over
  **216,756 real pairs on 8 corpora, all 372 slots**: the twelve `contrast_inc`
  slots (**f12 f25 f38 f51 f64 f77 f90 f103 f116 f129 f142 f155**) are the **TOP
  TWELVE by maximum** (worst **36,465.74** safesyn, 3,598 LIVE, 928 TID, 618
  KADID) and the thirteenth slot in the whole vector is **1.972** — a partition,
  not a tail. That is **×105,127** the gold holdout's own p99.9 (CID22, 0.34687).
  **Unlike F4** (whose 5.8e6 lives in a bigcodec sweep with no local pixels and
  fires on ZERO of these rows), F17 fires on five distortion corpora and the
  training leg, on 2.59 % of cells above 1 and 0.0198 % above 100.
  **What this means for numbers you read:** the "winsor already clamps it"
  mitigation covers **Profiles A and B only**. **Profile D — today's SDR default
  — carries NO `feature_transforms` block at all** and reads f116 (max 1,380)
  and f155 (max 2,127) raw into a 28-input linear head; **CHdr reads all twelve
  at `identity`**; C has 10 of 12 winsor-guarded, BHdr 4 of 6. So a bake-side
  transform is not the answer — it is what is already deployed and what the
  default lacks. Blast radius is the **same twelve at every layout** (MEASURED at
  `944full`/`924`/`372`/`156`), unlike F4's 132-vs-36 pool-state split.
  **A FOURTH hand-copy is in another repo** —
  `zenmetrics/crates/zensim-gpu/src/pipeline.rs:1305-1310` — so a rev2 wave using
  the GPU oracle must land the matching change there first or pin the oracle to
  the CPU walk. **Serving a rev1 bake rev2 features is cheap for D**: SROCC
  |Δ| ≤ 6e-5, dial mean −0.0003…−0.012 and ≤0.64 % of pairs past the 0.5-pt bar
  — ~4 orders below the −4.98 extractor-era and −3.658 decoder-era shifts. **THE ARM IS DECIDED — `HfGainForm::SaturatingExcess` =
  `g/(g+1)`** (216,756 rows, five arms from one binary, two slices x two
  solvers): the only arm passing the structural gates (`bexcess` 263,195
  order inversions because it reads the MAGNITUDE not the ratio; `cap` 67,224
  new ties; `log1p` unbounded at 10.504), and it wins CID22 + AIC-3 with
  CI-excluding deltas in 2 of 4 variants. **MEASURED: LIVE 0.7357 -> 0.9500
  (+0.214)**, TID +0.033, KADID +0.021 — and **KonJND REGRESSES
  -0.013..-0.080 on EVERY bounded arm**, so that cost is a property of
  bounding, not of the arm; recovering it would be an APPEND slot, never a
  reason to keep an unbounded one. Dial does not regress (mono within 0.0011,
  tied 0.0000-0.0001) at a reach cost of 2.2-11.8. Owner
  (one place, five runtime arms, revision-1 BIT-IDENTICAL):
  `zensim/src/hf_gain_form.rs`. Registry: `F17` + the `v1hfgain` era on
  `FormulaRevision::Rev2` (which now batches **three** eras — a wave declaring
  only `v1ssimcap` + `freecomp` under-declares every table by twelve slots).
  Record: `benchmarks/feature_rev2_2026-09-05.md` §11; pre-registration:
  `docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.
  **⚠ THE ARM WAS DECIDED ON ONE MODEL CLASS, AND IT IS THE WORST ARM ON THE
  CLASS THAT SHIPS (measured 2026-09-06, `benchmarks/rev2_d_arms_2026-09-06.md`,
  pre-registration §12).** All five arms re-run through the SHIPPED Profile-D
  chain, ×2 slices, in-era: **every arm that changes the feature loses CID22
  with a CI excluding zero, and `satexcess` is LAST of the four at both slices**
  — −0.00456 / −0.00406 against `log1p` −0.00088 / −0.00211 and `bexcess`
  −0.00153 / −0.00206 (the two cheaper arms are exactly the two the structural
  gates kill). Every arm drops the `avif-rav1e` floor below rev1's 0.6667 while
  every arm IMPROVES `jpeg` 0.7179 → 0.7692. **`REV2_HFGAIN` still STAYS
  `SaturatingExcess`** — one feature has one definition and no arm wins both
  classes — but **do not cite R6b's +0.0027 as what rev2 does to a bake**:
  it is a property of a 147-coefficient fit, and the D lineage's 28–39 lose.
  MEASURED mechanism: **the two revisions do not read the same slots** —
  `D_ratio_s156` reads ONE F17 id (f77) of 28 active, `D_satexcess_s156` reads
  FIVE of 33 — so bounding changes which inputs L1 keeps, not just their scale.
  On the 60 LIVE rows whose rev1 `contrast_inc` exceeds 100, `satexcess` at
  a156 is WORSE than rev1 (+0.93203 vs +0.93491); the bound does **not**
  demonstrably fix the ordering where the pathology fires. **THE BAKE-SIDE
  ALTERNATIVE IS REAL AND CHEAPER: `winsor_p99 [p0.1,p99.9]` on the twelve F17
  slots ONLY, rev1 features (`W-f17`), is the only non-rev1 model in that study
  with A7r 0 fails AND contract 6/6 and the best outlier-row ordering of all
  eight, at 76 % of `satexcess`'s CID22 cost and NO era break** — but the SAME
  guard on all 372 slots (the shipped Profile-B recipe) is the WORST model in
  the study. **And `add-winsor` CARRIES the input bake's output spline
  verbatim**, so the literal `add-winsor → extend-top` chain calibrates on the
  un-clamped net: that variant ties rev1 on CID22 and wins CSIQ **+0.04862**
  but breaks contract **C6**. Ledger: `docs/DATASET_HISTORY.md` §3.50.
  **★ RESOLVED 2026-09-06 — THE SERVE-TIME GUARD WITH A REFIT SPLINE COSTS
  NOTHING** (`benchmarks/rev2_d_arms_2026-09-06.md` §11 addendum,
  pre-registration §12.7, ledger §3.51). Keep revision 1's WEIGHTS, declare the
  guard on the bake, and refit the spline on the GUARDED net — the carry defect
  is avoided by **ORDERING**, because `shared-anchor` forwards the anchor
  through `build_fw_ops` (the bake's own transforms). **`W-f17-refit-p999`
  (winsor `[p0.1, p99.9]` on the twelve `contrast_inc` slots, serve-time) is an
  EXACT CID22 TIE at both slices** — 0.863666 vs 0.863666, CI `[0, 0]`, where
  `W-f17` cost −0.00348 and `satexcess` −0.00456 — **holds rev1's per-codec
  floors EXACTLY on all five codecs, contract 6/6, inversions PASS, and WINS
  LIVE** (+0.00032 / +0.00022, CI-excluding), with no era break. It is the
  RECOMMENDED ship candidate (user decision; nothing installed). Prefer p99.9
  over p99/p95: **0 of CID22's 51,504 `contrast_inc` cells** exceed its window
  (831 / 1,480 at the tighter ones) and **p95 collapses f25's window to
  `hi = 1e-9`**, pinning that slot to a constant. **§10.2's registered
  follow-up is HALF right:** the refit DOES remove the C6 break (above-identity
  1 → 0, contract 5/6 → 6/6) and the CSIQ **+0.04862** survives — **but the
  `avif-svt` floor loss is NOT the spline** (0.9744 either way) **and the CSIQ
  gain belongs to the ALL-372 guard, not to F17** (scoped to twelve slots CSIQ
  is a tie). **Honest frame: on that lineage the guard is nearly inert** — the
  re-fit reads ONE of the twelve F17 slots (f77), clamping 0 CID22 rows — so the
  bound is measured FREE, not a rank win; `DEFECT_F17`'s exposure table names
  **f116 + f155** for the SHIPPED ADD156 bake, a different support, and that
  measurement is REGISTERED, NOT RUN. Tools: `bake_dial_refit add-winsor
  --slots` + repeatable `shared-anchor --anchor` (`e8e4ace2`), gated by
  `scripts/verify_winsor_scope_identity.sh`.
  **★★ RUN 2026-09-06 ON THE SHIPPED BAKE — READY, AND IT COSTS NOTHING**
  (`benchmarks/rev2_d_arms_2026-09-06.md` §12, pre-registration §12.8 — read its
  §12.8.0 order-honesty note — ledger §3.53). Two things you must know before
  reusing the recommendation above. (1) **The twelve-slot scope has no meaning on
  this support.** The shipped bake declares 28 ids of which exactly **{f116,
  f155}** are F17; the other ten carry exactly-zero weight, so after `densify`
  the twelve-slot and two-slot arms are **BYTE-IDENTICAL**. The r6b re-fit read
  **f77** — **the two lineages share NO F17 slot**, so §11a's numbers never
  applied here. (2) **`extend-top` is NOT in the D lineage**: the shipped 4,222 B
  wide bake IS `fit-lasso`'s direct output (sha `921a8f67…`), so a chain copied
  from §11a verbatim inserts a step the artefact never had. **Measured on the
  runtime-era postC root + the FLOOR-DENSE ladder instruments:** CID22 an EXACT
  TIE (paired bootstrap CI `[+0.00000, +0.00000]`), all twelve board corpora
  inside `8.2e-5`, contract **6/6**, every per-codec `A7r` floor EXACTLY the
  shipped bake's on all five codecs, identity 100.000000 with 0 of 9,593 cells
  above it, inversions 0.0053 unchanged, ZCTH-tree-head corruption gate
  `pass_q10 = pass_q20 = 1.0000` unchanged, **0 of 4,292 CID22 rows perturbed**
  (262 of 196,086 training rows). The control `Sctl` (refit, no guard)
  reproduces the shipped verdict with CI `[0,0]` on all seven bootstrap corpora,
  so every delta is the guard. **INSTALL CANDIDATE staged, NOT installed:**
  `d_sdr_add156_id100_negrich_guard12_2026-09-06.bin` (1,523 B, sha256
  `c2db9e29…`) under `/mnt/v/output/zensim/rev2-d-arms-2026-09-06/guard/shipped/`
  — **the install recipe FORCES a `dense_bake_flip_gate` change** (that gate
  asserts `to_bits()` equality against a twin over the retired WIDE bytes, and
  the candidate is deliberately unequal on a clamped input), which must be
  **re-pointed at the guarded wide bake, never weakened**. **Honest limits:** the
  guard is a rank NO-OP, not a win, and a small outlier-ordering **LOSS**
  (−0.00987 on the 17 LIVE rows it fires on, n=17, two of them pinned at the dial
  floor −213.14861); and the windows are fitted on `canonical-2026-05-21` while
  the eval root is post-option-C — one extraction era apart, unfixable today
  because the postC root carries no safesyn leg. Board: `D_shipped@dguard2` +
  `D_guard12_p999@dguard2`. **Stale claim corrected:**
  `benchmarks/rev2_refit_2026-09-06.md` §8.6b says `can_bake` *"admits only
  `logistic`"*; it returns `("logistic", "hgb")` and the **ZCTH tree head IS
  servable**.

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

- **⛔ SHIPPED PROFILE C AND CHdr SERVE ON A DIFFERENT BANDVIS FORMULA THAN THEY
  WERE TRAINED ON (found + measured 2026-09-06, OPEN — needs a user decision).**
  `ComputeSet::from_block_profile` falls back to `everything` for any wide bake it
  cannot narrow, and `everything` hard-sets **`append2_dst_activity: true`**
  (`feature_v2.rs:2163`). The canonical 944 extractor defaults it **false** —
  `zensim/examples/v2_ab_extract.rs:414` reads `ZENSIM_APPEND2_DSTACT` and
  `scripts/canonical_corpus/extract_944_canonical.sh` never sets it — so every
  canonical 944 table C and CHdr were trained and evaluated on has the toggle OFF,
  and the adjudication (`benchmarks/bandvis_dst_activity_2026-08-02.md`, and the
  BANDVIS memory entry) says extraction stays OFF. The flag selects the `BV_DSTACT`
  combine (`feature_v2.rs:3689, 3778`), which changes the VALUES of `f924..f943` —
  slots both bakes read. **MEASURED on one CID22 pair through `Zensim::compute`:
  C 47.743838 (runtime) vs 48.609764 (train-consistent) = 0.866 zensim points;
  CHdr 68.980064 vs 68.669243 = 0.311.** Ruled out by measurement, not argument:
  not the walk width (CHdr's dense walk is also 944 and still moves), not the pools
  (both C bakes read 0 of the 216 `f156..371` lines), not the gather (the eight
  non-append2 bakes gather scattered ids and land bit-identical). **Consequence for
  readers: a Profile C or CHdr number produced by the RUNTIME is not comparable
  with one produced from a canonical 944 table.** The fix is one line
  (`everything`'s `true` → `false`) and it MOVES SHIPPED SCORES, so it is a user
  decision, not a lane's. It also **blocks** densifying C/CHdr and blocks
  collapsing `from_block_profile` into the id-space derivation — both would adopt
  the honest `false` and move those scores. Record:
  `benchmarks/dense_bake_contract_2026-09-06.md` §5.

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

- **✅ RESOLVED 2026-09-06 — THE DENSE-ID GATHER WAS `#[cfg]`-GATED, so `A`, `B`,
  `BHdr` and `D` were SILENTLY MIS-SCORED in every build without
  `feature-regime-v2`.** Kept here because it names a defect CLASS that keeps
  recurring, and because any number produced by a `default-features = false`
  build before 2026-09-06 is wrong. Since `cb2f412d` those four shipped bakes
  are DENSE — each declares the ids it reads and its layer-0 width is that read
  set, not 372 — and serving one requires GATHERING those ids out of the walk's
  identity-laid-out vector. The gather, and `feature_layout` behind it, sat
  behind `feature-regime-v2`. That feature is default-on, **so every number this
  repo has ever recorded is safe**; a consumer that turned defaults off fell
  through to `prep_bake_input_f32`'s POSITIONAL PREFIX branch and got a
  plausible wrong number with no error. **MEASURED** (default vs every default
  feature except `feature-regime-v2`, so tier and threading are held constant;
  6 profiles × 4 geometries × 3 distortions): **all 48 A/B/BHdr/D cells wrong** —
  `B`, which is what `ZensimProfile::codec_target()` returns, read **48.17 →
  13.49** on a blur (max |Δ| 34.68); `D` **48.43 → −213.15** (max |Δ| 261.58, a
  sign flip); `A` 94.11 → 86.44; `BHdr` 95.12 → 81.52; `C`/`CHdr` refused
  outright on all 12 cells. The same instrument's noise floor with the gather
  live in both arms (`avx512`+`threads` against neither) is **1.048e−5**.
  **FIXED by UNGATING**, which turned out to be free: `feature_layout` depends
  only on `feature_set_id` + `feature_defs` + `mlp::Model`, none v2-gated, and
  every shipped dense bake's highest declared id is < 372 — a width the legacy
  v1 buffered walk already emits. Same-parent A/B: `--no-default-features`
  `.rlib` **+134,772 B (+4.29 %)**, rebuild **+0.14 s**; default `.rlib`
  **+3,540 B (+0.05 %)**. (The non-default figure exceeds the module's own size
  because ungating it makes `feature_defs`'s registry reachable where nothing
  referenced it; `.rlib` includes metadata, so it upper-bounds shipped growth.) The two gather
  sites in `metric.rs` lose their `#[cfg]` forks, so there is ONE path and no
  legacy path left to refuse from. `candidate-profiles` now **requires**
  `feature-regime-v2` (only the v2 walk emits the 944 that `C`/`CHdr` declare),
  so **every profile a build can name it can serve** — the census reads zero
  refusals in every configuration.
  **⚠ THE LESSON, because it produced a SECOND instance in the same lane: a gate
  `#[cfg]`-gated on the same feature as the code it protects is not a gate.**
  `every_shipped_profile_is_servable` — the instrument whose entire purpose is
  this question — lived inside the `feature-regime-v2`-gated `feature_plan`, so
  it was blind exactly where it mattered; it now lives in the always-compiled
  `zensim/src/serving.rs`. Same shape, second instance: `zensim/Cargo.toml`'s
  `include` allowlist is only exercised by `cargo package`, which no workspace
  build runs, and MEASURED with `cargo package --list` it was missing **six**
  `include_bytes!` targets (the four `*_byid_2026-09-06.bin` bakes plus
  `c_sdr_purity944` / `c_hdr_l1t1944`) — **a published 0.3.0 would not have
  compiled for anyone**. Both are now checked from a configuration that can
  fail. Gates: `serving::tests::{every_shipped_profile_scores_its_pinned_value,
  dense_bakes_resolve_to_a_dense_layout_and_the_gather_is_not_a_no_op,
  every_shipped_profile_is_servable, every_included_bake_is_packaged}` (the
  pinned-score one runs under EVERY feature permutation CI builds),
  `scripts/serving_matrix.sh` (the cross-build diff, with a vacuity guard so at
  least one arm is genuinely v2-free), and
  `zensim-wasm-tests::large_image_noise_distortion_matches_the_pinned_score`.
  **⚠ THE PINNED-SCORE TOLERANCE IS `0.16`, AND THAT NUMBER IS ITSELF A
  MEASUREMENT — a same-day i686 CI failure corrected it.** Set at `1e-2` from an
  x86-64-only population (max 1.048e-5), it went red on
  `PreviewV0_1/single-LSB` (98.378873 vs 98.394763, delta 1.589e-2) — not a
  mis-serve: `PreviewV0_1` carries no bake, and all four dense profiles passed.
  MEASURED over 160 cells on the four arms CI runs, **there are TWO arithmetic
  classes, not four**: every FUSED-`mul_add` backend (AVX-512, AVX2, NEON)
  agrees to **2.1411e-5**, while magetypes' **scalar and wasm128** backends
  implement `mul_add` unfused, are **BIT-IDENTICAL to each other on 160/160
  cells**, and sit up to **2.8221e-2** from the fused class. Mechanism: on a
  near-identical pair the dissimilarity features are at the f32 cancellation
  floor (193/228 features differ, worst 7.4× RELATIVE at ~1e-6 absolute, raw
  distance 1.42 %) and `100 - 18·d^0.7` has slope -35.5 there. The kernel fix
  was tried and rejected in `e1324192` (shifts `sigma_sq`/`sigma12`, breaks
  `cross_platform::pixel_format_equivalence`; root cause upstream in
  magetypes). `0.16` is the GEOMETRIC MIDPOINT of two populations, each measured on
  the population that bounds it: noise **2.8221e-2** (160 cells; the gate's own
  16 cells top out at 1.5890e-2) and the smallest real mis-serve **on the
  gate's own cells**, **0.945195** — measured directly by bypassing the gather
  in a local build, where all 8 dense-profile cells move, from `B`/single-LSB
  +0.945195 to `D`/quantize+shift −172.261448 (`D` inverts to −155.77). So
  5.67× above the noise and 5.91× below the defect. **Consequences for anyone else: a cross-architecture SCORE bar for
  this metric cannot be tighter than ~3e-2; i686 and wasm32-simd128 are ONE
  class, so validating on either covers both; and a tolerance derivation must
  state the POPULATION it was measured on — a same-architecture one understates
  this by three orders of magnitude.**
  Record: `benchmarks/dense_serving_ungate_2026-09-06.md`; ledger
  `docs/DATASET_HISTORY.md` §3.55 (ROUND 106).

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
  **⚠ THAT FIX STOPPED ONE BRANCH SHORT — the INTERIOR was still uncapped, and
  it stayed that way for two months (found + closed 2026-09-06).** The product
  caps the interior Hermite segment too; the validate copy did not, so
  `bake_verdict` could publish a score `Zensim::compute` reports as exactly 100.
  **The reachable trigger is a KNOT whose `y` exceeds 100** — the wire format
  permits one and `parse_payload` does not reject it — and it is **NOT** Hermite
  overshoot: a draft gate tried to build an overshoot fixture and failed its own
  vacuity guard, because the Fritsch-Carlson derivative rule keeps the
  interpolant inside its bracketing knots (now pinned in-tree). **LATENT, and
  measured: 0 of the 49 bakes on disk** (`zensim/weights`, its `archive/`,
  `zensim-experimental/weights`) **declare such a knot**, so no published verdict
  moved. Closed by construction — `output_calibration_spline::apply` and
  `parse_payload` now delegate to `zensim::score_math::{pchip_eval_capped,
  pchip_derivs}`, so no second implementation exists to stop short again.
  **Recorded, NOT changed** (identical in BOTH implementations, so not an owner
  divergence, and changing it would move product numbers): the lower branch's
  `floor = ys[0] − (ys[n−1] − ys[0])` is a floor only for an INCREASING spline;
  on a decreasing one the `.max` returns exactly **200.0** at `x == xs[0]` —
  seven `zensim-experimental` bakes do. No shipped profile has a decreasing
  spline. Record: `benchmarks/score_owner_consolidation_2026-09-06.md` §3.

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

- **✅ FIXED 2026-09-06 — `train_corruption_head.py`'s BAKE WAS A FUNCTION OF THE BLAS
  THREAD COUNT (found 2026-09-06 at `478bc28e`, fixed same day).** Same recipe, same
  data, same commit, byte-identical source-held-out split, differing only in
  `OMP_NUM_THREADS`: `corruption_head_d228.bin` came out `6f97b653…` (1T) /
  `1229842d…` (4T) / `23ad9c5b…` (8T) / **`da411c8c…` (28T)**. The shipped 2026-09-05
  `d228` head is the 28T one, so re-running the identical command under
  `run-heavy --jobs 8` did NOT reproduce it. Mechanism: the lbfgs solve's BLAS
  reduction order (and, for `--model hgb`, `HistGradientBoostingClassifier`'s
  histogram reduction via libgomp) moved the weights in the last bits and the f16
  pack quantized them differently; the published `metrics.json` moved too (T = 0.9
  detection 0.89527 → 0.89465 at 8T, per-family recall up to 0.4 pt). Same *class* as
  the v1 extractor's `RAYON_NUM_THREADS` dependence below, far smaller amplitude.
  **FIX:** the owner now force-sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
  `MKL_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS`/`NUMEXPR_NUM_THREADS`/`BLIS_NUM_THREADS`
  to `"1"` (unconditionally, not `setdefault` — independence from the caller's own
  exports is the point) before `numpy` is imported, plus `threadpoolctl.
  threadpool_limits(1)` right after import as defense-in-depth for whatever's already
  loaded. **PROVEN, not asserted:** `scripts/v_next/corrhead_determinism_gate.py`
  refits the exact `d228` recipe at ambient 1/4/8/28 threads and gets byte-identical
  `corruption_head_d228.bin` / `..._w944.bin` / `metrics.json` / weights `.json` at
  every count (sha256 `6f97b653…` — which is exactly the historical "1T" value, since
  pinning to 1 thread reproduces the natural single-thread reduction order). A
  synthetic smoke test found no thread-order sensitivity in `HistGradientBoostingClassifier`
  either, before or after the fix, at the scale tested — not shipped either way
  (`can_bake` refuses `--bake-out` for it), but the fix's env-var mechanism covers it
  regardless since OpenMP reads its thread count at first use, not at `dlopen` time.
  **The fix does NOT reproduce the historical 28-thread shipped bake** (`da411c8c…`)
  **byte-for-byte — this is expected, not a defect**: no fixed thread count is
  simultaneously ambient-independent AND equal to a specific historical
  ambient-dependent run. The shipped `corruption_head_d228.bin` was **NOT replaced**.
  Registered delta, scored via `predict_features_with_bake` on the canonical
  `gb82_dog` held-out gate grid (the actual baked bytes, not the training log):
  detection at T=0.9 **83.929 % → 84.077 % (Δ +0.149 pt)**, FP on both `q10`/`q20`
  matched anchors **unchanged at 0.000 %**. **Quote a head bake's sha256 with the
  thread count that produced it** for any pre-fix artifact; every post-fix build is
  thread-count-invariant by construction, so the caveat no longer applies going
  forward. Full record + reproduction: `benchmarks/corruption_head_theories_2026-09-06.md`
  §11 (addendum to §9, which found this, and which still holds the original
  patched-owner ≡ `main` owner ≡ shipped-bake-at-28T parity chain); ledger
  `docs/DATASET_HISTORY.md` §3.48 (Ledger ROUND 100).

## OUTPUT POLARITY HAS ONE OWNER NOW — and four flags stopped being silent no-ops (2026-09-06)

Full record: [`benchmarks/best_of_all_2026-09-06.md`](../../benchmarks/best_of_all_2026-09-06.md) §1;
ledger [`docs/DATASET_HISTORY.md`](../DATASET_HISTORY.md) §3.56.

**The trainer has two readings of what its single scalar output MEANS, and until
2026-09-06 exactly one of eight polarity-sensitive loss sites knew which was in
force.** `Distance` = higher quality → LOWER raw output (the legacy rank-only
convention, what the synthetic gate asserts, what the TV within-ladder hinge
assumes in prose). `Score` = higher quality → HIGHER raw output (what every
absolute/MSE term assumes, because it regresses the output onto `human_score`).
`rank_target_sign` reconciled them at `mod.rs:2556` **and nowhere else**.

**This is why the fastclass2 campaign's best CID22 ORDERING arrived negative.**
Its 2-layer α-head arm read `|−0.8921|`, beating the plain path's `+0.8863`,
with the sign backwards, and `pack` could not spline it (the output calibration
spline is monotone increasing by construction). The 372 recipe declares `:both`
on 4 of 6 legs ⇒ the plain path is SCORE-shaped and the α head stayed
DISTANCE-shaped. Reproduced at raw SROCC **−0.9970** (depth 1) / **−0.9986**
(depth 2) in `tests/output_polarity.rs`; **4 of 5 of those tests FAIL at
`0c6307a7`**.

Ask `mlp_train::OutputPolarity` — never re-derive a sign. `rank_target_sign()`
for RankNet; `ladder_sign()` for BOTH ordering hinges, which now share the one
form `max(0, ladder_sign·(y_better − y_worse) + margin)`. Which member is better
is `quality_sign`, a property of the human scores, never of the convention.

**Four flags now REFUSE where they used to lie.** `--minibatch-size > 1` and
`--norm-in-norm-weight > 0` with an absolute term (the mini-batch helpers are
pure RankNet + PWRC and DISCARDED that term); `--n-hidden-layers >= 2` and
`--skip-connection` off the α-head path (`use_2layer`/`use_skip` are read at two
lines, both inside `train_mlp_per_sample_alpha_head`). A recipe that used either
combination did not train what it said it trained.

**`--leaky-alpha` was a train/serve divergence** — every emitter hard-coded
`Activation::LeakyRelu` while the runtime applies a hard-coded
`zenpredict::LEAKY_RELU_ALPHA`. `hidden_activation_for` now derives the byte and
panics on anything unrepresentable. Incidental: the runtime constant is an
**f32**, so `LEAKY_RELU_ALPHA as f64 = 0.009999999776482582`, not the f64 `0.01`
the trainer defaults to — a ~2.2e-10 gap every LeakyReLU bake has always carried.

**Nothing shipped moved.** `tests/legacy_bake_sha.rs` pins five rank-only
recipes to digests measured at `0c6307a7`; all five reproduce byte-for-byte,
including both TV arms — which is what proves the newly-reachable `--tv-margin`
(previously α-head-only) is the identical function at its `0.0` default. *(Corrected 2026-09-06, `9059b3cb`: those digests are a SAME-CLASS check — trained bakes are not bit-reproducible across SIMD tiers, so CI could never pass a Zen-4 digest on other platforms; `tests/legacy_bake_sha.rs` now asserts the invariant in-process — the polarity owner resolves every legacy recipe to `Distance`, whose multipliers are the IEEE identity, and each recipe retrains byte-identical to itself — while the pinned digests run only under `ZENSIM_ZEN4_GOLDEN_BAKE_SHA` / `just legacy-bake-zen4-golden`, an explicit opt-in, never a silent skip.)*

## `--nonneg-distance` — the dial's C5/C6 made STRUCTURAL, and what it does NOT buy (2026-09-06)

`benchmarks/dial_addressability_gate_2026-09-04.md` §10.3 proves **no monotone
output spline can satisfy both C2 and C6** when real grid cells out-rank a
perfect copy in raw space. It is a weights defect. `--nonneg-distance` fixes it
in the weights:

> `raw(x) = pin − g(x)` with `g ≥ 0` and `g(0⃗) = 0` **bit-exactly** —
> `scaler_mean := 0⃗`, hidden biases frozen at `0`, **ReLU**, output weights
> projected `≤ 0`, output bias frozen at `--nonneg-pin`. `raw(0⃗)` is then the
> **argmax over the entire input space** at f32/f16/i8 alike — **provided every
> active feature transform maps 0 to 0** (see the correction below).

Expressible in the SHIPPED wire format with **zero** runtime change, which is why
softplus / ReLU² / squared-norm were rejected (each needs a new `Activation` or
head in `zenpredict`, plus a change in every serving consumer).

**⛔ CONDITIONAL, and this wave does not meet the condition.** The guarantee
lives in STANDARDIZED space, and the chain is `caller row → feature transforms →
scaler → layer 0`. `raw(identity) = pin` therefore requires **every active
feature transform to map 0 to 0**, and `winsor_p99` with `lo > 0` returns `lo`.
The canonical 372 screen carries **28** such guards. MEASURED: `B_nonneg_s4004`
reads `raw(identity) = 99.6138` against a pin of 100.0. So `raw(x) ≤ pin` ∀x is
still structural (the projection gives it), but **identity is NOT the argmax
here** and C6 = 0 is a MEASUREMENT on 9,593 cells rather than a theorem. C5 still
passes because the identity ANCHOR rows take the same forward, so the spline maps
that same raw to exactly 100. The trainer now WARNS, naming the count and the
worst `t(0)`; making it structural needs zero-preserving transforms or a pin at
`t(0⃗)`, neither of which was done.

**MEASURED on the real 228-slot recipe, same seed, same chain as its control:**
C6 **1,642 → 0** *while* `tied` goes **0.0017 → 0.0000** — the either/or is
DISSOLVED, not traded; C5 **38 outside → 0**; grid max **exactly 100.000** (so
the pin survives training, f16 packing, dead-column pruning and the spline
refit); C4 **−146.04 → −207.42**, so the floor gets deeper. Cost at that seed:
CID22 **0.89036 → 0.88161**, and C1 falls **0.94868 → 0.93040** against a 0.93
bar.

**It does NOT buy A7r** (5 of 5 codecs still fail; the per-codec floors move in
BOTH directions — a redistribution) and it does not buy C3/C4 on its own (those
are the negrich anchor's). Three rows from one mechanism, not six. And **`g` is
CONVEX at one hidden layer**, which is a real restriction: the plain path is
1-hidden-layer only, and `--keep-features` is refused outright with
`--n-hidden-layers >= 2`.

**Two chain facts that cost this lane a wave each:**

- **`bake_dial_refit shared-anchor` asserts a SINGLE-LAYER linear bake.** An MLP
  cannot use it, so a second anchor (e.g. the 21-row identity anchor) has to be
  CONCATENATED into one parquet and passed to `pack --anchor`.
- **A DENSIFIED bake cannot be graded by the pinned probes.** `densify` on a
  contiguous-prefix read set (`f0..f227`) collapses the caller width 372 → 228
  and declares an IDENTITY layout; every registered negtail/identity probe is
  **372-wide**, and `bake_verdict` scores a probe only when its column count
  equals `caller_input_width`. So a densified bake reads **C3/C4/C5/C6 all NOT
  MEASURED** — the entire contract tier. **Score the PACKED bake** (372 caller
  width via `Drop` transforms) and treat `densify` as a servability artifact.
  *(Fixing `densify`'s own identity gate for that shape was a separate defect,
  landed with the scattered-read-set control beside it.)*

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
  corpora, all gates pass, but dial reach falls 96.85 → 85.74. **This proposal
  changed no shipped weight: `ZensimProfile::B` still resolves to the 2026-07-07
  LINEAGE.** *(Corrected 2026-09-06: B's shipped FILE is now
  `b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin` — the same
  2026-07-07 weights rewritten to the dense feature-id contract, bit-identical
  score, see the dense-flip section below. The anchor proposal is still
  unadopted; only the wire shape moved.)*
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
  **⚠ AMENDED 2026-09-06 (owner fix, `benchmarks/select_gaddr_prefilter_2026-09-06.md`,
  registry `select-rule-blind-to-dial-contract-2026-09-06`): PRIMARY + TIE-BREAK were BOTH
  blind to the G-ADDR dial-addressability CONTRACT tier** — measured on the best-of-all wave
  picking a CONTROL arm at G-ADDR contract 4/6 over arms at 6/6, because a floor count and a
  composite have no way to see `dial.addressability` at all. Fixed: a candidate (or, under
  `--seed-group`, any group with one member) that MEASURES a G-ADDR CONTRACT-tier (C1-C6)
  fail is now an ABSOLUTE selectability veto, independent of floor count or composite —
  dial addressability is a HARD ship gate (user rule 2026-09-04), not one more floor a high
  CID22 can outvote. `--floor-basis all` (default) also folds `A7r`'s per-codec
  representability into the floor count itself; `--floor-basis legacy` reproduces the
  pre-2026-09-06 rule byte-for-byte (audit/reproduction only, never for a real selection).
  Failing-first test + before/after re-runs (best-of-all wave + the 125-cell VERIFIED-FAIR
  board) in the linked doc.
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

Full record: [`benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`](../../benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md).

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
feature, and `mean_offset`. Gates: `zensim/tests/fold_engine_parity.rs` (**14
tests**, `to_bits()` equality over the **20-cell** geometry matrix ×
{serial, rayon}; the three pool-sweeping tests widen it to **24 cells** ×
rayon pools 1/2/3/8/16) and `fold_backed_fixtures_match_golden` in
`v1_golden_bytes.rs`
(the fold reproduces the PINNED golden arrays, not merely the buffered path).
Record: [`benchmarks/fold_engine_2026-08-31.md`](../../benchmarks/fold_engine_2026-08-31.md).

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

**Parallelism is the structural asymmetry** — but the numbers this paragraph
carried until 2026-09-05 were a full era stale, in the fold's favour by ~2×.
*(Corrected in place per DOCS: SEARCH + UPDATE, from `fold_mt_scaling_2026-08-31.md`
§3/§5 and `fold_footprint_2026-08-31.md` §7/§9, both of which landed the day
AFTER the paragraph was written.)* Buffered parallelises band-per-strip, degree
`layout_h.div_ceil(STRIP_INNER)` — it grows with image height.

**The fold's degree is NOT 3, and its producer is NOT rayon-free.** Read from
source: `feature_v2_stream.rs:626` runs the two image sides as a `rayon::join`
(SDR only — HDR keeps the serial loop, it shares one `hdr_row` scratch) and
`:769` fans the downscale cascade 6 ways over (side, channel); the conversion
chunks at `CONVERT_CHUNK_ROWS`, which is what `ADVANCE_ROWS` 128→256 exists to
feed. The consuming degree is **3 channels × 4 bands = 12 tasks per strip**,
nested, plus `mean_offset` row bands. MEASURED 1T→8T / 1T→16T:
buffered 4.53×/5.42× (1152²) and 5.14×/6.51× (2304²); **fold 3.42×/3.49×** and
**3.38×/3.71×** — not the 1.1–1.5× this said. Ratio fold ÷ buffered is now
**1.37–1.46× @8T** and **1.61–1.69× @16T** (was 2.30–2.54× / 2.93–3.25×), and
after the footprint lane **the fold is FASTER than buffered serially** —
0.78× at 1152², 0.87× at 2304². **Name a thread budget before trading one path
for the other** still holds; zenmetrics still dodges it by running the fold
`.with_parallel(false)` and parallelising across pairs.

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
paying for itself. ⚠ `v1_golden_bytes` fixtures were 64×64 and
`simd_padded_width(64) == 64`, so the golden set was entirely in the TIGHT
class and structurally blind to the defect C fixes; the rollout added a
non-tight golden (`v1_nontight_fixture_matches_golden`).

**★ C IS FLIPPED — `56bbcda2` (2026-08-30 15:43), "STAGE 1 of the C rollout".**
*(This paragraph said "**Not flipped** — default untouched pending the era
rollout" until 2026-09-05; that was stale by six days and cost a lane an
afternoon. Corrected in place, per DOCS: SEARCH + UPDATE.)* `blur::
pyramid_plane_stride` now returns the width and is the one greppable owner;
`mirror_pad_columns` and its three call sites are DELETED.
`pyramid_stride_has_no_phantom_columns` (in `feature_v2.rs`, NOT `blur.rs` —
it drives the owner from the caller's side) pins it over 24 widths.

**Consequence you must know before citing any 372 number: the shipped runtime
is one extraction era AHEAD of BOTH v1-372 eval roots.** The default root
`/mnt/v/zen/zensim-training/2026-08-30-full-features-372` was built at
`ea16c7ee`, **2026-08-30 13:21 — two hours before the flip**. MEASURED
2026-09-05 by re-extracting CSIQ at HEAD with the same tool on the same input
the root itself was built from (row alignment verified first: `human_score`
bit-identical positionally on all 866 rows): basic `f0..155` **120,804 of
135,096 cells differ, max |Δ| 4.536785**; peaks 34,566/62,352, max |Δ| 0.3264;
masked and IW 62,346/62,352 each. Every row differs, on 285-341 of 372 slots.
CSIQ is 512×512 — exactly the padded-width divergence class C removed. Ruled
out by measurement: `ZENSIM_ERA2_DENSE=0` reproduces HEAD byte-for-byte (so
`515001dc`'s era-2 flip is not the cause — it moves only `f372+`), and
`v1_golden_bytes` passes 5/5 because every fixture is tight or below the tile.
**The 944 roots are NOT affected** — `56bbcda2` verified structurally that the
fold's production path never references the padding owner.

**RUN 2026-09-05 — the runtime-era 372 root EXISTS**, and the era shift on RANK
is far smaller than the 372 era before it.
`/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC` (era token
`v1postc`, `build_commit 4fbd8ff8`, `feature_set_id
basic+peaks+masked+iw@w372/v1postc#d16a1091`; named once, in
`zensim_validate::eval_roots::POSTC_FEATURES_ROOT_372`). Eight corpora
re-extracted at HEAD, six byte-copied and era-stamped per file, `kon504`
derived; `pack_eval372_root.py` now REFUSES a fresh table whose `human_score` is
not bit-identical positionally to the superseded root's (it passed 8/8).
**MEASURED across five shipped 372-class bakes, `v1cur` → `v1postc` moves CID22
SROCC by ≤ 6.8e-4** (shipped D −5e-5, Profile A −5.9e-4, shipped B +2e-5) —
two orders below the `v1pre → v1cur` step's |0.489| worst case, because that
one moved `f228..371` while this one moves `f0..227` smoothly. **The DIAL moves
much more than the rank does**, so this is not licence to read a stored 372
dial number as current. **NOT the default** — which root a flagless
`bake_verdict` reads stays a user decision.

**⛔ The canonical 372 DIAL GRID cannot be re-extracted — its pixels are gone.**
`eval_panels_2026-05-29/qsweep_372_grid.tsv` points every distorted cell at the
`/mnt/v/input/zensim/images/<ref>/<codec>/q<N>.png` decode cache deleted
2026-06-22; **0 of 2,560 paths exist**, and the jpeg leg was `mozjpeg-rs-420-e4`
where the surviving `dial-grid-pixels-2026-07-27/` set is `zenjpeg`. A PRE-C
binary does not reproduce the stored grid from those pixels either (basic max
|Δ| 0.703 vs 0.055 for the whole preC→HEAD era step). The 2026-07-27 pixels are
nevertheless the RIGHT instrument: the registered `peer_ssim2` grid pins were
measured ON them (`dialcells_ssim2_qv2grid.tsv` IS the 944 table restricted to
these 4,424 keys), and `peer_ssim2` reproduces its canonical registry row on all
eight scalars there. Rebuilt grids + probes:
`/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/`, registered append-only
in the G-ADDR floor registry; grade on them with
`ZL_ERA=postC scripts/dialgate_arms.sh score …`. The identity probe is
ERA-INVARIANT (a HEAD rebuild is BYTE-IDENTICAL to the 2026-09-04 one).

Records: `benchmarks/d_ship_flip_2026-09-05.md` §3,
`benchmarks/d_peaks_372_postC_2026-09-05.md` §1.

**⛔ `extract_features_372col` DOES NOT EMIT ROWS IN INPUT ORDER — never
re-attach key columns positionally.** It ends with
`rows.sort_by(|a, b| a.0.cmp(&b.0))` (`extract_features_372col.rs:216`), a
stable sort on `ref_basename`. The row count is right, every key appears once,
and only the pairing is scrambled — it fails SILENTLY. MEASURED cost: the first
build of the 2026-09-05 dial grid shuffled features across its 38 references,
which collapsed every scorer's ladder monotonicity (shipped D **0.9847 →
0.5611**), made jpeg q0 read a dial of 75.6 on an image whose ssim2 truth is
−8.03, and moved `peer_ssim2`'s own `tied` from 0.0000 to 0.0672 — that last is
what caught it. **Do not change the sort**: other callers (the root build above)
compare positionally against stored tables built with it, and both sides being
sorted is why that gate passes bit-exactly. Instead carry a numeric `row_id`
column in the pairs TSV (`load_pairs_tsv` forwards every numeric extra) and
invert the permutation, checked — the pattern
`scripts/canonical_corpus/build_dial372_instruments.py` implements.

**Block-skipping: `V2NewFeatureToggles::v1_only`.** A 372-only request skips
every v2-era block AND its phase-A upstream (the four V-blur sweeps + the v2
activity chain). 249 M Ir vs buffered v1-372's 336 M (**0.743×**) and
944-full's 535 M (**0.466×**). Gate:
`free_extras_are_pure_addition_to_the_v1_only_walk` (bit-identical emitted
slots, skipped range asserted FINITE). *(Called
`folded_v1_only_matches_full_walk` in four source doc comments and several
campaign records until 2026-09-05 — that name has never existed in the tree;
corrected at the four live sites, left in the historical records.)*

**MT: the fold saturated at exactly 3 threads and REGRESSED past it** (2.26×
@3T, then worse) — the channel fan-out is the whole of its parallelism. Band
parallelism inside the channel (4 bands/strip) lifts it: 944-full best 7.75 →
**6.67 ms (2.57× @8T)**, and at 8T the 944-full/944-zeroed marginal
**vanishes** (9.4 vs 9.4 ms paired). Bit-exact ONLY because the merge is
**sequential in band order** — `((0+b0)+b1)+…` reproduces the in-place loop;
a tree/unordered reduction would not (f64 addition is not associative).

**The remaining cap is L3 CAPACITY, not the producer** *(corrected 2026-09-05;
this line read "Remaining cap: the serial `StripPlaneProducer`", which the
fold-MT and fold-footprint lanes superseded the next day)*. The MT lane's
N-independent-process test read a 3.38–3.54× saturation ceiling and concluded
the implementation sat at 94–108 % of the box's own bound; the footprint lane
re-ran it **CCD-pinned** and got **5.85× (CCD0) / 4.54× (CCD1)** from a change
that computes nothing differently — *"reading the fold's own footprint as the
machine's"*. The tell is the signature: before, the fold was CCD-INsensitive
(3.38 vs 3.33) at 52.7 MiB band scratch per process; after, it is CCD-SENSITIVE
(5.85 vs 4.54, 22 % apart) at 11.1 MiB — 8 processes then fit CCD0's 96 MiB and
not CCD1's 32 MiB. A DRAM-bandwidth ceiling cannot produce that pattern; an
L3-capacity one produces exactly it. **Every fold band is FULL WIDTH, so the
per-thread hot set is `2,016·W` bytes** (buffered's `1,512·W`) — linear in width
with no bound, and at 2304²/8T that is 35.4 MiB against CCD1's 32 MiB L3 while
buffered's 26.6 MiB fits. The registered lever is **column-tiling the bands**
(`2,016·W → 2,016·(Tw+20)`, width-independent, 3.9 % redundant H-blur at
`Tw=512`) — which changes the f64 accumulation ORDER inside a band and is
therefore an **era-2-enabled** design, not a bit-exact one. Producer pipelining
was analysed and DECLINED: it needs `unsafe` (the producer appends rows to a
`RollingPlane` while consumers hold `&[f32]` into the same `Vec`) and the
producer's work is inside the N-process bound anyway. ⚠ Note `feature_v2.rs:8425`
still says "4.2x against buffered's 10.9x"; both docs say 3.5× — the source
comment is the outlier.

**dense_block_kernel is the MT ceiling of the 944-FULL EXTRACTION — and it is
NOT on a scoring walk's critical path at all** *(scope corrected 2026-09-05:
`stream_phase_b` gates the whole dense + gradient + append + blockiness block
behind `v2_blocks`, and a fold-backed score asks for `v1_only`, so a scoring
walk never dispatches it — `fold_mt_scaling_2026-08-31.md` §1.2)*. **ERA-LOCKED
— do not restructure it without asking.** It is 23.2% of the 944-full walk and gets
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
⚠ **Both the 23.2 % and that Amdahl bound are `v3`-SCOPED** (callgrind cannot
execute AVX-512, so every Ir profile in this repo is the AVX2 tier with
`POOL_SIMD` — which is `v4x`-only — OFF). Measured by WALL CLOCK on the tier we
actually ship, the kernel is **13.5 % of the v2 block and 7.3 % of the 944 walk
at 2304²**; era-2 §28.5 independently uses 7.9 %. Discard the magnitudes on the
shipping tier; the STRUCTURAL argument (it cannot be split bit-exactly) is
unaffected.

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

**The gates are FEATURE-GATED, and `training` is NOT default.** All three
primary byte gates — `v1_golden_bytes.rs`, `fold_engine_parity.rs`,
`v1_feature_width_pure_function.rs` — open with
`#![cfg(all(feature = "training", ...))]`, so a plain `cargo test -p zensim`
compiles **none** of them and reports a green run that proved nothing. Run
them as `cargo test -p zensim --features
training,classification,custom-profiles,feature-regime-v2,threads`. The
canonical geometry list is `feature_v2.rs`'s
`v1_372_bit_exact_to_fold_at_every_width` (19 cells, duplicated verbatim a few
tests below it); the parity suite's own 20-cell matrix now has ONE owner,
`zensim/tests/common/parity_cells.rs`. The ONLY tolerances anywhere in the
parity suite are `mean_offset < 1e-10` in `compute_with_ref_cross_engine` and
the golden policy `|Δ| <= max(1e-6, 1e-5·scale)`; everything else is
`to_bits()`.

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

Full record: [`benchmarks/profile_d_notax_2026-09-01.md`](../../benchmarks/profile_d_notax_2026-09-01.md).

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

Full record: [`benchmarks/subset_quality_study_2026-09-04.md`](../../benchmarks/subset_quality_study_2026-09-04.md).
Registry: [`benchmarks/good_subsets_registry.json`](../../benchmarks/good_subsets_registry.json).

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

**BUT the SAMPLE seed still moves CID22 more than the INIT seed does** — the two
findings are not in tension. Split-seed pilot (fastclass C0, 5 runs, 120 x 50k,
scored on the board's own root): holding the subset fixed and varying init gives
a CID22-per-ref spread of 0.00118; holding init fixed and varying the subset
gives **0.00313** (2.6x; pooled CID22 4.6x), i.e. ~74-79 % of the wave control's
both-streams spread vs ~16-30 % for init. The order flips on KonJND / AIC-3 /
sdr25, where init leads. n=3 per arm — 3-point ranges, no CIs, and AIC-3's two
components both exceed its control spread, which is the tell that the ordering
is not reliable at this n. **The resolution: the sample seed does not change
WHICH rows are drawn (everything is drawn ~17x) — it changes the ORDER. Coverage
is the wrong summary statistic of a sequence.** So intentional subset design at
these settings must act on ORDER (curriculum, stratified interleaving, hard-pair
scheduling), not on inclusion. `--stratified-bands` is exactly such an
intervention and is the natural first sweep — UNMEASURED, not recommended.

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

## F4's ARM IS `Clamp`, AND F4 NEVER FIRES ON A CORPUS WITH LOCAL PIXELS (2026-09-05)

Full record: [`benchmarks/f4_arm_decision_2026-09-05.md`](../../benchmarks/f4_arm_decision_2026-09-05.md).
Pre-registration: `docs/PLAN_FEATURE_REV2_2026-09-05.md` §7 (pushed `090d55d7`
before any table was extracted).

`ssim_form::SsimLumaForm::REV2_LUMA` is **`Clamp`** — `max(0, 1 - D^2)`, not the
registered prior `SsimLumaC1`. **`SHIPPED_REVISION` is still `Rev1`** and the
`v1ssimcap` entry is still `Proposed`; the arm is decided, the era is not flipped.

**The fact that decides everything downstream: F4's pathology does not occur.**
`clamp` differs from the shipped form ONLY where `(mu1-mu2)^2 > 1`, so it is a
detector — and over **217,756 rows** (cid22val, kadid, tid, konjnd, aic3, csiq,
live, the full 196,086-row safesyn leg, the 9,593-cell ladder, 400 identity
pairs) it moves **0 cells**, with no slot anywhere above `|f| > 2`. The
**5,814,302** that motivates F4 belongs to `bigcodec_hqdedup_traindigits`, which
has **no local pixels**. Consequences you must not re-derive:

* **`clamp` is BIT-IDENTICAL to revision 1** on features, Gram, lasso/BVLS solve,
  output spline and ZNPR bytes (all six R6 bakes sha-for-sha). A rev2 flip
  re-extracts NOTHING whose content resembles those corpora.
* **Do not price the F4 era break as a full re-extraction** until you have
  checked the target population with the clamp detector.
* `c1` (Weber) moves **29.4 M** healthy cells and `lorentz` **24.0 M**, for at
  most `+0.0025` CID22 in one of six variants. Neither is a fix; both are
  redesigns. `Lorentz` is the registered successor if tail ORDER ever becomes
  load-bearing — never `c1`.

**Three corrections this lane measured, all live for readers:**

1. **F4's blast radius keys on POOL STATE (`feature_set_id`), not width.**
   `ext944`/`ext924` have `f156..371` all-zero (36 moved slots); the 2026-09-05
   pools-live ladder grid (`foldapp2pools`) is **98.7 %** nonzero there (132).
2. **The "winsor already clamps it" mitigation covers Profile B ONLY.**
   `ssim_moment_explosion_2026-07-16.md` §7b's "the pathological rows never reach
   the linear head" is true of B (372 x `winsor_p99`). **Profile D — the SDR
   default — carries NO `feature_transforms` and NO `feature_bounds`**, and
   neither does `v47_strict_qat_native` or the 944 Profile-C family.
3. **⛔ The unbounded feature that actually fires is `contrast_inc`, not F4.**
   Max rev1 `|f|` over all 372 slots: safesyn **36,465.7**, LIVE **3,598.2**, TID
   927.9, KADID 618.3 — every one a `contrast_inc` slot (local 12 of each basic
   group) = `hf_energy_gain = max(0, hf_dst_L2/hf_src_L2 - 1)`, unbounded above by
   exactly F4's flat-source mechanism, while its siblings `var_loss` and
   `tex_loss` are bounded at 1.0 and `mse` at 0.169. **122 of 779 LIVE rows
   (15.7 %) exceed 100**; p99.9 is 1,088 there. It has **no registered defect, no
   arm and no gate**. Registering it is the obvious next lane.

**Per-shipped-bake F4 exposure** (read set ∩ the 132 measured F4 slots), so a
re-verdict is priced not guessed: `v47_strict_qat_native` **125/285**, `bhdr_*`
20/50 and 49/133, `d_sdr_add156_*` (the SDR default) **4/28**, `b_sdr_linear_*`
12/95, the 944 Profile-C family 36/667-697.

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

Wave skeleton + the priced anti-pattern table: [`docs/WAVE_PLAYBOOK.md`](../WAVE_PLAYBOOK.md).

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
- **The tile's LAST slab can be `radius + 1` columns wide, and that was a
  latent debug-build panic (found + fixed 2026-09-04).** The last tile stages
  `width % tile` interior columns plus a `radius` left halo, so
  `tw = (width % tile) + radius`; at `width % H_TILE_WIDTH == 1` that is exactly
  `radius + 1` and the H kernels' one-step right-edge mirror
  `2*(tw - 1) - (x + radius + 1)` underflowed `usize` — **12 tests**, including
  `Zensim::compute` through `fold_engine_parity`'s `(2049, 40)` cell.
  **Release bytes were never wrong** (that index feeds only the final `x`'s
  running-sum update, which nothing reads; pre/post-fix release `to_bits` dumps
  identical at 1025/2049/3073 and at every control). The rule now has ONE owner,
  `blur::h_mirror_add_idx`, matching the V kernels' long-standing
  `saturating_sub`; gate:
  `blur::tests::h_entries_are_bit_exact_at_a_degenerate_last_column_tile`.
  **When you change `H_TILE_WIDTH` or add an H entry, `width % tile == 1` is the
  cell to test** — a width-sweep that steps by anything but 1 will miss it.

- **Carry an identical-code-path control.** The tile does nothing when
  `width <= tile`, so 576²/1152² cells run the *same code* in both arms and
  must read 1.000×. What they actually read is the measurement's noise floor:
  **±0.3 % at 1T, ±1.8 % at 16T, up to 6.5 % at 8T.** Any threaded cell inside
  that band is unestablished, not a result — a 0.891 × "regression" published
  earlier in the same session evaporated under it.

## ⇒ POST-COMPACT / NEW SESSION: read [`SESSION-RESUME.md`](../../SESSION-RESUME.md) FIRST

Then return here. `SESSION-RESUME.md` is the canonical entry point —
it points at every other doc + lists the current critical-path
tasks. Reading order on resume:

1. [`SESSION-RESUME.md`](../../SESSION-RESUME.md) — current state, ~1 min
2. [`docs/TOP_MODELS_COOKBOOK.md`](../TOP_MODELS_COOKBOOK.md) — **the validated
   science + exact reproduction of the top models + the new-model loop + the
   pitfall list.** THE entry point for any model work (2026-07-18).
3. [`docs/MODEL_SELECTION_SCORECARD.md`](../MODEL_SELECTION_SCORECARD.md) —
   the five-gate exam (RANK/DIAL/STEER/RD/TARGET) every ship candidate takes.
4. This doc (`CLAUDE.md`) — methodology + workflow + gotchas. **NOTE: the V0_x /
   PreviewV0_5-era historical sections were excised 2026-07-19 to
   [`docs/HISTORY-2026-05-v0x-era.md`](../HISTORY-2026-05-v0x-era.md); the
   cookbook supersedes them for current state.**
5. [`docs/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) — **the traceability
   spine**: how a number chains back to bytes (verdict → bake sha → manifest →
   input shas → trainer commit), which gate enforces each link, and the honest
   list of gaps. Read before making or citing any measurement.
6. [`RESEARCH.md`](../../RESEARCH.md) — corpus map + workflow recipes
7. [`benchmarks/INDEX.md`](../../benchmarks/INDEX.md) — find prior experiments
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
- **Monotone in perceived codec quality** (so target-hitting converges), with
  **identity at 100 and a working negative tail**. This is the product contract,
  not a claim that every current profile passes it on every input.
- **NEGATIVE zensim values MUST work** — inputs worse than the worst codec output
  score BELOW 0 (do NOT clamp at 0; the lower spline extrapolation + profile
  `extrapolate_score` carry it). Negative-tail training data:
  `canonical-2026-07-15/train/kadis_negrich.parquet` (negative-rich).
- **The diffmap MUST serve the scalar** — its per-block "where to adjust"
  signal must improve the score the loop targets. The July legacy-map mismatch
  is historical; model-sensitive maps and later loop experiments exist. Grade
  the actual scorer/map pair with the current coherence and codec-loop
  instruments; see `docs/MODEL_SELECTION_SCORECARD.md` and
  `docs/STEERING_PAIR.md` (including its August 29 adoption update).

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
  Plus (2026-09-06) **THE OPERATIVE DIAL RULER**: the board's dial-addressability
  column, the **NOT-SHIPPABLE badge**, the `floors ok` (A7r) column and the G-ADDR
  panel read a cell's **`dial_ladder`** block (the 2026-09-05 FLOOR-DENSE instrument —
  five codec families incl. both AVIF backends, 66 floor-dense steps) when it has one,
  else its canonical `dial.addressability`. **ONE owner picks** (the operative-block
  pick in `load_fulleval`, mirrored by `freeze_check::gaddr_block`) so no two consumers
  can disagree about which instrument they describe. A ladder-graded cell carries the
  canonical read beside it as **`dial_canonical`** — **hover context only**: never
  sorted on, never counted, never mixed into a column, and every tooltip NAMES its
  ruler. **They are different INSTRUMENTS — read a pair DOWN, never across.** 450 of
  508 cells carry a ladder block; the other 58 are NOT MEASURED with a recorded reason
  (never a zero). Record: `benchmarks/board_ladder_ruler_2026-09-06.md`.
  ⚠ **The fair board's size margin is now 53,984 B under the 12 MiB cap** (12,528,928 B
  at 171 cells) because 279 cells gained a first-ever G-ADDR block — re-measure before
  adding another per-cell block.
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
  Plus (2026-09-05) **URL COMPARE SETS**: `#compare=<id1>,<id2>,...` in the fragment
  (a bare `#<id1>,<id2>` list too, when the fragment carries no `key=`) pins the board to
  exactly those models. Ids are board names AS RENDERED — matched exactly and
  case-sensitively, with a case-insensitive fallback that the banner always reports. The
  list sets `state.visible`, the ONE selection owner, so every list, table and chart
  follows with no parallel filter, and it OVERRIDES every default: curated, family
  toggles, the dominated default-off, the gate pre-filter and the forced `peer_ssim2`
  reference row (an explicit list means explicit). The scoreboard RESTRICTS to the listed
  rows rather than dimming them and defaults to the FRAGMENT order (sortKey `cmp`); a
  header click sorts normally after that. A missing id raises a full-width `role=alert`
  banner (`#cmpbanner`, above the sticky bar, theme-aware in both modes, not dismissible)
  giving requested/found counts, every missing id verbatim, and up to 3 nearest board
  names per miss as one-click REPLACEMENTS (bounded Levenshtein + prefix/substring,
  client-side — no library, no request); if NONE resolve, the banner shows and the normal
  default view is rendered rather than an empty board; a fully exact list renders NO
  banner (the compare strip in the bar is that status surface). Plus (2026-09-05): a
  suggestion click REPLACES the missing id and rewrites the hash even when the
  replacement collapses into an id already present (the `syncHash()` found-array-equality
  bug this closed), a "drop missing ids" control clears every not-found id at once so the
  banner keeps shrinking toward empty rather than requiring one click per id, and a bare
  prefix resolves automatically when it names EXACTLY one board row — reported in a small
  "prefix expanded" note under the pickers, never the banner — while an ambiguous prefix
  (>=2 rows) is never guessed at and stays a plain miss with suggestions; gated by
  `gauntlet_gates.sh` cases (d) and (e). Two-way: a selection edit rewrites
  `location.hash` via `history.replaceState` — no scroll jump, no `hashchange` — and ONLY
  when the set actually changed, so a shared link keeps a typo's evidence across a reload;
  `hashchange` re-applies an in-place edit; *copy link to this comparison* sits by the
  pickers with a try/catch-guarded clipboard call that ALWAYS also shows the URL as text.
  Gate 4 in `gauntlet_gates.sh` runs three cases (two known ids → exactly those rows in
  fragment order, banner absent / known + a proven-absent typo → banner names it verbatim
  with real-board-name suggestions / `#compare=` empty → default view, no banner), each
  also click-testing the copy control; the ids are READ OUT OF THE BOARD so the gate
  cannot go stale, the location/history surface lives in the harness SHIM, and `CMPMODE`
  is read from the app's own `state.cmp` through the vm context so the harness never
  becomes a second owner of the matching rule. Examples + the two shareable URLs:
  `benchmarks/fair_gauntlet_2026-09-04.md` "URL compare sets".
The first two modes' plots: per-bake scatter+trend, grouped 10-band SROCC bars,
calibration curve, residual, candlestick, SROCC heatmap, 2-panel Pareto trade
(CID22 vs nonphoto / KonJND), composite ranking bar, per-codec dial plots +
dial-mono %, full Mohammadi stat panel (incl. low-tail/high-tail SROCC), 10-band
table, honesty/provenance panels. Run from `scripts/v_next/` (imports `blend_lib`/
`gauntlet` from cwd). `bake_report.py` adds the 2×4 8-corpus scatter grid with 4PL
fit + PWRC (reports/); `bake_verdict --html` is the single-bake Rust report.

> **Historical (May-2026 V0_x / PreviewV0_5 era):** the training goals + three-trail SOTA + shipping/experiment-rigor policies, 2026-05-1x eval mandates, V_20/V39 learnings, canonical-2026-05-18 corpus archaeology, the interactive-site spec (since shipped: `site/compare.html`), V_X experiment workflow, V0_1-era weight status, and V0_7 e1 fill were moved verbatim to [`docs/HISTORY-2026-05-v0x-era.md`](../HISTORY-2026-05-v0x-era.md) on 2026-07-19. Current guidance: [`docs/TOP_MODELS_COOKBOOK.md`](../TOP_MODELS_COOKBOOK.md).

## G-ADDR — the dial ADDRESSABILITY gate, and what it measured on the SHIPPED dial (2026-09-04)

USER RULE (2026-09-04): **"floor and ceiling dial addressability is crucial … any model that
limits dial range cannot ship."** `bake_verdict`'s DIAL panel now emits a **G-ADDR** section
on every run and carries `dial.addressability` (+ `dial.min` / `dial.max`) in `--full-json`.
Owner: `zensim-validate/src/dial_addressability.rs`; append-only registry
`benchmarks/dial_addressability_floor_2026-09-04.json`. Full record:
[`benchmarks/dial_addressability_gate_2026-09-04.md`](../../benchmarks/dial_addressability_gate_2026-09-04.md).

**Two tiers, never merged.** REGRESSION (`A1`–`A9`) bars against a REFERENCE scorer's own
value on the SAME instrument; CONTRACT (`C1`–`C6`) are absolute product bars. Separate
because **the shipped dial fails four contract rows**. Absent probe = NOT MEASURED, never a
pass; an unregistered dial grid = NOT MEASURABLE. Run it with `--negtail-probe` +
`--identity-probe` (both pinned under `/mnt/v/output/zensim/dialgate-2026-09-04/`) or those
axes read `—`.

**⛔ SINCE 2026-09-05 `C1` COUNTS ONLY *DIAL*-ATTRIBUTED INVERSIONS — a
backwards rung BOTH reference metrics confirm is the ENCODER's.** USER
DIRECTIVE, verbatim: *"for inversions, we should choose say ssim2 and butter and
only flag true inversions where they agree, and we can then file or update
tracking issues on codecs for when they are nonmonotonic."* A dial ladder walks a
codec's own settings, so a backwards rung has two causes and the pre-ruling
instrument could not tell them apart. THE rule has one owner —
`dial_addressability::encoder_inversion` — shared by C1's `mono` input and
`bake_verdict`'s zone census: **`Δssim2 ≤ −0.5` pt AND `Δbutteraugli-pnorm3 ≥
+0.05` distance**, both required. `--inversion-truth single|agree` selects the
reading (`agree` is the DEFAULT); `--reference-truth <tsv>[:variant]` supplies
the per-cell table; `--encoder-inversion-census <tsv>` writes the
BAKE-INDEPENDENT set (the codec bug-report evidence).
**Facts you must not re-derive:**
(a) **The butteraugli margin is NOT a noise margin — there is no noise.** A
from-scratch re-run of the instrument's jpeg leg reproduces 2,574/2,574 cells at
`max |Δ| = 0` on BOTH butteraugli variants. The margin is the **p85 of |Δ| on
FORWARD pairs whose Δssim2 ∈ [0.45,0.55], rounded UP to the next 0.05** — pnorm3
0.0481 → **0.05**, max 0.2189 → **0.25**. Rounding UP is conservative: a larger
margin excuses FEWER rungs.
(b) **pnorm3 is PRIMARY on measurement, not taste** — 94.30 % direction agreement
with ssim2 over 9,411 pairs vs `max`'s 75.27 %; and `peer_butteraugli_max` fails
C1 (0.9286) under both readings.
(c) **BOTH, not EITHER, is load-bearing**: of 105 ssim2-alone material
inversions, butteraugli moves the worse direction on only 47. On D's ten worst,
**ssim2 alone confirmed 9/10; both references confirm 5/10** — and on two of
those, ssim2 reads a 7-10 pt loss while butteraugli says the higher setting is
BETTER.
(d) **UNKNOWN IS NEVER AN EXEMPTION** — a pair with no reference row stays
charged to the DIAL and is counted in `dial.inversion_truth.n_attribution_unknown`;
a requested `agree` with no usable table degrades to `single` LOUDLY.
(e) **The board cannot move and did not**: `mono_agree ≥ mono_single` always,
`mono` gates exactly ONE row (C1, a `≥` bar), and all 130 board G-ADDR cells
already read C1 PASS ⇒ badge count unchanged at **47**. The two-reference reading
is **NOT MEASURABLE** on the canonical-372 / 944-POOLS grids: their only
butteraugli is the `max` variant (identified at median rel err 0.0029 over 4,105
cells) and re-scoring pnorm3 would be a decoder-era confound.
(f) **`--inversion-truth single` is byte-identical to the pre-ruling binary**
(0 JSON differences on shipped D) — every published pre-2026-09-05 inversion
number stays reproducible, and is scoped in `eval_annotations.json` as
`inversion-counts-single-reference-pre-2026-09-05`. Never compare a `single`
count with an `agree` one.
Measured C1 inputs (ladder instrument, single → agree): **D 0.99310 →
0.99470** (15 rungs), D-prev 0.99420 → 0.99540 (12), A 0.98030 → 0.98120 (8),
B 0.97760 → 0.97870 (11), **`peer_ssim2` 0.98880 → 0.99160** (26) — so D's
dial-attributed inversion rate is **0.53 % against the mentor's 0.84 %**.
Codec issues filed from the census: **`imazen/zenjpeg#201`** (5 pairs, all
costing bytes), **`imazen/zenrav1e#42`** (20 pairs / 14 refs, 13 costing bytes),
**`imazen/zenav1-svt#19`** (1 pair + the 36.4 % setting-saturation observation);
**jxl and webp are CLEAN — zero confirmed.** Record:
[`benchmarks/inversion_truth_2026-09-05.md`](../../benchmarks/inversion_truth_2026-09-05.md),
gate doc **§18**.

**⛔ SINCE 2026-09-05 THE OPERATIVE FLOOR WINDOW IS `resolvable`, AND `A1`-`A6`
ARE REPORT-ONLY.** USER RULING (the "ok" opening *"ok, is there poor resolution
compared to ssim2? … what should be the new SDR and HDR [defaults]"*, read as
accepting both recommendations put to the user — the reading is recorded in the
gate doc **§17.1** and in the registry, because it is an inference, not a
verbatim instruction). Two changes, both reversible with no code change:
**(1)** `A7r` grades only the lowest settings the MENTOR itself resolves by
≥ 0.5 points (`--floor-rule resolvable --floor-margin 0.5`), not the three
literal lowest positions — the old window graded steps ssim2 cannot separate,
and on jpeg it graded ELEVEN encoder-identical settings as three. **(2)** the six
dial-VALUE rows `A1`-`A6` moved from hard regression bars to `Tier::Report`:
still measured, still printed with their bars, gating nothing, so the REGRESSION
headline is carried by `A7r` alone. The CONTRACT tier is untouched, so **the
board's NOT SHIPPABLE badge cannot move with either change** (asserted by a unit
test AND by a measured board count, 63 → 63). The OPERATIVE window is a REGISTRY
property (`negative_tail_bars.active` → `floor-representability-resolvable-2026-09-05`,
read by `dial_addressability::operative_floor_rule`), so reverting is a
one-line registry edit; per-invocation levers are `--floor-rule distinct` and
`--gaddr-value-pins hard`, **proven to reproduce `main@origin`'s grading
row-for-row on 4 cells against a pristine binary**. A mentor-windowed rule with
no `--gaddr-grid-truth` reads `A7r` **NOT MEASURED** — never a silent fall-back
to `distinct`'s window. Record: `benchmarks/dial_addressability_gate_2026-09-04.md`
§17; proposals: `benchmarks/default_proposals_2026-09-05.md`.

**THE 2026-09-05 DEFAULT PROPOSALS (nothing installed; the user decides).**
**SDR — keep Profile D.** On the FLOOR-DENSE 372 ladder under the operative rule
it is the ONLY scorer passing both tiers (`SHIPPABLE`): 5/5 codec floors,
CONTRACT 6/6, **exceeding** the mentor on `avif-rav1e` (0.6667 vs 0.6410) and
`jxl` (1.0000 vs 0.9615). Under the retired `distinct` window it FAILED jpeg by
one ladder — the ruling is what lets the installed default pass the axis it is
in fact best at. **All 97 re-graded board cells fail `A7r` on that instrument;
zero alternative candidates exist.** Profile B still leads RANK (CID22 0.8821 vs
D's 0.8634) and fails the dial on every codec — the dial and the ranker are not
the same decision. Shipped D is on the board only as `d_id100_negrich@did100lane`
(bake sha verified identical). **HDR — keep `BHdr`**: UPIQ pooled **0.7536**,
above ssim2-PU (0.7044), above PU-SSIM (0.7395), and above every one of the 24
HDR944 cells and every arm mean (best single seed 0.7254, `CHdr` 0.6664). But
**G-ADDR on HDR is NOT MEASURED and no instrument can measure it** — an HDR
ladder is the registered follow-up — and `bhdr_…@cur372`'s G-ADDR verdict is an
SDR-instrument grading of an HDR-only profile and must never be quoted as its HDR
dial. BHdr's own promotion was **maxT p = 0.221, not significant**, and it
overlaps its own HDR census instrument.

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
  `reach` **153.731188** · `DR` **85.1959783** · identity **100.000000** on all 38 refs ·
  **0** of 4,424 cells above identity. *(The negative-tail bars `min` −770.619744 /
  `p1` −187.13142579 / `frac<0` 1.0000 are **RETIRED 2026-09-05** — see the tail re-pin
  below. They remain the grading of every G-ADDR number published before that date.)*

**⛔ THE NEGATIVE TAIL IS NOW *FLOOR REPRESENTABILITY*, PER CODEC — A7/A8/A9 RETIRED, and
NO DIAL VALUE IS A BAR (USER RULING 2026-09-05, operative form).** Verbatim: *"i care that
the lowest configurable settings per codec are representable, not that negative fifty is in
that specifically."* (Two earlier forms — *"the negative tail bar is entirely arbitrary.
below -5-50"*, then *"i said -50 not -5, codecs are all different, some go lower than
others"* — are recorded in the registry as the path to it. **Neither −5 nor −50 is a bar
anywhere in the active tier**; a test fails if any retired bar key reappears.)

  | row | tier | axis | bar |
  |---|---|---|---|
  | `A7r` | regression | **per codec** on the dial grid: fraction of `(image_id, codec)` ladders whose **K=3** lowest configurable settings are REPRESENTED | the **mentor's own fraction** on the same cells, registry-pinned |
  | `A8r` | **report-only** | the negtail probe: pooled `min`/`p1` + its truth extremes | **none** |
  | ~~`A9r`~~ | — | **dropped**; its per-codec quantity folds into the report block | — |

  **REPRESENTED** = (i) the dial strictly increases across the bottom K steps *and* into the
  next step up (no ties, no inversions), and (ii) no bottom step sits within 1e-9 of the
  dial's instrument-wide minimum, unless that ladder is its **sole** holder (two or more
  sharing it = a collapsed floor). `q` is quality-oriented on every codec — JXL's `q=0` is
  its **largest** distance — so "lowest settings" is always the smallest `q`.
  **The bars, MEASURED** (identical on the canonical/preC/postC grids): avif **1.0000**,
  jxl **0.9697**, webp **1.0000**, jpeg **0.0000**. **jpeg's 0.0 is the ENCODER saturating,
  not a scorer defect** — on 22/22 ladders its three lowest settings are byte-identical
  output (max |Δ| over 372 features exactly 0.0; ssim2 itself returns −8.045 at all three).
  That is what makes exemption a *measurement*: a codec the mentor cannot represent bars
  nothing. `A8r` is report-only because the probes carry **no codec identity at all** (KADIS
  distortion types), and grading that instrument per distortion family at a fixed depth was
  measured to fail every bake ever built on one n=8 family.
  **THE INSTALL ANSWER (16 scorers, postC instruments): shipped Profile D is the ONLY bake
  that is both `A7r`-PASS on every codec and CONTRACT-PASS** — it beats the mentor on jxl
  (1.0000 vs 0.9697). All 11 λ-sweep arms and D-peaks FAIL on **jxl** (0.818-0.909); shipped
  B fails avif + jxl; Profile A fails avif. Under the RETIRED pins shipped D read A7=FAIL,
  i.e. the old bars faulted the dial that is in fact best at the floor. **Reproduce any
  pre-ruling grading with `--gaddr-tail-pins retired`** (row-for-row on 88 of 97 board cells;
  the other 9 moved for an unrelated registry append). The board's NOT SHIPPABLE badge is
  contract-driven and **unchanged (47 measured / 46 on board, 0 contract rows changed),
  asserted not assumed**. Record: gate doc **§16**.

- *(The three bullets below describe the 2026-09-04 ssim2 re-pin of `A1`-`A6`, which the
  2026-09-05 floor re-spec did NOT touch.)* **It is NOT a relaxation — it moved the
  difficulty to the FLOOR.** Re-grading all 17
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
  user option. The 2026-09-05 floor re-spec sidesteps it for the FLOOR: `A7r` asks whether
  a codec's lowest settings still RESOLVE, which is a calibration-free question, and its bar
  is the reference metric's own answer on the same cells.
- **Flags that make all of this measurable:** `--negtail-peer-scores` /
  `--identity-peer-scores` (a reference metric has no bake, so its floor axes were previously
  unmeasurable; peer mode is now **all-or-nothing per axis** and never fills a probe from
  `--bake`), `--gaddr-json` (the G-ADDR block alone at full f64, stamped with which scorer it
  describes — peer-safe where `--full-json`/`--fulleval` are refused), `--gaddr-reference`.
  Re-grade an existing bake with `scripts/dialgate_arms.sh score <label> <bake.bin> [regime]`.
- **⭐ THE BOARD'S OPERATIVE DIAL RULER IS THE LADDER INSTRUMENT (2026-09-06, gate doc §19).**
  Record: [`benchmarks/board_ladder_ruler_2026-09-06.md`](../../benchmarks/board_ladder_ruler_2026-09-06.md);
  plan: [`docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md`](../PLAN_BOARD_LADDER_RULER_2026-09-06.md).
  **450 of 508** board cells now carry a **`dial_ladder`** block — the read on the
  2026-09-05 floor-dense instrument of their own width — and the board's
  `dial.addressability` column, the **NOT-SHIPPABLE badge**, the `floors ok` (A7r)
  column and `freeze_check --select`'s CONTRACT veto ALL read it, through ONE owner
  (`gauntlet.py`'s operative-block pick; `freeze_check::gaddr_block`). The canonical
  read is **retained byte-for-byte** as `dial.addressability` and surfaced on hover as
  **`dial_canonical`** — **0 of 508 `dial` blocks changed**. **They are DIFFERENT
  INSTRUMENTS: read a cell's two blocks DOWN, never across.** The **58** cells with no
  ladder read are NOT MEASURED with a recorded reason (34 era mismatch — the bake reads
  `f156..371`, which its own 944 instrument zeroes at slot-set `026c0aba`/689 and the
  ladder-944 populates at `b6811ae0`/905, so it would be scored through weights that
  never took a gradient; 21 at a width the ladder was never built at; 2 peer metrics
  with no per-cell table; 1 bake not on disk). Tooling:
  `scripts/gaddr_board_ladder.py {grade|graft}` + `promote_fulleval.py
  --graft-gaddr-ladder` (refuses any grid that is not a REGISTERED ladder instrument;
  `--graft-gaddr`'s same-grid gate is NOT relaxed) + `freeze_check --gaddr-block
  auto|canonical|ladder` (`canonical` is byte-identical to the pre-2026-09-06 binary,
  proven against a pristine build).
  **What moved:** the badge count 63 → 75, but **21 of the +22 are FIRST-EVER coverage**
  on cells that had no dial ruler at all — the true ruler effect is **ONE up-flip
  (`A2b_l0.002`, C1) against TEN down-flips**. The switch is **directional per row, not
  uniformly stricter**: **C1 gets HARDER** (the ladder samples q 0..30 at step 1, so far
  more near-flat adjacent rungs) and **C2 gets EASIER** (floor-dense sampling puts fewer
  cells near the ceiling clamp — `W10L9PH_s4007_packed` 0.0908 → 0.0306 against a 0.05
  bar). `freeze_check --select`'s pick does **NOT** move (`11e243eb0b86` =
  `fc2_372_S228_H128_s4004/5/6` under all three rulers), but **10 fair cells go
  `NO — CONTRACT FAIL` → `yes`**: the veto's removal-only property holds *within* a
  ruler, not *across* one.
  **⚠ THE BOARD'S A7r FRACTIONS ARE `resolvable`, NOT the `distinct` ones the ladder
  record's §9 tables carry** (operative bars `avif-rav1e` **0.6410** / `avif-svt` 1.0000 /
  `jpeg` **0.6667** / `jxl` 0.9615 / `webp` 1.0000). Under `distinct` shipped D fails
  jpeg; **under the operative rule the ADD156/D lineage passes all five codecs** (0.6667 /
  1.0000 / 0.6667 / 1.0000 / 1.0000) and is the only bake family on the board that does —
  `peer_ssim2` is the only other five-codec pass, and `peer_butteraugli` fails four of
  five. Board-wide only **10 of 450** cells clear all five and **293 clear none**; every
  best-of-all / fast-class arm collapses on `avif-rav1e` (0.13–0.21 against 0.641).
- **BOARD COVERAGE (2026-09-04, gate doc §15) — SUPERSEDED as the operative ruler by the
  bullet above; still the record of the CANONICAL-instrument pass:** all **97** fair-board cells are graded under
  both pin sets and 96 carry the verdict; the board draws a **NOT SHIPPABLE** badge off a
  MEASURED contract FAIL (**47** cells; C3+C4 dominate at 39 each — dials that never go below
  zero on an all-negative probe — then C2 at 23). The **only** contract-clean cell is
  `v47_strict_QAT_native` (6/6). A probe is scored only when its column count equals the
  bake's caller width, so the 372 probes reach 3 cells; two **944** negative-tail probes were
  cut in-era by `scripts/cut_gaddr_negtail_probe.py` (the committed owner of that cut; its
  control reproduces the stored 372 probe EXACTLY) and are deliberately **NOT registered** —
  no reference has been measured on them, so A7–A9 stay NOT MEASURED there while the
  absolute-bar C3/C4 read normally. Graft with `promote_fulleval.py --graft-gaddr`, or
  re-grade the whole board with `scripts/gaddr_board_regrade.sh {grade|graft}` (committed
  2026-09-05; it reconstructs every cell's invocation from the 2026-09-04 as-run logs and
  refuses to graft if the contract-fail count moves).

**★ THE FLOOR-DENSE LADDER INSTRUMENT (2026-09-05) — and it FAILS the shipped dial.**
Record: [`benchmarks/ladder_instrument_2026-09-05.md`](../../benchmarks/ladder_instrument_2026-09-05.md);
plan: [`docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md`](../PLAN_LADDER_INSTRUMENT_2026-09-05.md).
Grade on it with `ZL_ERA=ladder scripts/dialgate_arms.sh score <label> <bake> 372`.

- **Every grid before this one asked the A7r question of jpeg and could not answer
  it.** MEASURED: **zenjpeg emits ONE bitstream for every q in 0..10** (identical
  bytes AND identical ssim2, on every reference), so the old grids' bottom three
  jpeg steps — q 0/5/10 — are one setting sampled three times, and the mentor's jpeg
  bar was a vacuous **0.0000** that anything passes. It is visible in the incumbent's
  own grading as jpeg `bottom_medians` **22.22 / 22.22 / 22.22**.
- **Per-codec floors, measured** (two settings are the same setting when encoded
  bytes AND ssim2 agree): `zenjpeg` plateaus q 0..10, first distinct **q=11**;
  `zenwebp` q=0 only; `zenavif/zenravif` q 0..1; `zenavif/svt-rs` q 0..1 then
  **pairwise ties** (quality 0..100 onto QP 0..63); `zenjxl` **distance >= 25
  saturates** — 26/30/40/50 are byte-identical to 25.0, so 25.0 IS the floor and
  "extending" it buys nothing.
- **The rule is dedup by encode hash, never a per-codec step table.** `avif-svt` is
  **36.4 %** duplicate settings against `avif-rav1e`'s **3.0 %** on the same axis.
  The instrument keeps DISTINCT settings only (which is what lets
  `dial_addressability.rs` stay unchanged — its "bottom K" are then the bottom K
  *configurable* settings); the full archive keeps every step with a `saturated`
  flag.
- **The two AVIF backends are two ladders** (`avif-svt`, `avif-rav1e`). Verified
  against source: `FloorMeasure::from_grid` groups by `(image_id, codec)` from
  whatever strings the grid carries, so this needed **zero** production code change.
  The backend is a KNOB (`--codec zenavif --knob-grid '{"backend":["svt-rs"]}'`,
  feature `avif-svt`) — there is no `zenavif-svt` codec string.
- **⛔ SHIPPED PROFILE D FAILS A7r ON THIS INSTRUMENT UNDER THE `distinct` RULE — on
  `jpeg`, by one ladder** (0.5128 vs the mentor's 0.5385; 20/39 vs 21/39), plus A1
  (99.99996372 vs ssim2's exact 100.0) and A3 (p95 93.884 vs 93.974). It passes A7r on
  every older grid. **⚠ CORRECTED 2026-09-06 (gate doc §19): that is the RETIRED
  `distinct` window. Under the OPERATIVE `resolvable` rule (margin 0.5, bars
  `avif-rav1e` 0.6410 / `avif-svt` 1.0000 / `jpeg` 0.6667 / `jxl` 0.9615 / `webp`
  1.0000) the ADD156/D lineage PASSES all five codecs** (0.6667 / 1.0000 / 0.6667 /
  1.0000 / 1.0000) — the two rules are different quantities on the same cells and the
  difference flips the verdict, so always read `floor_rule` beside a fraction.
  **Do not read a pre-2026-09-05 A7r pass as evidence the dial resolves jpeg's
  lowest settings — it never measured that.**
- **MEASURED, and it closes off a whole class of fix: re-anchoring cannot repair
  it.** All **19** of D's failing jpeg ladders are inversions in the RAW
  (pre-spline) model, and raw-vs-dial ordering verdicts agree on **39/39**. A
  monotone spline preserves rank, so no `shared-anchor` / `extend-top` / anchor
  choice can turn A7r into a pass. **The lever is the WEIGHTS.** Spline-only levers
  still move A1/A3 (range properties).
- The peaks arms (`lam1em3`, `Dpeaks`) fail **three** codecs each here — the jxl
  inversion `d_peaks_jxl_floor` §4 localised to the weights **persists** on new
  pixels and a new encoder era, as pre-registered.
- **`zenav1-svt` is pinned at `2d75a105f`** in zenavif (`2ebca1b4`). MEASURED on
  zenavif's own AVIF still-encode path: **1.498x** on summed `encode_ms` / 1.482x
  wall, with **9/9 cells byte-identical**. That is 1.50x, **not** the "2x" the
  encoder work was described as.
- **`imazen/jxl-encoder#101` came out of this run**: at butteraugli **distance >=
  10.0 exactly** (9.9 is fine), the encoder writes a `SizeHeader` rounded UP to even,
  so a 513x769 source declares 514x770 and cannot round-trip. Read from the
  codestream's own header, so it is encoder-side. Pre-existing (the 2026-07-27 sweep
  shows it). It removes those 13 ladders' entire jxl FLOOR, so they are excluded as
  truncated-floor rather than graded several steps up the curve.

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

## `--regime` IS DERIVED AND PRINTED; a feature-set id has NO layout (2026-09-06)

Records: [`benchmarks/regime_derived_2026-09-06.md`](../../benchmarks/regime_derived_2026-09-06.md),
[`benchmarks/layout_free_feature_set_id_2026-09-06.md`](../../benchmarks/layout_free_feature_set_id_2026-09-06.md),
[`benchmarks/table_densify_census_2026-09-06.md`](../../benchmarks/table_densify_census_2026-09-06.md).

**Stop typing `--regime N` to mean a width.** `bake_verdict` derives it from the
bake (`feature_set::derive_regime` — the narrowest REGISTERED layout that
carries every id it reads) and prints it on every run. A passed value that
cannot carry the bake is **refused**, so a bare `bake_verdict --bake <944 bake>`
now refuses instead of silently scoring at the 372 root. **Shipped `B` at
`--regime 720` is now refused too** — 720 zeroes `f156..371` and `B` reads 49
lines there; nothing checked 720 before. Still allowed, deliberately: a WIDER
regime that carries every id (the board's ext720 era rows), a `…pools` root
(the block is live there), and `--regime 720` beside an explicit
`--features-root` (the frozen as-run LOO drivers). The flag survives as a
**preset** selector — it also picks the corpora list and both grids — so the 195
`720/924/944` literals in scripts are NOT redundant and were not deleted.

**A feature-set id no longer carries `@w<N>`.** Canonical form is
`<compute>/<era>#<hash8>`; two ids differing only in the width are EQUAL and
hash equal. Every `@w<N>` string ever written still parses and still resolves —
the registry is append-only and indexes the layout-free spelling of every key.
The width lives on the ARTIFACT (`FeatureSetRef::layout`), and `check` reports
it only as a SHORTFALL (the consumer needs a wider row than the producer emits);
a NARROWER consumer is the dense design, not a mismatch.

**Two table findings, both measured, neither fixed here:**

- **An all-zero column is not an absent column.** The postC 372 root declares
  all 372 populated and still has all-zero columns (`f25` in aic3, `f12` in
  konjnd, EIGHT in the 50-row `ext_sdr25`) — small-corpus accidents, prune
  class 3. So `rescore_parquet --densify` takes the populated set from the
  DECLARATION (`--keep-ids`, mandatory) and uses a full-column scan only as a
  GATE. **Never infer a table's populated set from its values.**
- **A 944 table declares 39 ids the walk never writes** (`APPEND_SKIP_B_SCALE0`
  + `LUMA_MEAN_REF` + the HDR-gated `HL_BIN*`), and both the registry regime and
  `Plan::emit` say `0-943`. Modelling the placement rules would make `check`
  refuse any bake reading one — and **shipped `CHdr` reads 8 of them**. Pinned
  by `a_944_regime_declares_39_ids_the_walk_never_writes`; the fix is a product
  decision.

**No live table root was rewritten.** A dense-by-id table is one the loaders
cannot READ yet — they now REFUSE it loudly rather than truncating at the gap,
which is what a dense table used to get (a `f0..f155, f372..f943` table loaded as
**156-wide with no error at all**).

## SHIPPED BAKES DECLARE THE IDS THEY READ — four of six flipped (2026-09-06)

Full record: [`benchmarks/dense_bake_flip_2026-09-06.md`](../../benchmarks/dense_bake_flip_2026-09-06.md).
Contract + tool: [`benchmarks/dense_bake_contract_2026-09-06.md`](../../benchmarks/dense_bake_contract_2026-09-06.md).
Plan: [`docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../PLAN_CRUFT_PURGE_2026-09-06.md).

USER RULING (2026-09-06, verbatim): *"get rid of the cruft and confusion … a 372
layout where the bake skips features and features aren't computed is a bad
contract"*.

**`A`, `B`, `BHdr` and the default `D` now ship `*_byid_2026-09-06.bin`** —
`caller_input_width() == n_inputs() == |declared ids|`, **zero `Drop`**
transforms, `zentrain.feature_ids` naming the ids, and the scorers GATHER.
`D` reads **28** ids (`f6..f155`) where it used to declare 372; `B` 95, `BHdr`
133, `A` 285. **No number moved** — `Zensim::compute` is bit-identical on the
20-cell parity matrix and `bake_verdict --full-json` on 31–33k fields shows
**0** statistic diffs; shipped B still reads CID22 **0.8821166166351724**.
The wide bytes stay committed and the gate `include_bytes!`es both.

**Three things to know before citing or changing a bake:**

- **A bake's `n_inputs` is no longer its caller width's stand-in.** For a dense
  bake `n_inputs` is the PACKED width. **Never slice a feature row to
  `n_inputs`** — build a `CallerGather` (`bake_runtime::CallerGather::for_model`,
  the one owner, resolving through `zensim::declared_feature_ids`) and let it
  fill. That defect shipped once already: the `--full-json` kadis per-pair block
  sliced positionally and got **4,920 of 4,928 predictions wrong** on shipped D,
  silently, in a block the gauntlet renders.
- **Two admission rules exist and they are not interchangeable.**
  `accepts_row_width` is the GRID rule (exact `==`); `accepts_prefix_row_width`
  is for sources whose contract is "take the leading `n_inputs` columns" (the
  720-column kadis per-pair table feeding a 372-input bake). Using the grid rule
  on a prefix source silently DROPS the block — a coverage loss, not a wrong
  number, and just as bad in a published verdict.
- **`C` and `CHdr` are deliberately NOT converted, and it is a pending USER
  decision, not a backlog item.** They serve on `append2_dst_activity: true`
  (the `everything` fallback) while their training extraction ran it **false**,
  so the honest conversion moves a shipped score by **0.866** (`C`) / **0.311**
  (`CHdr`) zensim points. Do not touch their bytes or their served toggle.
  `dense_bake_flip_gate::flipped_bakes_are_dense_and_the_c_pair_is_deliberately_not`
  fails if either is densified while the decision is open.

## FEATURE SETS ARE NAMED, NOT COUNTED (2026-09-05, user directive)

**A count is not a feature-set identity.** `156` / `372` / `720` / `924` / `944`
/ `156+free` / `+classC` are LEGACY ALIASES only: "944" has named **seven**
different feature sets in this repo (two eras of one compute set; three pool
states — zeroed / all-216-live / ten carriers — at the same width; two free-set
arms sharing only `f0..227` with any of them; populated-slot counts
728/728/944/944/738/265/289), and one root's manifest records its regime as
"mixed by file". The identity is
**`<compute>@w<layout>/<era>#<slots-hash8>`** — e.g.
`basic+peaks+moments@w944/era2r4#4fcef1d6` — where COMPUTE is the closed token
vocabulary of populated slot families, LAYOUT is the emitted width, ERA is the
registered extractor build class, and the hash is
`zensim::feature_set_id::slots_hash8` (the ONE owner) over the sorted slot list,
so **the name is a handle and the hash is the identity**. Design +
alias table: [`docs/FEATURE_SET_IDS.md`]; registry (append-only):
`benchmarks/feature_sets_registry.json`. Every new bake carries
`zentrain.feature_set_id`, every new table carries `"feature_set_id"` in its
`_MANIFEST.json`, and `bake_verdict` prints both ids and reports every
disagreement (`--require-feature-set-match` refuses) — which is what closes the
`--regime 944` silent-mis-scoring bug at the root rather than at one block.
This SITS UNDER the append-only feature-numbering directive (2026-07-19: "new
v2 features use indices after all v1 features; we deprecate not renumber"): an
id names a SUBSET of the existing numbering and never renumbers it, and a new
block extends the token vocabulary exactly as it extends the numbering.

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
| **The SCORE arithmetic** — both mixing heads, the tanh output pin, the distance→score mapping, PCHIP derivs + eval, `POOL_STD_FLOOR` | **`zensim::score_math`** (`#[doc(hidden)] pub`), routed through `zensim::det_math::PowForm` | a `powf`/`exp` anywhere on a score path — gated by `zensim-validate/tests/no_score_path_libm.rs` |
| A `--bake-post` mode (`raw`/`extrapolate`/`clamp`/`mapped[:A,B]`) | `bake_runtime::apply_post_mode` | a per-bin `fn apply_post` (there were three, byte-identical) |
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
`/home/lilith/work/zen/zensimx/zensim-validate/src/bin/bake_verdict.rs`.
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
`--full-json`).

**★ THE HEAD IS A TREE NOW, AND IT HAS ITS OWN WIRE FORMAT — `ZCTH` v1
(2026-09-06, `benchmarks/corruption_head_serving_2026-09-06.md`, ledger ROUND
101).** The right model form is a gradient-boosted tree, not a logistic
(`corruption_head_theories_2026-09-06.md`: **98.90 % detection at 1.23 % honest
and 2.38 % near-lossless FP**, against the shipped logistic's 86.01 / 11.37 /
**50.00**, with no dial guard). It is served by `zensim::corruption_head`
behind the **non-default `corruption-head`** feature.

* **`--corruption-head` now takes EITHER format, sniffed by MAGIC** — `ZNPR`
  (the incumbent logistic) or `ZCTH` (the tree). Neither → a named refusal.
  The tree is deliberately **not** a ZNPR metadata blob: `zenpredict` is frozen
  at the `zenanalyze-api` contract, and every consumer holding a
  `zenpredict::Model` dispatches through `Predictor::predict`, so a tree behind
  a plausible identity layer is **silently mis-scored by anything that does not
  know to look** — the `--regime 944` defect shape.
* **`zensim::corruption_head::gate_score` is the ONE owner of the deploy
  composition** (`min(perceptual, 0)` when flagged, else passthrough). Adopting
  it left `bake_verdict --full-json` **byte-identical** on the logistic head.
  **It can only sort a corruption below an anchor whose own score is ABOVE
  ZERO** — true of the grid's q20 anchors, not automatic.
* **Numbers you can quote:** through the Rust evaluator the tree head
  reproduces the theory lane's gate grid EXACTLY — DEPLOY `pass_q20`
  **0.9985119047619048** (671/672), head `pass_q20` 1.0, D's dial alone
  0.26785714285714285. Parity vs sklearn: **0 ulp** on `decision_function`,
  **max |Δ| 3.33e-16** on the calibrated probability, fire set identical on
  35,607 rows.
* **Cost, and it inverts the intuition:** the tree forward is **659 ns**
  (0.63× Profile D's own forward, ~0.011 % of a 576² compare) while the
  **INCUMBENT LOGISTIC head costs 1.76 µs — 2.7× the 6,100-node tree and more
  than the dial it guards.** The nonlinear head is not a speed tradeoff.
* **`f156..228` is not just cheap at D, it is PRESENT:** `ZensimProfile::D`
  emits 372 features with the peaks block populated **72/72**, zeroing only
  `f228..371`. `CorruptionHead::check_servable_by` refuses a head that reads
  outside the profile's plan, so attaching one can never widen the walk.
* **Gotcha when reimplementing sklearn's isotonic:** `IsotonicRegression`
  builds a `scipy.interpolate.interp1d(kind="linear")`, but `interp1d.__init__`
  routes plain `linear` to `_call_linear_np` = **`np.interp`** (rightmost
  bracket, slope form) — NOT the `_call_linear` you find by reading the source
  (leftmost bracket, convex combination). MEASURED: `np.interp` is bit-identical
  to `iso.predict`, the other is off by 1.11e-16 *on the knots*. Reading the
  constructor is not reading the call.
* **Deadband:** the trainer's heuristic is form-dependent (0.9 for the logistic,
  **0.5** for `hgb`, whose severe FP is already <1 % there), so pass
  `--deadband-t 0.9` to bake the REGISTERED operating point. `bake_verdict`
  prefers a ZCTH head's own baked deadband unless
  `--corruption-head-threshold` is passed, and `--full-json` reports the value
  it APPLIED.
* **Nothing shipped and the public API delta is ZERO.** The runtime companion
  (`Zensim::with_corruption_head` / `corruption_verdict`) is `#[doc(hidden)]`,
  feature-gated, and off `compute`'s path; the proposed public shape awaits the
  user's approval in `docs/PLAN_CORRHEAD_SERVING_2026-09-06.md` §3. Gates:
  `scripts/verify_corrhead_serving.sh` + 18 in-crate tests.

**⚠ THE CORRUPTION HEAD: read this before quoting any corruption number
(2026-09-05, `benchmarks/corruption_head_d_2026-09-05.md`, ledger ROUND 97).**

* **`--corruption-head` takes a ZNPR BAKE, never the JSON head.** Every
  2026-07-24 artifact under `corruption-head-2026-07-24/` is a `.json`, so
  **no 372 head had ever been through the gate** until this lane baked one.
  `train_corruption_head.py --bake-out` now emits it (via `zenpredict-bake`;
  the winsor clip rides as a raw-space `winsor_p99`, unused lines are `drop`
  so `caller_input_width` stays at the grid's width, and the layer is NEGATED
  so the score is quality-oriented `100*(1-P)` and the spline stays monotone).
* **A head must match its profile's read-set or it is not free.** `Off` and
  `Peaks` cost the SAME (`fold_engine.rs`), so `f0..227` is free for shipped D
  and `f228..371` would force `V1PoolsMode::Full`. The 2026-07-24 head reads all
  372, so attaching it to D would silently change extraction cost. Within D's
  free set, peaks are worth +1.0 point of detection and cut severe-honest FP
  0.41 % -> 0.31 % for nothing — always take `f0..227` over `f0..155`. But
  masked/IW is worth **+4.8 points** on top (bake-vs-bake on identical held-out
  rows: 2026-07-24 **90.7 %** vs `d228` **85.9 %**, same 0.31 % severe FP), so
  the 2026-07-24 ablation's "the signal needs mask/iw/peak" is **SUPPORTED**.
  D's companion is a trade — 85.9 % free, or 90.7 % by forcing `Full`.
* **`train_corruption_head.py` reports a model it does not ship.** The threshold
  curve comes from a `CalibratedClassifierCV`; the persisted head is a plain
  `LogisticRegression` refit on train with an isotonic fit on val. True since
  2026-07-24. The gap is material (`d228` held-out ladder FP 15.83 % reported vs
  **11.22 %** as baked). **Quote the bake**, and measure it with
  `predict_features_with_bake`, not the training log.
* **The gate cannot select a head.** Its only honest rows are two anchors from
  ONE reference (`gb82_dog`, 672 triples). The `--no-broad-honest` ablation WINS
  the deploy gate (99.1 % vs 91.4 %) by being trigger-happy and is the worst arm
  on honest content. Always read the gate beside
  `scripts/v_next/corruption_head_honest_fp.py`.
* **The false positives are at NEAR-LOSSLESS, not at low q.** On the ladder
  instrument `d228` fires on 0.0 % of cells below q50 and **53.7 % at q95-100**
  (avif-rav1e **97.2 %**); 1,134 of 1,139 flagged cells sit at q ≥ 80. A
  corruption confined to an 8×8 square is *also* nearly identical to its
  reference, so globally-pooled v1 features cannot separate the two — more
  negatives do not fix it. A `dial < 90` guard reads 64.0 % gate at 0.74 %
  honest FP (`dial < 80`: 47.9 % at 0.00 %); that guard is a **proposal**,
  measured and implemented nowhere. `bake_verdict`'s DEPLOY section implements
  the registered unguarded `min(perceptual, gate)` (`--corruption-head-threshold`,
  default 10.0 = `T = 0.9` in `100*(1-P)` units) and nothing else.
* **D's corruption weakness is intrinsic, not an era artifact.** Re-extracting
  the persisted gate PNGs at HEAD moves 73.7 % of basic cells (max |Δ| 4.35) and
  D's `pass_q20` goes 26.9 % → 26.8 %. The era DOES matter for the tables (every
  stored corruption table is pre-option-C; negrich moved on 100 % of rows) — it
  does not matter for this ordering. **`freeze_check`** (same crate) turns one bake's fulleval
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
`/home/lilith/work/zen/zensimx/zensim-validate/src/bin/subset_sim.rs`.
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
`/home/lilith/work/zen/zensimx/zensim-validate/src/bin/panel.rs`.
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
`/home/lilith/work/zen/zensimx/zensim-validate/src/bin/zensim_mlp_train.rs`.
Supports `--group <name>:<path.parquet>:train_w:val_w` (auto-detects
.csv vs .parquet by extension), `--target-column NAME`,
`--feature-set extended_iw|extended|standard`, full PWRC+NiN flags,
auto-transforms via `--auto-transforms <screen.tsv>`.

### Eval matrix comparison (N bakes side-by-side)
**`scripts/cvvdp_matrix_compare.sh`** at
`/home/lilith/work/zen/zensimx/scripts/cvvdp_matrix_compare.sh`. Runs
bake_verdict on every `*.bin` in a dir + emits a per-corpus
SROCC/Z-RMSE/PWRC table for ship-decision review.

### Bake dial / spline refit + tail gate (bake_dial_refit)
**`bake_dial_refit` binary** at
`/home/lilith/work/zen/zensimx/zensim-validate/src/bin/bake_dial_refit.rs`.
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
# DENSE CONTRACT: declare the ids the bake reads, drop the wide caller width
# and every FeatureTransform::Drop. Two gates (dead-column + identity), both
# bit-exact; refuses to write on either failure. docs/PLAN_CRUFT_PURGE_2026-09-06
bake_dial_refit densify --in <bake.bin> --out <dense.bin> [--dry-run] [--gate-rows N]
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

## API Stability

**Run `just api-doc-check` before every push that touches `pub` items.** It
regenerates the `docs/public-api/{zensim,zensim-regress}{.txt,.features.txt,.internal.txt}`
snapshots — via the workspace-`exclude`d `apidoc/` package (`zenutils-apidoc`
crate; see `apidoc/Cargo.toml` for why it's excluded) — and fails if the
committed files don't match what a fresh regen produces, so a change to the
public (or `#[doc(hidden)]`) surface always shows up as a reviewed diff next
to the code change that caused it. `just api-doc` regenerates in place
(commit the diff); `just api-doc-check` only verifies, same as CI's
`api-doc-check` job in `.github/workflows/ci.yml` (ubuntu-only, cached via
`Swatinem/rust-cache@v2`). That CI job did not exist before 2026-09-06 — the
snapshots could, and did, drift silently with nothing to catch it. The
rustdoc-JSON nightly is PINNED (`apidoc_toolchain` in the justfile, kept in
sync with the CI job's `ZEN_API_DOC_TOOLCHAIN`): an unpinned tracking nightly
churns cross-crate path rendering with zero repo changes (MEASURED
2026-09-06: the `core::io` re-homing turned `std::io::error::Error` into
`core::io::error::Error` across 11 lines of `zensim-regress.txt`, a false
diff). Bump the pin deliberately, in the same commit as a `just api-doc`
regen.

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
