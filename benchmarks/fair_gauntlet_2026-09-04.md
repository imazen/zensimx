# FAIR GAUNTLET — corrections applied + the fairness filter (2026-09-04)

**User request, verbatim:** *"generate an updated gauntlet with all of the things we
discovered corrected, and the data/models filtered to those we can verify we did
fairly."*

Two boards ship. Nothing was deleted; no threshold was relaxed; **no statistic in this
pass was re-derived** — every number rendered is read from a fulleval JSON, the
annotations registry, the G-ADDR floor registry, or a verbatim transcription of the
ssim2 exam.

| board | rows | bytes | cap |
|---|--:|--:|---|
| **`summer_gauntlet_fair.html`** — the VERIFIED-FAIR board (primary) | 97 of 433 | **7,585,643 (7.2 MB)** | **under the 12 MB cap** |
| `summer_gauntlet.html` — the all-rows companion | 433 | 20,331,817 (19.4 MB) | over cap; was 22,949,515 (21.9 MB) |

The all-rows board is **2.5 MB smaller than before** despite gaining four panels' worth
of columns, because LEGACY rows now lose their embedded per-pair point cloud (the
registered `--strip-per-pair` rule, extended). Their per-pair data is untouched in the
source verdicts. **The all-rows board still exceeds the cap and is reported, not
silently trimmed** — cutting it further would mean dropping rank/dial DATA, and the
brief forbids that.

Gates on both files: `scripts/v_next/gauntlet_gates.sh` — `node --check` on every
`<script>` block, then the DOM-shim render harness. **PASS on both.**

---

## 1. The corrections, and which were already consumed vs newly rendered

| # | correction | state before this pass | now |
|---|---|---|---|
| 1 | **best-of-k seed headlines** | not represented anywhere on the board | **NEW**: seed grouping (`gauntlet.seed_group_key` / `build_seed_groups`), `k` / `composite k-mean` / `k-spread` columns, per-seed values on hover, UNREPLICATED badge, registry entry `seed-group-single-draw-2026-09-04` |
| 2 | **ensemble/teacher units defect** (`58baf010`) | not registered, not rendered | **NEW**: two registry entries — one flagging the 31 teacher-derived rows, one **clearing** the 15 ensemble rows (their numbers are on the correct side of the fix) |
| 3a | stored-372 thread-dependent reads | registered `names[2]`, **consumed** | consumed; now also a fairness criterion (b) |
| 3b | `board372-row-read-on-ext720-root` | registered `names[7]`, **consumed** | consumed; read as a *measured correction*, so it does **not** fail a row |
| 3c | KADID ext inversion | registered `names[188]` + `{all}` corrected, **consumed** | consumed; never fails a row — KADID is an integrity guard, not ranking signal |
| 3d | diluted KonJND ruler | repaired-form `names[17]` consumed; the two *pre-fix* entries are `manual` → **badge nothing** | reported (below); the machine-active repaired entry is what a reader sees |
| 3e | jpeg-aic-family-holdout | `manual` → **badged nothing** | **NEW machine-scoped twin**, `{all: true}` on `rank.aic3/aic4/sdr25` |
| 3f | era-2 tiling exception (BHdr) | `manual` → **badged nothing** | **NEW machine-scoped twin**, `names[2]` |
| 3g | hfnl = ssim2 self-target | **NOT IN THE REGISTRY AT ALL** | **NEW** `hfnl-ssim2-self-target-circular-2026-09-01`, `{all: true}` over all four circular axes |
| 4 | free-40 `GLOBAL_CGAIN`/`CLOSS` precision skew | not registered | **NEW**, `names[33]`, bounded (annotated, not invalidated) |
| 5 | **G-ADDR dial addressability** | emitted by `bake_verdict`, *"nothing reads it yet"* | **NEW** `G-ADDR p/f` column + per-axis hover |
| 6 | **the composite's circularity** | unstated on the board | **NEW** Beats-ssim2 evidence panel + W1–W7 columns, `peer_ssim2` as the reference row |
| 7 | **train==val corpora** | `kadid-tid-train-eq-val` registered `{all}`, badge only | now **dimmed + italic + reason on hover**, and excluded by construction from every ranking view |
| 8 | `serde_json` float-parse ULP hazard | fixed at the Rust owner | **verified, no action needed** — see §6 |

### 1.1 Seed groups — the rule, and how it is presented

**Grouping rule** (owner: `gauntlet.seed_group_key`; `freeze_check --select` is adding a
`--seed-group` key concurrently and **must adopt the identical rule** — see §7):

1. Single-model rows only. An ensemble is an evaluation FUNCTION over members, not a
   training replicate.
2. Key = embedded `zentrain.repro.argv` with `--seed` and the output-path flags (and
   their values) removed. No argv ⇒ UNGROUPABLE.
3. Collapse by `repro.seed`. Two cells with the same recipe **and the same seed** are
   one training run promoted twice, so `k` = **distinct seeds**, never board rows.

**Rule 3 is load-bearing and was measured, not assumed.** `A4b_s4004` and `FC_C0_s4004`
have identical argv modulo seed, identical seed 4004, and identical CID22 to 16 digits
(`0.8902951295793322`) with different bake sha256. There are **42 such rows in 33
same-seed groups**. Counting them separately inflates `k` and makes a group look better
replicated than it is (registry `duplicate-promotion-same-seed-2026-09-04`).

**VALIDATION.** The rule reproduces the fastclass record's §7.1 table exactly, on that
record's own arm and without reading it: k=3, KonJND best **0.4327** / mean **0.3561** /
spread **0.1329**, composite best **0.8664** / mean **0.8572**. That is an independent
check of the grouping against a number nobody computed for this pass.

**Presentation (USER CORRECTION, 2026-09-04):** *"data-subsets are not equal though — one
might be more representative and diverse, while another sucks, objectively."* A seed
drives pair **sampling**, so part of a group's spread is subset coverage rather than
model variance. Therefore:

* a group renders as **mean + spread (min–max) + k**, with **per-seed values on hover**
  and every member row still on the board;
* the mean **ranks** the default view — it is the honest estimator against best-of-k —
  but it is **never labelled definitive**;
* the badge says so verbatim: *"seed spread partly reflects subset coverage; coverage
  metrics pending (ownerfix lane adding `zentrain.sample_coverage`)"*;
* when a bake carries `zentrain.sample_coverage` it renders beside the seed. **Absent on
  433/433 rows today**, so it renders NOT MEASURED — never a zero.

### 1.2 Ensemble units — who is actually affected

`58baf010` (2026-09-04) added `predict --score-units`: `bake_dial_refit predict
--ensemble` accumulated **RAW** network output while `bake_verdict --ensemble`
accumulates **SCORE** units. Determined by provenance, not guessed:

* **The 15 ensemble ROWS are unaffected.** Their rank/dial numbers come from
  `bake_verdict --ensemble` — the correct side. Registered as a *clearing* annotation
  (`ens-board-rows-score-units-unaffected-2026-09-04`) so the defect is not read onto
  cells it never touched.
* **31 teacher-derived rows ARE affected** — every board row whose repro argv names the
  HYA teacher table (`A4b`, `A4_r4`, the 21 `FC_*`, the 8 `a4bkon_*`). Their own
  measured numbers are honest; what is invalid is the **teacher premise**: the KonJND
  peak is at w≈0.5–0.6 (0.5390), not the w=0.84 they were built at, and the teacher they
  actually distil from reads **0.5019 — below ssim2's 0.5272**. These fail criterion (f).

---

## 2. The fairness filter

A row is **VERIFIED-FAIR** iff all of:

| id | criterion |
|---|---|
| `a_repro` | embedded `zentrain.repro` (or a committed byte-verified reproduce script) |
| `b_era` | no registered ERA defect invalidates its ruler |
| `c_no_train_eq_val` | its ranking composite uses no train==val corpus |
| `d_seed` | seed group aggregated (k≥2), or badged UNREPLICATED |
| `e_no_invalidated` | no unresolved `invalidated` registry entry applies |
| `f_ens_units` | if ensemble/teacher-derived, built post-`--score-units` |
| `g_split` | its eval read the canonical held-out val group |

**Criterion (d) never fails on its own** — an unreplicated cell is fair as long as it is
*badged* as one. That is the brief's rule, and it is what keeps k=1 rows honest instead
of invisible. Rows passing everything but (d) are **FAIR-NOTED**.

**(a)'s script exception** covers the three bakes with a committed byte-verified
reproduce script: `b_sdr_linear_cid80_inclwinsor_dense_dial` (`scripts/reproduce_b.sh`),
`bhdr_linear_shaped_cvvdpmix` (`reproduce_bhdr.sh`), `v47_strict_QAT_native`
(`reproduce_v47.sh`).

### 2.1 Counts

| tier | rows | note |
|---|--:|---|
| **VERIFIED-FAIR** | **42** | in **14 distinct recipes** (every one is a k≥2 seed group) |
| **FAIR-NOTED** | **55** | 43 UNREPLICATED (k=1) + 12 ungroupable |
| **LEGACY** | **336** | default-hidden on the all-rows board; absent from the fair board |

| failing criterion | rows |
|---|--:|
| `e_no_invalidated` | **278** |
| `a_repro` | 40 |
| `f_ens_units` | 31 |
| `b_era` | 2 |

### 2.2 The single largest source of unfairness is a stale cached scalar

`e_no_invalidated` is driven almost entirely by
**`composite-stale-after-rank-graft-2026-08-28`: 276 of 433 cells (64% of the board)**.
The 2026-08-28 family-aware reslice replaced `rank.imazen26` + `rank.nonphoto` — two of
the six terms of `product_composite` — and did not update the stored `composite`, so
those cells **disagree with their own rank blocks** (|Δ| up to 0.0237, median 0.0017).
That field is **the scoreboard's default sort key**. It was registered on 2026-08-31 and
badged; what this pass adds is that it now *decides tier*, so the default view cannot be
ordered by a number 64% of the board disagrees with.

### 2.3 A registry field-scope defect found and corrected (not silently overridden)

A first cut put **164 extra rows** in LEGACY via `balanced-composite-bandtail-abs`, whose
`fields` is `["composite"]`. Its reason and evidence are both about `freeze_check`'s
**`balanced_composite`**, but on a *fulleval* `composite` is **`product_composite`**,
whose weights (CID22 1.00 + imazen26 0.50 + nonphoto 0.30 + KonJND 0.20 + AIC-3 0.10 +
AIC-4 0.05) carry **no band term** — the `(|B3|+|B9|)/2` defect cannot reach it. One
dot-path names two different quantities.

Handled by the registry's own sanctioned mechanism rather than a quiet special case: a
new entry `balanced-composite-bandtail-field-scope-corrected-2026-09-04` records the
correction, and `gauntlet.E_FIELD_SCOPE_SUPERSEDED` is the code that reads it. The
original entry is unchanged and still governs `balanced_composite`. **Fix at the owner:
give `balanced_composite` its own field name.**

### 2.4 Criterion (e) is scoped to the ranking view, and that too was measured

A blunt *"any matched `invalidated` entry"* rule put **all 433 rows** in LEGACY, because
two `invalidated` entries carry `{all: true}` over `gates` / `class`. Those are caveats
on other columns and are already badged there. The rule instead uses the registry's own
coverage semantics: an entry fails a row iff one of its `fields` covers `composite` or
`rank.<axis>.srocc_signed` for an axis that survives the circularity and train==val
exclusions. So `rank.kadid.*` never fails a row, and `rank.cid22.bands` never does
either — a band is not the ranking scalar.

---

## 3. What moved at the top once best-of-k was replaced by means

**BEFORE** — today's default sort (raw `composite`, best-of-k, all 433 rows):

| composite | bake | tier | k |
|--:|---|---|--:|
| 0.8721 | `mlp_2L_diverse_H128@cur372` | LEGACY | — |
| 0.8687 | `PH_s4004_f054` | LEGACY | — |
| 0.8681 | `FC_D2_s4005` | LEGACY | 3 |
| 0.8676 | `HYA_w084` | FAIR-NOTED | 1 |
| 0.8672 | `PH_s4004_e060` | LEGACY | — |
| 0.8664 | `W10L9PH_s4004_packed` | **VERIFIED-FAIR** | 6 |

**Every one of the current top three leaves the fair view**, and for three different
reasons: two on a stale composite, one (`FC_D2_s4005`) on the teacher-units defect —
and that one is *also* the best of 3 seeds, whose group mean is 0.8657.

**AFTER — VERIFIED-FAIR only**, every row a k≥2 group, ranked by k-mean composite:

| k-mean | best-of-k | spread | k | CID22 k-mean | KonJND k-mean | recipe |
|--:|--:|--:|--:|--:|--:|---|
| **0.8595** | 0.8657 | 0.0123 | 2 | 0.8828 | 0.4769 | `A5_r4_s4004` |
| 0.8593 | 0.8664 | 0.0117 | **6** | 0.8848 | 0.4609 | `W10L9PH_s4003_packed` |
| 0.8589 | 0.8602 | 0.0028 | 2 | 0.8879 | 0.4613 | `W10L9_s4003_packed` |
| 0.8585 | 0.8601 | 0.0032 | 2 | 0.8868 | 0.4434 | `A1foldapp2_r4_s4004` |
| 0.8569 | 0.8581 | 0.0026 | 3 | 0.8842 | 0.4493 | `W10L9PBR_s4003_packed` |
| 0.8563 | 0.8620 | 0.0176 | **9** | 0.8823 | 0.4592 | `W10L9P_s4003_packed` |

The top two are **0.0002 apart on a median k-spread of 0.0164** — i.e. not
distinguishable, and the k=6 and k=9 groups are the only ones with enough draws to say
so. **Best-of-k inflates the composite by a median +0.0066 and up to +0.0222** across
the 14 verified-fair recipes.

**The finding that matters more than the ordering:** once the fair filter is applied,
**the highest-scoring rows are almost all k=1**. `HYA_w084` leads the combined
fair view at 0.8676 with **k=1**, and 8 of the top 8 combined-fair rows are
UNREPLICATED. The board's leaderboard is largely made of single draws, on a population
whose measured composite spread at fixed recipe is 0.0164 median / 0.0445 max. That is
the same failure the fastclass record found on one arm, generalised.

---

## 4. G-ADDR — what is rendered, and what is honestly absent

`dial.addressability` is emitted by `bake_verdict --full-json` as of 2026-09-04, and
**no board fulleval carries it** (0 of 433 — the gate landed after their verdicts). So
the column renders what *is* stored: the six axes every cell already has
(`p5`/`p95`/`reach`/`dynamic_range`/`mono_pct`/`tied_pct`) against the committed bars —
**A3, A4, A5, A6** (REGRESSION, bar = shipped B on the canonical grid) and **C1, C2**
(CONTRACT). The other nine — A1/A2 (pooled min+max), A7–A9 (negative-tail probe),
C3–C6 (negative-tail + identity probes) — render **NOT MEASURED with that reason**,
never a zero. If a cell ever carries an emitted block, it is picked up automatically.

**The `reference = peer_ssim2` pin set LANDED mid-pass** (the concurrent lane's
`56d405ad`, *"re-pin the REGRESSION tier to peer_ssim2 — the reference metric, not the
incumbent"*, arrived on the rebase). The registry was re-read as it then stood and the
consumer re-pinned to it. The discriminator is a `reference` field on each instrument
row; the registry's own `_schema.reference_sets` names `peer_ssim2` **ACTIVE (bars)** by
user decision (*"I don't think we should pin to B, ssim2 seems a better mentor"*) and
`shipped_b` **"printed, never a bar"**, labelled BIASED. The board renders both: the
ssim2 pins decide PASS/FAIL, the shipped-B column rides alongside as context so a reader
can always tell *"worse than the mentor"* from *"worse than what shipped"*.

The re-pin reproduces the registry's own headline on the board. Shipped **B** now reads
**A3 PASS / A4 FAIL / A5 FAIL / A6 PASS** — the difficulty moved from the ceiling to the
**floor**, exactly as `_schema.reference_sets["not a relaxation"]` says (ssim2's grid min
is −55.35 against B's +3.13). Board-wide over the six measurable axes, the modal cell is
**3 pass / 3 fail** (278 of 433).

~~The board still does **not** badge NOT-SHIPPABLE on a CONTRACT failure, because **no cell
has a CONTRACT-tier measurement**~~ — **SUPERSEDED 2026-09-04, same day, by the
board-coverage lane (gate doc §15).** All 97 fair cells were then graded through the owner
under both pin sets; 96 carry the verdict as a grafted `dial.addressability` block, the
G-ADDR column reads `pass/15`, and **47 cells fail a contract row and now badge NOT
SHIPPABLE** (46 of them on the fair board — `ebothg_m504`'s read was refused by the
same-grid gate). C5/C6 remain NOT MEASURED on 94 cells because no in-era 944 identity probe
exists and the 372 one cannot be widened (measured: the 944 identity vector is not the zero
vector). An INCOMPLETE contract still badges nothing — unmeasured is never a fail.
(`~/tmp/gaddr_repin_READY.md` did not exist at this pass; nothing was assumed from it — the
landed registry was read instead.)

---

## 5. The composite is circular, and the board now says so

`product_composite` = CID22 1.00 + **imazen26 0.50** + **nonphoto 0.30** + KonJND 0.20 +
AIC-3 0.10 + AIC-4 0.05, normalised by present weights. **imazen26 and nonphoto are
ssim2-anchored**, so 0.80 of 2.15 — **37% of the ranking scalar** — is agreement with
ssim2. That is the arithmetic by which a `peer_ssim2` row can top a board built to rank
models *against* ssim2.

The **Beats-ssim2 evidence panel** answers the question without inventing a new
composite. It shows the six genuinely held-out human axes (CID22, CSIQ, LIVE, AIC-3,
AIC-4, KonJND-JPEG-504) as **differences against the `peer_ssim2` row**, with the exam's
own derived δ (0.010 pooled CID22, 0.010 elsewhere), a **count of losses beyond δ** —
which is W1's own rule applied per row, a count of already-measured differences, not a
new statistic — and the **transcribed W1–W7 verdicts**
(`benchmarks/ssim2_exam_scorecard_2026-08-31.json`). Excluded by construction:
`nonphoto`, `imazen26`, `hfnlproxy`, `hf_nearlossless` (circular) and `kadid`, `tid`
(train==val). Rows outside the exam's six scored candidates read **NOT MEASURED**, never
a fail. The panel repeats the exam's headline: **nobody passes.**

---

## 6. Verified, no action needed: the `serde_json` ULP hazard

The dialgate lane found that `serde_json`'s default float parser is not correctly
rounded (a bar written `99.98330778475787` parsed back as `…788`, and the reference bake
failed its own bar by one ULP). Checked both consumers this pass:

* **`freeze_check` — COVERED.** The fix is `serde_json = { features = ["float_roundtrip"] }`
  at the `zensim-validate` crate level, so every binary in that crate (including
  `freeze_check`, `bake_verdict`, `panel`) gets the correctly-rounded parser, guarded by
  `registry_floats_round_trip_bit_exactly`.
* **`gauntlet.py` — NOT AFFECTED.** CPython's `json` float parser uses the platform
  `strtod`, which is correctly rounded. No fix applied, because none is needed.

Carried forward for anyone else: **a Rust tool outside `zensim-validate` that parses
these JSONs does not inherit the feature.**

---

## 7. Findings worth carrying forward

1. **Half the registry badges nothing.** 30 of 61 entries (now 30 of 70) carry
   `scope: {"manual": …}`, which by the registry's own rule matches zero cells. Three of
   the items this pass was asked to render were in that half, and one — the hfnl
   self-target — **was not in the registry at all**, despite being written in the exam
   doc and in CLAUDE.md. A finding that is registered but unscoped is invisible in
   exactly the way the `_schema`'s own D10 note warns about.
2. **The Python mirror of `ann_matches` had drifted from its Rust owner.** It read only
   `o["name"]`; `freeze_check.rs`'s `bake_name` falls back `name` → `bake` → `"?"`. A
   fulleval with `bake` but no `name` matched in one and not the other. **Fixed** — the
   Python side now mirrors the fallback.
3. **A gate hole: the render harness's last third never gated.** The only
   `if (failed) process.exit(1)` sat at the file's midpoint, so the registry-badge check,
   the ECharts SSR check and the entire failure-panel test could print `FAIL:` and still
   exit 0 — the pre-2026-09-04 board printed **three** such FAILs on every run while the
   gate reported PASS. **Fixed** with a terminal check, and **verified with a negative
   control**: a deliberately broken failure-panel heading now returns rc=1 where it
   returned rc=0 before. Nothing was relaxed.
4. **The harness's copy of the default-visible rule is a liability.** It recomputes
   `curated && !dominated && !knobfail` independently of the page. When the default
   changed, the stale copy did not catch a bug — it *invented* one. Updated in lockstep
   and commented as such; a shared constant would be better.
5. **`sample_coverage` is the missing measurement.** Seed spread is currently
   un-decomposable into model variance vs subset coverage. 433/433 rows read NOT
   MEASURED. The board is wired for it the day the ownerfix lane emits it.
6. **PENDING UNIFICATION.** `freeze_check --select` is gaining a `--seed-group` key in
   the ownerfix lane; it had not landed at this pass's HEAD (`grep` clean across
   `zensim-validate/src/`). The rule above is the board's owner **for now**, and the two
   MUST converge on it — a `--select` winner and a board leader that disagree about what
   `k` means is worse than either alone.

---

## Reproduce

```sh
export ZEN_PANEL_BIN=<repo>/target/release/panel   # cargo build --release -p zensim-validate --bin panel
cd scripts/v_next
# the VERIFIED-FAIR board (primary) + the committed audit TSV
python3 bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
  --fair-only --fairness-tsv /mnt/v/output/zensim/reports/fairness_tiers_2026-09-04.tsv \
  --out /mnt/v/output/zensim/reports/summer_gauntlet_fair.html
# the all-rows companion
python3 bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
  --out /mnt/v/output/zensim/reports/summer_gauntlet.html
# MANDATORY on every emitted HTML
../../scripts/v_next/gauntlet_gates.sh /mnt/v/output/zensim/reports/summer_gauntlet_fair.html
../../scripts/v_next/gauntlet_gates.sh /mnt/v/output/zensim/reports/summer_gauntlet.html
```

Per-row audit: `benchmarks/fairness_tiers_2026-09-04.pointer.md`.
Registry appends: `benchmarks/eval_annotations.json` (9 new entries, append-only —
503+ insertions, 0 deletions).
Exam transcription: `benchmarks/ssim2_exam_scorecard_2026-08-31.json`.
