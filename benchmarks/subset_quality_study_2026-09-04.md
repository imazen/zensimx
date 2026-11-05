# Subset-quality study — what a "good" training subset looks like, and whether it explains seed spread

**Status: PRE-REGISTERED 2026-09-04. Hypotheses, descriptors, regression form and
falsification below were written BEFORE any descriptor→score analysis was run.**
The only measurements taken before locking this file were *mechanism* reads of the
trainer source (which RNG stream drives what) and a **power probe** sizing the
variance of the coverage descriptors under the real wave-r4 group structure — a
design input, not an outcome. Section 3 states what that probe was allowed to
change (grid/window sizing) and what it was not (hypotheses, targets, falsifier).

Lane: `claude-subsets`. Workspace `~/work/zen/zensim--subsets`.
User direction (2026-09-04): *"log good seeds and the subsets they find, across
multiple models, and figure out what good diversity/subsets look like and make
such more intentionally."*
Coordinator amendment (2026-09-04): *"init seed and sampling should be different
seeds"* — see §6.

---

## 1. Mechanism (read from source, not assumed)

`zensim-validate/src/mlp_train/mod.rs` runs **two independent RNG streams**, both
`SplitMix64`, both derived from the single `--seed`:

| stream | construction | drives |
|---|---|---|
| init | `SplitMix64::new(seed)` | `w1`/`w2` He-normal init |
| **sample** | `SplitMix64::new(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xDEADBEEFCAFEBABE))` | every RankNet pair draw |

(the 2-layer / min-max entry point at `mod.rs:6873` uses
`.wrapping_add(0x0123456789ABCDEF)` instead; `zensim-train-core`'s three legacy
heads use `seed ^ 0x5A5A…` / `seed ^ 0xA5A5…`.)

The separation is deliberate and documented in-source: it exists so a 228-vs-372
A/B sees the *same* pair draws even though init consumes a different number of
normals. **Consequence for this study: sampling is already decoupled from init in
the RNG, but not in the CLI — one `--seed` sets both, so every existing bake
confounds them.** Existing bakes are therefore attributed JOINTLY (§6).

The per-draw sequence (identical at `mod.rs:2265`, `3169`, `4003`; `7709` adds a
stratified variant) is:

```
u = rng.next_f64_unit()                       // 1 draw
train_pos = cdf.partition_point(|c| c < u).min(len-1)
n = groups[train_indices[train_pos]].features.len()
if n < 2 { continue }                          // consumes 1 draw
(ia, ib) = within_ref ? rb.draw(next_u64 ×3)   // 3 draws
         : per_row_cdf ? (next_f64_unit ×2)    // 2 draws
         : (next_u64 % n, next_u64 % n)        // 2 draws
if ia == ib { continue }
```

`train_indices` = groups with `train_weight > 0.0`, **in declaration order**;
`cdf` = cumulative normalised `train_weight`.

**Therefore the drawn multiset is a pure function of
`(seed, [train_w], [rows], epochs, pairs_per_epoch, boost flags, within_ref flags)`**
— and every one of those is recorded in the fulleval's embedded `zentrain.repro`
block (`inputs[].rows` is written as `g.human_scores.len()`, which is exactly the
`n` the draw uses). **No feature matrix is needed to reconstruct a subset.**

## 2. Population

312 board fullevals carry a usable `repro.argv` + `seed`. Canonicalising argv with
the seed value and `--out` elided gives **124 distinct arms, 83 of which have k≥2
seeds** (largest k=12, then 9, 9, 6, 6, 5, 5, 5, then twenty-two k=3 arms).
Model classes covered: the 944-class sota944 arms (`C_em944`, `C_co3a`, `C_nt944`,
`FS_K944`, `Q_hdr944`), the W8–W12 balance waves, the HDR944 arms, the
fastclass 156+free wave (`FC_C0/D1..D4/F1/G1` ×3), and the wave-r4 A-arms.

## 3. What the power probe was allowed to decide

Under the real wave-r4 structure (10 train groups, 708,301 train rows,
120 × 50,000 = 6,000,000 draws → ≈17 row-hits per row) whole-run coverage is
**expected a priori to saturate**, which would make binary coverage descriptors
degenerate and the study unfalsifiable-by-construction. The probe measures that
saturation so the descriptor set can be sized honestly. It may change:

- the **window** over which coverage descriptors are computed (whole-run vs
  epoch-1 vs first-N-draw prefixes), and
- whether multiplicity-balance descriptors replace binary-coverage ones.

It may **not** change the hypotheses, the targets, the regression form, or the
falsifier below. If whole-run coverage is saturated, that is reported as a
**result** (H1), not quietly dropped.

## 4. Hypotheses (pre-registered)

- **H1 (degeneracy).** At production settings, whole-run subset coverage is
  saturated: every coverage descriptor's between-seed relative spread within an
  arm is < 1 %. *If true, whole-run coverage cannot mechanically explain a
  seed-to-seed CID22 spread of order 0.01–0.05 SROCC, regardless of any
  correlation observed.*
- **H2 (early-window).** Coverage/balance measured over the first epoch (50,000
  draws, ≈14 % row coverage) carries real between-seed variance and predicts
  held-out score with |ρ| > 0.3 and consistent sign across ≥2 model classes.
- **H3 (balance).** Among balance descriptors (per-group share vs declared
  weight; entropy and CV of per-row multiplicity; near-threshold band share),
  at least one predicts held-out score with |ρ| > 0.3 and consistent sign across
  ≥2 model classes.
- **H4 (decomposition, needs §6 split seeds).** The seed-to-seed spread in
  held-out score decomposes into an init component and a sampling component. The
  pre-registered directional guess is **init ≫ sampling**.
- **H5 (lucky seeds).** Seeds that are outliers-high within their arm
  (`A4b_s4004`; the `a4bkon` K-arm best seeds) do **not** have descriptor vectors
  distinguishable from their siblings' (i.e. luck is not coverage).

## 5. Descriptors, targets, regression, falsifier

**Descriptors**, per (model class, arm, seed), computed on the whole run AND on
the epoch-1 prefix:

| # | descriptor |
|---|---|
| D1 | distinct rows touched / total train rows (per group, and pooled) |
| D2 | distinct refs touched / total refs (groups with `ref_ids`) |
| D3 | distinct (ref, quality-band) cells touched / total cells |
| D4 | per-group pair share vs its declared normalised `train_w` (L1 deviation, and χ²) |
| D5 | Shannon entropy of the per-row multiplicity distribution, normalised by log n |
| D6 | CV of per-row multiplicity (pooled and per group) |
| D7 | near-threshold band share — fraction of drawn pairs whose max endpoint score ≥ 90 (the KonJND-relevant zone) |
| D8 | within-image vs cross-image pair ratio |
| D9 | duplicate-pair rate — fraction of draws repeating an already-drawn (g, ia, ib) |
| D10 | identical-index skip rate (`ia == ib`) — a pure RNG-luck control that *should* carry no signal |

**Targets** — genuine holdouts only: `rank.cid22.srocc_per_ref_mean` (within-image
CID22, the product-relevant form), `rank.cid22.srocc_signed`, `rank.konjnd`,
`rank.aic3`. **KADID and TID are excluded** — they are train==val on these arms
and would reward memorisation, not skill. CID22 MOS is never a training target
anywhere in this study.

**Regression.** Seed is the unit. Within arm: Spearman ρ of each descriptor
against each target (owner: `zenstats` via `zensim_validate::panel` / the `panel`
binary — no hand-rolled stats anywhere). Pooled across arms: rank-transform both
descriptor and target **within arm** (removing arm fixed effects by construction,
which is the only defensible pooling at k=3), then Spearman on the pooled
residual ranks. Report per-class as well as pooled, plus effect size in target
units per descriptor unit from a within-arm OLS on the raw values.

**FALSIFIER (pre-registered).** *If no descriptor predicts held-out score across
≥2 model classes with |ρ| > 0.3 and consistent sign, subset coverage is not the
driver of seed spread.* On that outcome the study reports the null, lands the
registry and the stratified sampler, and does **not** ship coverage-targeted
steering (§7c is skipped, by pre-registration, not by choice after the fact).

## 6. Split seeds — coordinator amendment

`--seed` currently sets both streams. The `claude-ownerfix` lane is splitting it
into `--init-seed` + `--sample-seed` (backward-compatible: bare `--seed N` stays
byte-identical). Consequences adopted here:

1. **Every existing bake's single seed drove BOTH** init and sampling. Phase-2
   findings are therefore attributed **jointly**; no Phase-2 result can separate
   the two components, and none will be claimed to.
2. **The Phase-3 pilot uses the split flags** to decompose: replay the
   best-covered subset under 3 init seeds (sampling fixed) AND hold init fixed
   under 3 sample seeds. Spread within the first triple is the **init component**;
   within the second, the **sampling component**. This is the study's cleanest
   deliverable and the direct test of H4.
3. **The registry keys on the SAMPLE seed / sample-sequence digest**, with the
   init seed recorded as a separate field.

## 7. Deliverables

- (a) `benchmarks/good_subsets_registry.json` — append-only, schema header; per
  (model class, arm, sample seed) the sample-sequence digest + descriptor vector
  + held-out scores + init seed. Small by construction; the full descriptor
  tables live under `/mnt/v/output/zensim/subset-study-2026-09-04/` with a
  committed pointer.
- (b) A prescription: the descriptor profile of the top-quartile subsets per
  class, as a target coverage spec — **or the explicit null**.
- (c) `--pair-sampling stratified`, and coverage-targeted steering **only if the
  §5 falsifier does not fire**.
- (d) The pilot: fastclass `C0` recipe, seed-split triples per §6.2.

## 8. Owner discipline

The re-simulation must not become a second sampler. The draw step is extracted
into ONE owner used by the training loops **and** the simulator; a bake
byte-identity gate plus a sample-sequence digest prove the extraction changed
nothing. Any descriptor stat goes through `zenstats`. Default paths stay
byte-identical.

---

# RESULTS

**The study falsified its own hypothesis.** Subset coverage does not explain
seed-to-seed spread in held-out score, and the evidence is not merely "we found
nothing" — a pre-registered pure-luck control descriptor, which *cannot*
causally matter, correlates with held-out score **more strongly than any real
coverage descriptor does**. That is what makes the null a finding rather than an
absence.

## Phase 1 — the reconstruction, and the proof it is faithful

`mlp_train::sampling` is now THE owner of the pair-draw step: the four training
loops and the replay all call `sampling::draw_pair`, so the study's subsets are
the trainer's subsets by construction, not by a re-implementation kept in sync.
`subset_sim` reads a bake's embedded `zentrain.repro` and replays its sampler
with **no feature columns read** (`parquet_loader::load_scores_and_refs`) and no
model built.

Three independent gates, all MEASURED, on three sampling paths (uniform /
within-ref / high-q-boost):

| gate | result |
|---|---|
| per-epoch training trajectory (loss + per-group SROCC/PLCC/PWRC, full printed precision), pre-extraction binary vs post | IDENTICAL, 3/3 |
| bake bytes, `zentrain.repro` stripped (its timestamp/argv/cwd make whole-file identity impossible by construction) | IDENTICAL, 3/3 |
| replay digest vs a REAL run's `ZENSIM_SAMPLE_DIGEST=1` | MATCH, 3/3 — `03eb9f61c99f76da` / `fc62d239b67a2e62` / `ff243de6fb5517a4` |

Reproduce: `scripts/subset_study/byte_identity_gate.sh` (binary paths via
`OLD_TRAIN`/`NEW_TRAIN`, so it cannot fossilise onto a cleaned-up worktree).

Two further proofs at **production scale**, from the Phase-3 pilot rather than a
fixture (8 train groups, 465,587 rows, 120 × 50,000 = 6,000,000 draws):

- **The replay reproduces a real production run's draw sequence exactly.**
  `subset_sim --fulleval FC_C0_s4004.fulleval.json --seeds 4004` returns digest
  `647a088f42232bab`, matching what the live training run printed under
  `ZENSIM_SAMPLE_DIGEST=1`.
- **Cross-build reproduction.** The pilot's origin cell
  (`--init-seed 4004 --sample-seed 4004`) is **byte-identical** — repro block
  stripped, sha `a29b610fa16e251d…` — to the shipped `C0_s4004.bin`, which was
  built by a *different binary, at a different commit, in a different worktree*
  (`c993f45f`, the wave-r4 tree). That simultaneously shows the split seeds
  collapse to the old `--seed` behaviour, that none of this lane's trainer
  changes moved the production recipe, and that the fastclass recipe is
  reproducible from its embedded repro alone.

Population actually reconstructed: **202 (arm, seed) cells across 66 seed-sibling
arms**, from 312 board fullevals carrying a usable repro. The 17 remaining
candidate arms all failed for **one** reason — every one of them trained on
`ext_safesyn_full.parquet`, which no longer exists on disk — so they are absent,
not silently substituted with a different corpus. That is a real limitation of
the replay approach and worth stating: it reconstructs a subset from the recipe,
so it inherits the recipe's dependency on the corpus files still being there.

## Phase 2 — H1: whole-run coverage is saturated

At production settings (120 × 50,000 = 6,000,000 draws over ~700,000 training
rows, ≈17 hits per row) coverage does not vary between seeds in any meaningful
sense. Between-seed **relative** spread within an arm, over 66 arms:

| descriptor | median relspread | max relspread |
|---|---:|---:|
| pooled distinct-row coverage | 1.37e-4 | 4.58e-4 |
| reference coverage | 1.31e-5 | 8.47e-5 |
| row-multiplicity entropy | 3.85e-6 | 4.76e-5 |
| (ref, band) cell coverage | 8.82e-5 | 2.51e-4 |
| ≥90-endpoint share | 7.74e-4 | 3.85e-3 |

**12 of 29 descriptors never exceed 1 % relative spread on any arm.** Against
that, the quantity to be explained moves far more: within-arm relative spread is
**3.9e-3** (median) for within-image CID22, **1.0e-2** for CID22, **1.3e-2** for
AIC-3, **1.3e-1** for KonJND. Row coverage varies ~28× less than the CID22
target it would have to drive, and ~940× less than KonJND. **H1 CONFIRMED** —
whole-run coverage is mechanically incapable of being the driver, independent of
any correlation one might compute.

## Phase 2 — the falsifier, and why the control is the point

The pre-registered class axis (model input width) turned out **degenerate on this
board: 65 of 66 sibling arms are 944-input**, so "replicates across ≥2 model
classes" cannot be evaluated as written. Reported as a limitation. The
substitute axis actually available is the **group structure** (which corpora at
which weights an arm trained on — 48 distinct structures), and on it a long list
of descriptors "replicates" at |ρ| > 0.3.

That list is noise, and the control proves it:

| | max \|ρ\| | median \|ρ\| |
|---|---:|---:|
| **pure-luck controls** (`same-row skip rate`, `duplicate-pair rate`) | **0.3707** | 0.0985 |
| real coverage descriptors | 0.3380 | 0.1002 |

`e_D10_samerow_rate` — how often the RNG happened to draw the same row index
twice, a property of nothing but RNG luck, carrying no information about the
training data — is the **single strongest correlate of within-image CID22 on the
entire board** (ρ = +0.3707, n = 119), and its own permutation null (R = 500)
calls it "significant" at p = 0.002. `D10_samerow_rate` vs KonJND likewise:
ρ = −0.3007, p = 0.010. The control also "replicates across ≥2 group structures".

So the machinery manufactures |ρ| ≈ 0.3 with p < 0.01 for a descriptor that
cannot matter. The mechanism is repeated measures, and the registry measures it
exactly: the 202 cells contain only **132 distinct subsets over 51 distinct
seeds**, and **88 cells share their subset with at least one other cell** — one
subset (`bb1ac16f…`) is the training subset of **22 different cells**, another
of 19, another of 13. Arms that share a group structure and a seed draw *the
identical pairs in the identical order*, so a seed-pure descriptor takes one
value across all of them, and permuting *within* arm leaves that cross-arm
dependence intact. The pooled tests report n = 94–132 while the effective
independent count is far smaller; the null is under-dispersed and every p-value
in that table is optimistic. **The correct reading is that no descriptor clears
its own control.** The pre-registered falsifier fires:

> *no descriptor predicts held-out score across ≥2 model classes with |ρ| > 0.3
> and consistent sign* — and by the stronger, control-based standard, no
> descriptor predicts it at all.

Per §5, §7(c)'s coverage-targeted steering is therefore **not shipped**. That was
decided in advance, not after seeing the numbers.

## Phase 2 — H5: the lucky seeds are not better covered

Taking each arm's best seed by within-image CID22 and comparing its descriptors
against its siblings' mean, in units of the arm's own descriptor SD: the largest
mean z over 31 arms is **0.467**, and the pure-luck control sits third at 0.428.
A real coverage advantage would sit near ±1. **H5 CONFIRMED (luck is not
coverage).**

## Phase 2 — seeds are not transferably good

A separate question the registry makes cheap: is a "good seed" a property of the
seed? Ranking seeds within each arm and testing whether per-seed mean rank is
more consistent than chance (2,000 permutations, 31 arms, 8 seeds with ≥4
appearances):

| target | observed sd of per-seed mean rank | null p95 | emp p |
|---|---:|---:|---:|
| CID22 within-image | 0.1570 | 0.1940 | 0.205 |
| CID22 | 0.2051 | 0.1940 | **0.035** |
| KonJND | 0.1258 | 0.1940 | 0.446 |
| AIC-3 | 0.1111 | 0.1940 | 0.587 |

One of four targets is nominally significant, which is what four tests produce by
chance, and the orderings disagree across targets (seed 4006 is 2nd best on CID22
and 3rd worst on AIC-3; 4003 is near-worst on CID22 and best on AIC-3). **There
is no transferable "good seed."** Consequence for the deliverable: a registry of
good seeds cannot be an oracle, and the registry says so in its own schema
header.

## Phase 3(c) — the stratified sampler already existed, and was unreachable

The brief asked for `--pair-sampling stratified` if the concurrent ownerfix lane
had not landed it. It did not need building: **`--stratified-bands N` has existed
since STRATEGY-2026-07-02** — quantile bands per group, row A drawn band-uniform
then row-uniform inside the band. Per the one-owner rule, a second stratified
sampler was not written.

It was, however, **a silent no-op on the path every model actually trains
through.** `strat_bands` was built in exactly one of the four training loops
(`train_mlp_per_sample_alpha_head`); the other three, including
`train_mlp_strategy`'s standard path that all 312 board bakes used, passed an
empty band table. MEASURED on the default path before the fix:
`--stratified-bands 0` and `--stratified-bands 8` produced the **same
sample-sequence digest, `127b831bed8a3873`** — the flag was accepted and
discarded. Fixed (`fix(train): --stratified-bands was a SILENT NO-OP…`), with a
failing-first test and the default proved byte-identical; after the fix the two
digests are `127b831bed8a3873` vs `3ea736b63181a351`.

Worth stating plainly, because it is the kind of defect this study was well
placed to find: **a knob that accepts a value and discards it is worse than a
missing knob**, because a sweep over it returns a clean table of identical
results that reads as a finding. No board bake sets it, so nothing published is
affected.

**Not claimed:** that stratified sampling helps. It is now reachable and
testable; whether it improves anything is unmeasured, and this study's own result
is a reason for caution rather than optimism.

**The same defect family, four more members.** `--ema-decay`, `--hard-pair-frac`,
`--dro-eta` and `--listwise-weight` are all read only inside
`train_mlp_per_sample_alpha_head`. `train_mlp_strategy` already guards three
sibling flags against exactly this (`--monotonicity-reg`, `--mse-weight`,
`--triplet-weight`, whose comments record that the earlier versions "were dead
code that could never fire" and were "found by a should_panic test that did not
panic") — these four were simply never added to the list. They now fail loud.

## Phase 3(a) — the registry

`benchmarks/good_subsets_registry.json` (append-only, schema header, 28.6 KB):
one row per (arm, seed) cell — 202 cells over 66 arms — carrying the
**sample-sequence digest** that identifies the drawn subset, the sample seed and
init seed as separate fields, a `seeds_were_fused` flag recording that every
current row predates `--init-seed`/`--sample-seed` (so its single `--seed` drove
both streams and the two effects are not separable for that row), and the
held-out scores.

The 29 coverage descriptors are deliberately **not** duplicated into it: they are
a deterministic function of the subset, so
`subset_sim --fulleval <bake>.fulleval.json --seeds <sample_seed>` regenerates
them exactly and the digest proves the right subset was regenerated. The computed
tables live in block storage behind
`benchmarks/subset_study_data_2026-09-04.pointer.md`.

The schema header states the null in its own `finding` field, so a future session
cannot pick a seed out of this file by its coverage numbers without reading that
it must not.

## Phase 3(b) — the prescription: there isn't one, and that is the result

The pre-registration committed to skipping coverage-targeted steering if the
falsifier fired. It fired. **No target coverage spec is issued**, because the
evidence says a coverage spec would be steering on noise:

1. whole-run coverage is saturated — 12 of 29 descriptors never move more than
   1 % between seeds, while the targets move 28–940× more;
2. a pure-luck control out-correlates every real descriptor;
3. the best seeds are not better covered (max mean z 0.467, control third at
   0.428);
4. good seeds do not transfer across arms.

What a practitioner should take from this: **do not tune the pair sampler hoping
to reduce seed variance at production settings.** With ~17 draws per row, which
rows a seed draws is effectively fixed; what differs between seeds is the *order*
they arrive in and the weights they start from.

The obvious remaining lever was `--ema-decay`, whose own CLI doc advertises
"seed-variance reduction". **It is not available on this recipe**: like
`--hard-pair-frac`, `--dro-eta` and `--listwise-weight`, it is read only inside
`train_mlp_per_sample_alpha_head`, so on the standard path — the one all 312
board bakes train through — it was accepted and discarded, silently. This lane
did not wire it (an EMA/DRO/listwise step through three more training loops is a
real change to the optimizer and the bake path, not a drive-by); it added the
**fail-loud guard** the file already applies to `--monotonicity-reg`,
`--mse-weight` and `--triplet-weight`, so the flag now stops the run instead of
producing a table that reads "EMA does not help" when EMA never ran. No board
bake sets any of the four, so nothing published is affected.

That leaves **seed averaging**, which the ensemble rows already do, as the only
variance lever currently reachable on this path — and wiring EMA into the
standard loop as the concrete next piece of work, now that its absence is
explicit rather than silent.
