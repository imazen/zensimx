# Feature-calculation defect audit — 2026-09-05

**The question (user, verbatim): "do we have bugs in feature calculations?"**

**The answer: yes — 27 distinct defects are on the record, 8 of them still open,
but only ONE of the open ones is a live arithmetic error in a shipped feature.
The rest are data-era consequences of defects already fixed, deliberate
accuracy/perf trades, or plumbing that decides *which* features get computed
rather than *what* they compute. The extraction engines themselves are, at
HEAD, deterministic, engine-agnostic, tier-agnostic within policy, and free of
NaN on every degenerate input tried.**

Scope: every feature the zensim extractors emit (v1 basic `f0..155`, peaks
`f156..227`, masked `f228..299`, IW `f300..371`, v2-348 `f372..719`, append
`f720..923`, append2/BANDVIS `f924..943`, the free / raw-moment and class-C
tranches), across the engines (buffered v1 walk, streaming fold, 944 fold,
attribution density), the eras, thread counts 1–28, SIMD tiers v4x and v3,
image widths tight / non-tight / odd / sub-64 / past `H_TILE_WIDTH`, and both
product entry points.

Instrument: `zensim/examples/feature_invariant_probe.rs` (new, this lane).
Gates: `zensim/tests/feature_invariants.rs` (new, 10 tests, all passing).
Artifacts + shas: `/mnt/v/output/zensim/feature-audit-2026-09-05/` with
`_MANIFEST.json`.

---

## 1. The verdict, counted

| class | n | what it means |
|---|--:|---|
| **FIXED** | 8 | a real arithmetic or coverage error, corrected in code, with a gate |
| **OPEN — data** | 4 | the code is fixed; stored tables / shipped bakes still carry the pre-fix values |
| **OPEN — live** | 1 | an arithmetic error still reachable in a shipped feature today (**F4**) |
| **OPEN — plumbing** | 3 | which features are computed/consumed, not their arithmetic |
| **BEHAVIOUR, not bug** | 11 | deliberate accuracy/perf trades and era-locked accumulation orders, each measured and each with a named reason not to change it |
| **NEW this lane** | 7 | see §4 — one API trap now gated, one long-standing claim corrected, four measurements, and the servability census (§4A) |

**The one live arithmetic defect is F4**: v1's SSIM per-pixel dissimilarity `d`
has a `.max(0)` floor and **no upper cap**, and its `num_m = 1 − (mu1−mu2)²`
term carries no `C1`, so on high-magnitude chroma `d` reaches **5.8e6**
(`f313` `iw_ssim_4th` = 5,814,302 against a photographic p99.9 of 0.48). It is
open **by decision, not by oversight** — the shipped winsor guard clamps the
symptom, and the campaign that found it recorded "not panel-justified now". The
consequence to carry: 144 features hold a clamped-outlier value on the weakest
content class, so any model fit without that guard is mis-specified there.

**Servability is the largest single gap** (§4A, added at the coordinator's
request as the architecture lane's phase-gate baseline): **400 of 433 board
bakes, 3 of 11 shipped bakes, 11 of 14 registered feature sets, and 2 of 10
selectable shipped profiles cannot be served by `Zensim::compute` at all** —
one cause (the product entry emits a 372-layout vector; anything declaring a
wider read is refused), four symptoms. `ZensimProfile::C` and `CHdr` are the
sharp end: `candidate-profiles` is default-on and C's bake ships to crates.io,
yet `Zensim::new(ZensimProfile::C).compute(...)` fails on every image — while
returning a healthy-looking **100.000000** on a ref-vs-ref smoke test, because
the identity short-circuit runs before the model.

**Non-monotone features are NOT bugs, and there are a lot of them** — see §3(f).
95 slots are persistently, amplitude-really non-monotone under a
control-validated distortion ladder. Rectified (one-sided) pairs like
`GLOBAL_CGAIN`/`GLOBAL_CLOSS` and band-limited detectors are *supposed* to rise
and fall. This is written down here because anyone fitting a monotone head, a
dial, or a per-slot sign constraint needs the list and has not had it.

---

## 2. Inventory

`ids` = feature slots. `status`: FIXED / OPEN / ERA-BREAK (fixed in code, an
era boundary in the data) / BEHAVIOUR (measured, deliberately unchanged).

### 2.1 Numeric defects — the arithmetic of a feature was wrong

| # | name | ids | mechanism | blast radius (measured) | status | gate |
|---|---|---|---|---|---|---|
| F1 | masked/IW activity map read never-written strip-overlap rows ⇒ **thread-count-dependent features** | `f228..371` | the activity map read `bufs.mu1` at strip-overlap rows the fused V-blur never writes; the band layout was `rayon::current_num_threads()`-derived, so the thread count chose *where* those rows fell | CID22 4,292 pairs, stored-vs-HEAD: masked 288,418 cells over tol (max abs 0.0374), IW 294,081 (max abs 0.1235), **100 % of rows**. `RAYON_NUM_THREADS` 1/2/8/28 gave **four different md5s**. Shipped **B** puts 23 of 95 live inputs there incl. its largest weight (`f353`, norm 182.4); runtime-vs-eval CID22 SROCC 0.87638→0.88212, KonJND 0.54665→0.64967, dial mean **−4.98** pts | **FIXED** code (`2dab8f30`, `6af83b60`); **OPEN data** — B was fit AND calibrated pre-fix and serves post-fix | `v1_372_is_bit_identical_across_rayon_pool_sizes`, `v1_masked_and_iw_blocks_are_thread_invariant` |
| F2 | pyramid walk emitted **93/186/279-wide** vectors, silently | whole vector | three entries did not reflect-pad; the scale walk starts at `simd_padded_width(W)` but plain `H`, so 4 scales need `simd_padded_width(W) ≥ 64 AND H ≥ 64`. One (`compute_zensim_with_config`) returned a short vector; two PANICKED | **1,368 of 20,812** R1b rows short (6.5 %, size-correlated). Zero canonical corpus pairs could fire it; the 944 fold immune | **FIXED** `f9fac41e` (one owner: `needs_pyramid_pad`) | `v1_feature_width_pure_function.rs` (8 tests, 5 red pre-fix); +`sub64_inputs_still_emit_a_full_width_vector_on_both_entries` (this lane) |
| F3 | buffered v1 pooled **phantom mirror-padded columns** | all 372 v1 slots at every non-tight width | v1 walked `simd_padded_width(width)` and pooled the mirror fill; the fold walked `width` | up to **81.6 % relative** on a pool slot; 512/576/768/1024/1152/2304 all divergent ⇒ ~60 % of real corpus rows. Fixing it is also **cheaper** (Ir −9.02 % @576) | **FIXED / ERA-BREAK** `56bbcda2` | `pyramid_stride_has_no_phantom_columns` (24 widths), `v1_372_bit_exact_to_fold_at_every_width` |
| F3b | **consequence: the runtime is one extraction era AHEAD of every 372 eval root** | all 372 | the default root was built at `ea16c7ee` 13:21; option C flipped at `56bbcda2` 15:43 the same day | CSIQ re-extracted at HEAD with the same tool on the same input: basic **120,804/135,096** cells differ, max abs **4.536785**; every row differs on 285–341 of 372 slots. 944 roots NOT affected | **OPEN / ERA-BREAK-REGISTERED** — re-extraction registered, not run | `v1_golden_bytes` passes 5/5 **and is blind** (every fixture tight or sub-tile) |
| F4 | **unbounded SSIM per-pixel `d`** | the `ssim_*` family in `f228..299` + `f300..371` | `num_m = 1 − (mu1−mu2)²` has **no C1**; `d` has a `.max(0)` floor and **no upper cap**; L4 pooling amplifies | `f313` = **5,814,302**, `f241` = 5,797,029 on 2.3 M scanned rows, against photographic p99.9 = **0.48**. Denominator-cancellation hypothesis tested and FALSIFIED (worst effect 1.2×) | **OPEN — the one live arithmetic defect.** Not fixed by decision ("winsor already handles it") | instrument only (`dump_ssim_moment_explosion`, `#[ignore]`) |
| F5 | **free-40 raw-moment route parity fails on real pixels** (train/serve skew) | the 37 raw-moment slots; misses concentrate in `GLOBAL_CLOSS` / `GLOBAL_CGAIN` | `global_stats_from_raw_moments` uses `Σs²/n − (Σs/n)²`, a catastrophic-cancellation form; the append kernel reduces per ROW, the fused kernel per BAND | 773 real pairs: **2,607 of 28,601 cells (9.12 %) over the 2e-5 bar, worst 3.63e-3**, worst relative ~55×. class-C 24 slots: **0/18,552**. basic+peaks bit-identical | **OPEN — reported, not fixed** | the gate that MISSED it (`free_extras_match_the_944_append_block`) is synthetic-image-only |
| F6 | `RawMoments` emission gate written as `==` | all 40 raw-moment slots | a superset request (`RawMomentsPlusBoundedErr`) zeroed all 40 | "a model reading those slots would have been served zeros with nothing failing" | **FIXED** | `class_c_extras_are_pure_addition_to_the_free_walk`; +`class_c_request_still_carries_the_raw_moments` (this lane) |
| F7 | attribution density **dropped the whole append2/BANDVIS block** | `f924..943`; 8 real dropped (`BANDVIS_GAIN/LOSS` × 4 scales) | sliced `s[720..min(len,924)]` | M3a over 32 bakes: median **+0.0487**, max +0.1045, 19/32 change tier, GOLD count **2 → 16**; flipped a `freeze_check --select` winner | **FIXED** `299ccc8c` | `attribution_covers_expected_slots_per_width` (probes every width) |
| F8 | dial grid masked/IW were **GPU odd-dim garbage** | `f228..371` on 9 of 115 ladders | a `zensim-gpu` odd-dim pyramid pathology emitting non-NaN garbage, bit-constant across each ladder | a webp knob-blocker was inferred **that never existed** (quarantined p10 83.7 vs 9.4) | **FIXED by quarantine** (`ae4209a8`) | grid quarantine |
| F9 | **era-1 horizontal reduce is backend/tier-dependent** | dense/v2-era `f372..719` + 1 in `f720+` | `reduce_add` pairs lanes differently per backend; `(l0+l4)+(l1+l5)` ≠ `(l0+l1)+(l2+l3)` | same binary two boxes: era-1 diverges on **66 of 105** dense slots; **era-2 is 0 of 105** | **FIXED by era-2**, default-on 2026-08-31 / ERA-BREAK | `era2_band_merge_and_tail_are_structural` |
| F10 | column tiling applied at **call sites, not entries** | any H-blur-derived slot | phase A's banded H call tiled while the v1 reference path did not | 4 cross-path gates red at `ZENSIM_H_TILE=0032`; **zero goldens re-pinned** at the shipped width | **FIXED** (every H entry tiles or none does) / ERA-BREAK (`H_TILE_WIDTH` is semantics) | `fold_engine_parity` `CELLS` gained `(1153,72)`, `(2049,40)` |
| F11 | `2*(width−1) − add_raw` mirror index **underflowed** on a degenerate last tile | none in release | hand-copied into 38 kernel bodies; underflows when the last slab is exactly `radius+1` wide (`width % 1024 == 1`) | **12 red tests**; release `to_bits` dumps **byte-identical pre/post (12/12 files)** — no score, verdict or board row moved | **FIXED** `f22ade56` (one owner `blur::h_mirror_add_idx`) | `h_entries_are_bit_exact_at_a_degenerate_last_column_tile` |
| F12 | `fused_blur_h_mu_inner_*` scalar tail vs vector body differ by ulps | mu-derived slots on the last `height % 8` rows — **the production band shape hits this** | tail does `sum + (add − rem)`, body does `(sum + add) − rem` | MEASURED 2528.7349 vs 2528.7344 | **OPEN — deliberately not fixed**; converting moves v1's shipped bytes | recorded in the reference test's doc |
| F13 | `dense_block_kernel` **per-pixel accumulation-order sensitivity** | dense slots | bit-exact merging needs `POOL_SIMD` AND `width % 8 == 0` at every scale; neither holds generally | 0 ulps / **−2** with a scalar tail / **13** with per-pixel pools. Amdahl bound on fixing: 1.17×@8T against re-extracting and re-training every 944 table | **BEHAVIOUR — era-locked, do not restructure without asking** | — |
| F14 | `ERA2_BAND_ROWS` banding is a different grouping from the serial fold | era-2 dense accumulators | banding IS a different grouping | the first "band-merge == serial" gate FAILED at 127×93 by design | **BEHAVIOUR** — declared SEMANTICS, not a tuning knob | `era2_band_merge_and_tail_are_structural` |
| F15 | `PJND_FRAGILITY` reads a **constant 1.0** on a v1-only 944 walk | `f393 422 451 480 509 538 567 596 625 654 683 712` | `finish_channel_scale` produces `1.0` from zeroed accumulators — a formula artifact | **773/773 rows**; a model built from "whatever columns are non-zero" gets twelve constant columns that do not exist in the stored tables | **OPEN**, mitigated by explicit slicing | now gated: `identity_nonzero_slots_are_reference_only_pjnd_or_fp_residue` (this lane) |
| F16 | **decoder-era drift** in every stored-pixel corpus | all slots, via the pixels | the `q<X>.png` decode cache was deleted; re-decoding surviving bitstreams gives different pixels than the generation-era decoder | 360 rows: `|Δ| > 1.0` on **14 cells (0.025 %)**, all XYB-JPEG / JXL. **Dial: shipped B mean −3.658 pts** = **73 %** of the −4.98 extractor-era defect, same sign | **OPEN, unfixable in principle** — pick a deliberate decoder era and record it per format | companion FIXED: `extract_features_372col` now decodes via `zen_decode.rs`; gate `zen_decode_formats.rs` |

### 2.2 Plumbing defects — which features get computed or consumed

These change numbers without any arithmetic being wrong, which is what makes
them dangerous: they produce structural zeros or a wrong-width read, and
nothing errors.

| # | name | status | note |
|---|---|---|---|
| P1 | **`free_extras` hard-coded OFF on the product path** — `compute_folded_v1_372_streaming_impl` builds its toggles with `..Default::default()` and `V1FreeExtras::default() == Off` | OPEN | every raw-moment and class-C slot reaches the forward pass as a structural `0.0` on any `Zensim::compute` |
| P2 | **372 truncation** — `compute_fold_backed` truncates to ≤372; the free slots live at `f720+` | OPEN | past the cut even if they had been computed |
| P3 | **wide-bake dead code** — `wide_bake_v2_read`, the function that *would* pick the needed `V1FreeExtras`, is `allow(dead_code)` and reached only from tests | OPEN | consequence: a 944-declared bake is unservable through `Zensim::compute` (measured: two bakes REFUSED with `ModelForwardFailed`) |
| P4 | **`V1FreeExtras` is silently inert unless `append_block` is also declared** | **NEW, now gated** | see §4.1 |
| P5 | **`Zensim::compute` short-circuits byte-identical input BEFORE the model**, fabricating `(100, 0, zeros)` | **NEW framing**, see §4.2 | the 372 "identity is the zero vector" property has never been measured |
| P6 | `--regime 944` silently mis-scores a 372 bake that uses `f156..371` (shipped B: CID22 **0.3862** vs its true **0.8764**) | OPEN, mechanised by feature-set ids | |
| P7 | `f156..371` is zeroed at some 944 roots and LIVE at others; a peaks-carrying slice needs a `…pools` root | OPEN, mechanised by feature-set ids | |
| P8 | `--extract-only --format tid2013` **silently drops 120 of 3,000 TID pairs** on decode failure, printed only as a count | OPEN | |
| P9 | `num_scales != 4` diverges; the fold is hard-wired to `NUM_SCALES` and falls back rather than scoring a different pyramid | BEHAVIOUR (correct fallback) | |
| P10 | two `self_blur` predicates disagreed (sizing vs the strip loop) — planes allocated, first-touched, never written, on the one mode the fast class ships | FIXED | `self_blur_sizing_predicate_matches_the_strip_loop_predicate` |
| P11 | `feature-regime-v2` was not a default feature, so Profile D's fast wiring was a **no-op on a plain `cargo add zensim`** | FIXED 2026-09-01 | |

---

## 3. Invariant probe results

All probes run at `4fbd8ff8` on the dev box (Zen 4, native tier v4x unless
capped). Correctness only — a trainer was live throughout, so no timing claim
is made or implied.

| probe | scope | result |
|---|---|---|
| **(a) identity** | ref == dist, 4 routes × 11 geometries | **The 944 identity vector is NOT zero: 286 of 944 slots**, independently reproducing DATASET_HISTORY §3.36's count on a different (synthetic) population. Resolved into exactly 3 classes — see §3.1 |
| **(b) determinism** | 5 repeats × 12 geometries; rayon pools 1 and **28** | **BIT-IDENTICAL, 12/12 and 4/4.** (1/2/3/8/16 were already gated; 28 was not) |
| **(c) engine parity** | buffered v1-372 ↔ fold v1-only ↔ fold944-full ↔ product buffered ↔ product fold, 11 geometries | **BIT-EXACT, 33 of 33 comparisons.** Also bit-exact with dispatch capped to v3 |
| **(d) tier parity** | v4x (AVX-512) vs v3 (AVX2+FMA), full 944 vectors, 22,397 cells | 2,982 v1 cells differ, **max abs 3.48e-8, max rel 8.75e-6**, and **0 cells over the golden tolerance policy** `max(1e-6 abs, 1e-5·scale)`. **The v2-era block `f372..943` is BIT-EXACT across tiers** — the era-2 reduce fix (F9) holding |
| **(e) width independence** | tight vs non-tight vs odd vs past-`H_TILE_WIDTH`, incl. the 1-column remainder tile | no width-class effect anywhere; the (c) and (b) results hold identically at 512/576 (the old padded class) and at 1153/2049 |
| **(f) monotonicity** | 944 slots × 3 ladders × 12 images, **with an MSE stimulus control** | **95 persistent amplitude-real non-monotone slots** — see §3.2. One ladder had to be DISCARDED because its own control failed |
| **(g) scale consistency** | scale *k* on the original vs scale *k−1* on a 2× box-downscale, 234 cells | max relative disagreement **0.203**, none ≥ 0.5, 26/234 under 0.01. No gross violation; the residual is the expected mismatch between a box downscale and zensim's own pyramid filter |
| **(h) degenerate inputs** | flat / off-by-one / all-black-vs-white / single lit pixel / saturated opposing channels, × 4 geometries × 3 routes | **0 NaN, 0 Inf, 0 non-finite scores** in 72 measured vectors |
| **(i) input depth** | u8 vs u16-replicate vs linear-f32 of the same content | 348–355 of 372 slots differ, **max abs 1.98e-4** — a route-precision skew of the same order as F5's, worth knowing before mixing depths in one table |

### 3.1 The identity census, classified

On an identical pair the 944 walk populates 286 slots. Every one falls into
exactly three classes, and the classification is now a gate:

| class | slots | max \|v\| | verdict |
|---|--:|--:|---|
| **reference-only** — `GRAD_SRC_MEAN` (append local 16, 11 cells) + `LUMA_MEAN_REF` (append2 local 2, 4 cells) | 15 | 0.890 | **CORRECT.** `∂f/∂dist ≡ 0`, so a non-zero value on `ref == dist` is what these features mean |
| **`PJND_FRAGILITY`** (v2 local 21) | 12 | 0.395 | **F15 concerns the uncomputed v1-only value.** Reads exactly **1.0** on a `v1_only` walk and **0.395** on the full walk. **Corrected 2026-09-25; contradictory wording removed 2026-10-04:** the full-walk identity value is not a defect. The slot is reference-only (`1 − saturate(mean grad_src_mag)`), and on the Rev4 bank's 88 identical keys it equals every same-reference sibling bit for bit (REVIEW_PARTB). Only the v1-only 1.0 is a defect. |
| **floating-point residue** | 259 | **1.12e-3** | acceptable. v1 blocks ≤ 1.12e-3, v2/append ≤ 2.4e-4 |

The bar the new gate enforces is the third row: any *new* non-zero identity
slot above 2e-3 that is not a registered reference-only slot means a difference
feature stopped cancelling at zero distortion.

### 3.2 Monotonicity — and the stimulus control that saved the finding

A first pass over three ladders reported 176 / 127 / 109 amplitude-real
non-monotone slots. Then the control was added: **plain MSE(ref, rung)**,
emitted as a pseudo-slot through the same violation counter.

| ladder | MSE control | usable? |
|---|---|---|
| box blur, 1..6 passes of radius 1 | **non-monotone on 12 of 12 images** — `29.13 → 26.02 → 29.11 → 31.62 → 34.53 → 37.66` | **NO — discarded** |
| additive noise, amplitude 4..48 | monotone 12/12 | yes |
| quantization, step 4..64 | monotone 12/12 | yes |

The blur ladder dips at rung 2 and so does every feature that "violated" on it:
`f372`, `f156`, `f300`, `f384` all read `step1 > step2 < step3 < …` on 12/12
images. **The stimulus was the defect, not the features.** Repeated radius-1
box blur is not a monotone degradation ladder — a single 3×3 box leaves more
error energy against the original than the smoother two-pass kernel does.

**Do not use repeated box blur as a monotonicity stimulus, and do not accept a
monotonicity result that has no control.** A metric-shaped conclusion drawn
from an uncontrolled ladder is unfalsifiable — this one would have shipped 176
false violations.

On the two ladders whose control passes:

| ladder | live slots | strictly monotone on all 12 | any violation | amplitude-real | **persistent** (≥9/12 images) |
|---|--:|--:|--:|--:|--:|
| noise 4..48 | 864 | 681 | 182 | 127 | **40** |
| quantize 4..64 | 871 | 629 | 237 | 109 | **55** |

Persistent violators by block — noise: v2 15, append 12, basic 11, append2 2;
quantize: v2 24, append 17, basic 12, append2 2.

**These are mostly not bugs.** 62 of the violating series contain an exact
`0.0` alongside non-zero values, which is the signature of a **rectified
(one-sided) feature**: `GLOBAL_CGAIN` and `GLOBAL_CLOSS` are a gain/loss pair,
and heavy quantization with a `+step/2` reconstruction offset *increases*
contrast, so contrast-LOSS correctly collapses to zero at the top rung
(measured on `f837`/`f854`/`f871`, all `GLOBAL_CLOSS`: `… 9.74e-3 → 0.0`).
Band-limited detectors behave the same way by construction.

**What this means for anyone fitting a monotone head, a dial, or a per-slot
sign constraint:** roughly 5–6 % of live slots will fight a monotonicity
penalty because they are not supposed to be monotone, and the population is
concentrated in the v2 and append blocks. The list is in
`/mnt/v/output/zensim/feature-audit-2026-09-05/ladder3.tsv`, with each slot's
raw series, so a recipe can exclude them by name rather than by guess. This is
the same family as the already-registered per-slot findings (`f162`'s local
bump, `f161`'s sign disagreement, `f93`'s share of D's jpeg-floor inversions) —
those are individual instances of a property the whole feature set has.

---

## 4. New this lane

### 4.1 `V1FreeExtras` is silently inert unless `append_block` is declared — NEW, now gated

`append_block` does double duty: it declares the **layout** (720 → 924, and with
`append2_block` → 944) *and* it enables the append **compute**. Every
raw-moment slot lives at `f720+`. So a `v1_only` walk that asks for
`V1FreeExtras::RawMoments` **without** setting `append_block` gets a 720-wide
vector in which those slots do not exist — no error, no warning, and a
populated-slot count **identical to `V1FreeExtras::Off`**.

MEASURED (`feature_invariant_probe widths`, 200×150):

| v1_only | append | append2 | free | width | non-zero above f720 | non-zero total |
|---|---|---|---|--:|--:|--:|
| true | false | false | Off | 720 | 0 | 228 |
| true | false | false | **RawMoments** | 720 | **0** | **228** ← inert |
| true | true | false | RawMoments | 924 | 22 | 250 |
| true | true | true | RawMoments | 944 | 26 | 254 |
| true | true | true | PlusBoundedErr | 944 | 38 | 278 |
| false | true | true | *(any)* | 944 | 170 | 878 ← append kernel owns them, free writes nothing (as documented) |

The documented half — "with the append block on, the append kernel owns
`GLOBAL_*` and this writes nothing there" — is confirmed by the last row. The
undocumented half is the second row, and it is the same failure shape as the
`==`-vs-`!=` emission-gate defect (F6) reached by a different route: a training
table full of structural zeros, produced without error. This test file was
itself written wrong on its first pass for exactly this reason.

Gated by `free_extras_are_silently_inert_without_the_append_block_declaration`,
which pins both directions so the trap is a contract rather than a landmine.
The class-C tranche is only partly affected: its twelve v2-348 `MSE` cells live
at `f372..719`, inside the 720 layout, so they survive (228 → 240); its twelve
`LUM_*_ERR` append cells do not.

### 4.2 The 372 identity vector is FABRICATED and has never been measured — NEW framing

Both product-facing SDR entries short-circuit `source == distorted` before any
walk and synthesise the answer:

* `metric.rs::identical_result` — behind every `Zensim::compute*`
* `compute_zensim_with_config` — behind **both** v1-372 extractors

Each returns `(score = 100.0, raw_distance = 0, vec![0.0; width])` and flags it
`mark_identical()` so the model forward pass is skipped entirely.

So the property `zensim-validate`'s `dial_addressability` states as measured —
*"ref == dist yields all-zero features for every image; identity dial =
dial(0-vector)"* — is at 372 **a property of the short-circuit, not of the
extractor**. It is unfalsifiable by construction at that width. DATASET_HISTORY
§3.36 already recorded that the constant "is wrong at this width [944] and is
emitted verbatim into every `--gaddr-json`"; this lane adds *why* the 372 half
looked true.

MEASURED, same code, same pixels, computed rather than fabricated: the v1 block
on an identical pair populates **144 of 372 slots** (basic 36, peaks 24, masked
36, IW 36), max \|v\| **1.12e-3**. Small — but not zero, and the runtime and the
944 training tables disagree about it.

Gated by
`identity_is_fabricated_by_the_short_circuit_and_differs_from_the_computed_vector`,
which asserts both halves and will tell a future reader to delete the
fabrication if the computed vector ever becomes genuinely all-zero.

### 4.3 The identity score cliff — MEASURED, handed to the G-ADDR lane

Because identity is fabricated, the shipped dial has a step at zero distortion.
Perturbing **one byte in one channel of one pixel** out of 90,000:

| geometry | `Zensim::compute` on `(r, r)` | on `(r, r ± 1 LSB at one pixel)` | step |
|---|--:|--:|--:|
| 200×150 | **100.000000000** | **96.229616318** | **3.770** |
| 512×384 | **100.000000000** | **96.186393137** | **3.814** |

The feature vector moves by ≤ 2.6e-3 (all in the max-pooled peaks block, which
is *supposed* to respond to a single pixel). The **score** moves 3.77 points.

This is a MODEL property of shipped B, not a feature-calculation bug, and it is
not asserted in the gate file for that reason. It is recorded here because it
is the mechanism behind an already-registered G-ADDR finding: *"shipped B ranks
266 of 4,424 dial-grid cells (6.01 %) ABOVE a perfect copy"* and the resulting
`C2 ⊻ C6` either/or. Those 266 cells are not anomalous — the model's response
to *almost* no distortion is ~96.2, so anything scoring 96.3–100 legitimately
out-scores it, and only the fabricated 100 hides that. Registered as
`identity-score-cliff-fabricated-2026-09-05`.

### 4.4 CORRECTION — the v1-372 basic block and the 944 fold's basic block agree BIT-EXACTLY at HEAD

`docs/FEATURE_SET_IDS.md` §1 failure #9 reads: *"The v1-372 `f0..155` is NOT the
944 fold's `f0..155`. Measured on 4,424 shared dial cells: 156 of 156 slots
differ, max abs 1.0214. 'The basic block' names two different quantities."*

That measurement compared two **stored instruments built in different extractor
eras**. In one process, at one commit, on the same pixels, they do not differ at
all: **372 of 372 slots bit-identical at every one of 11 geometries**, through
the public free function both v1-372 extractors actually call, at both SIMD
tiers.

The failure #9 row is still a true and useful warning about the *instruments*.
It is not true of the *code*, and reading it as a code claim would send someone
hunting a divergence that option C (F3) already closed. Gated by
`extractor_entry_is_bit_exact_to_the_fold_v1_block`. The `FEATURE_SET_IDS`
row's framing is left to that document's owner; the correction is registered as
`feature-set-ids-row9-is-an-era-artifact-2026-09-05`.

### 4.5 The two remaining new items

* **28-thread determinism** was ungated (`fold_engine_parity` stops at 16 while
  the box has 32 hardware threads and the fleet nodes have more). Band merging
  is bit-exact *only because* it is sequential in band order; a scheduler change
  that reorders the merge would land exactly there. Now gated.
* **Degenerate-input finiteness** was ungated anywhere. A NaN in a feature
  vector poisons a model forward pass silently and would still pass a
  `to_bits()` comparison against itself. Now gated over five pathological input
  families.

---

## 4A. SERVABILITY CENSUS — "make sure everything can be served"

Measured today, at `3376baee`, on the committed golden real pair
(`zensim/tests/fixtures/v1_golden_real_{ref,dist}.png`, 96×96), through
`Zensim::compute` — **the production entry**, not a training one. Bake-level
probing uses the fastclass2 campaign's `zensim/examples/serve_custom_bake.rs`
(already on `main`); the profile-enum sweep uses this lane's probe
(`feature_invariant_probe profiles`), because `serve_custom_bake` takes a bake
path and cannot reach the `ZensimProfile` variants.

**Headline: 92 % of the bakes this repo has produced cannot be served by the
product entry point today, and two of the ten selectable shipped PROFILES are
among them.** This is the architecture lane's phase-gate baseline; this lane
does not fix it.

### 4A.1 The three censuses

| population | n | SERVED | REFUSED | served-but-MISMATCHED |
|---|--:|--:|--:|--:|
| **shipped bakes** (`zensim/weights/*.bin`) | 11 | **8** | **3** | 8 (see 4A.4) |
| **board bakes** (distinct bake files behind the 467 fullevals, all present on disk) | 433 | **32** | **400** | 32 |
| **registered feature sets** (`benchmarks/feature_sets_registry.json` → `sets`) | 14 | **3** | **11** | 3 |
| **shipped profiles** (`ZensimProfile` variants, default features) | 10 | **8** | **2** | 8 |

Every single refusal, in all four populations, is the same error:

```
ModelForwardFailed { reason: "bake declares more input features than the caller supplied" }
```

### 4A.2 The mechanism (one cause, four symptoms)

`Zensim::compute` emits a **372-layout** vector with `free_extras: Off` —
`compute_folded_v1_372_streaming_impl` builds its toggles with
`..Default::default()` (**P1**), and `compute_fold_backed` truncates to ≤372
(**P2**). The function that would decide which wider read a bake needs,
`wide_bake_v2_read`, is `allow(dead_code)` and reached only from tests (**P3**).

So the servability rule is exactly: **`caller_input_width() <= 372` serves;
anything wider is refused.** Measured declared-width distribution over the 433
board bakes — 4 at 156, 28 at 372 (these 32 are the SERVED set), 1 at 504, 8 at
720, 2 at 924, **389 at 944**.

### 4A.3 The two refused shipped profiles are the sharp finding

| profile | status | note |
|---|---|---|
| `B`, `BHdr`, `codec_target()`, `A`, `D`, `PreviewV0_1`, `PreviewV0_2`, `LegacyLinearV0_2` | **SERVED** | 372 (or 228) layout |
| **`C`** | **REFUSED** | `c_sdr_mlp944_corrmix_2026-08-05.bin`, `caller_input_width = 944` |
| **`CHdr`** | **REFUSED** | `c_hdr_l1t1944_2026-08-29.bin`, `caller_input_width = 944` |

`candidate-profiles` is **ON by default**, and
`c_sdr_mlp944_corrmix_2026-08-05.bin` is in `zensim/Cargo.toml`'s crates.io
`include` list — so it ships. A consumer who writes
`Zensim::new(ZensimProfile::C)` gets `ModelForwardFailed` on **every** image.
`c_sdr_purity944_2026-08-29.bin` and `c_hdr_l1t1944_2026-08-29.bin` are the
same class on disk (not in `include`).

**And the short-circuit hides it at exactly the input a smoke test would use.**
`C` and `CHdr` return **`IDENTITY (ref vs ref) score = 100.000000`** — because
`mark_identical` fires before the model is ever consulted (**P5**, §4.2). A
health check that scores an image against itself passes for a profile that
cannot score anything else. Registered as
`profile-c-chdr-unservable-2026-09-05`.

### 4A.4 Served-but-MISMATCHED — all of them, and it is not a new defect

For every population above, the SERVED column and the MISMATCHED column are the
same number, for one reason: **the vector the runtime serves is not the vector
any stored table holds.**

* The runtime side is **self-consistent**, measured this lane: the v1-372
  extractor entry, the fold's v1 block, the full 944 walk's v1 block, and both
  product engines agree **bit-exactly at 372 of 372 slots across 11 geometries
  and both SIMD tiers** (§3, probe (c); gates
  `extractor_entry_is_bit_exact_to_the_fold_v1_block`,
  `v2_era_blocks_do_not_move_the_v1_block`).
* The table side is **one extraction era behind** — **F3b**. The default 372
  root was built at `ea16c7ee` (13:21); option C flipped at `56bbcda2` (15:43)
  the same day. Re-extracting CSIQ at HEAD with the same tool on the same input
  file moves **120,804 of 135,096 basic cells, max \|Δ\| 4.536785**; every row
  differs on 285–341 of 372 slots.

So the mismatch is **temporal, not a second numeric defect**: no served-but-
mismatched case in this census has a mechanism other than F3b. Closing it needs
the registered-not-run 372 root re-extraction, no code change. The 944
populations are not affected by F3b — but they are 100 % unservable anyway, so
the question does not arise for them until P1–P3 are fixed.

**One consequence worth stating plainly for the architecture lane:** when the
increment lands and 944 bakes become servable, they will be served vectors from
the CURRENT extractor while their published verdicts were read on stored 944
roots of a declared era. The feature-set id (`<compute>@w<layout>/<era>#<hash>`)
is what makes that checkable, and `bake_verdict`'s
`--require-feature-set-match` is what makes it refuse rather than mis-score.
The servability fix should land with that check wired into the serving path,
not only the verdict path — otherwise fixing P1–P3 converts a loud
`ModelForwardFailed` into a silent wrong number, which is strictly worse.

### 4A.5 Reproduce

```bash
# shipped bakes
for f in zensim/weights/*.bin; do
  cargo run --release --example serve_custom_bake \
    --features custom-profiles,candidate-profiles -- \
    "$f" zensim/tests/fixtures/v1_golden_real_{ref,dist}.png
done
# the profile enum
feature_invariant_probe profiles out.tsv
```
Raw output: `serve_weights.txt`, `serve_board.txt`, `profiles.tsv` in
`/mnt/v/output/zensim/feature-audit-2026-09-05/`.

---

## 5. What would need an era break to fix

| item | why it is an era break | cost named |
|---|---|---|
| F4's cap or `C1` on SSIM `d` | changes v1's shipped bytes on every image with high-magnitude chroma | re-extract + re-verdict the whole 372 lineage; the winsor guard currently absorbs it |
| F12's `mu` scalar tail | changes v1's shipped bytes on the last `height % 8` rows | golden-gate policy decision, not a perf drive-by |
| F13's `dense_block_kernel` restructure | changes the 944 accumulation order | Amdahl upper bound **1.17×@8T** against re-extracting every 944 table AND re-training every 944 model |
| F3b's 372 root re-extraction | no code change — the code is already correct | a decode-bound corpus pass, registered and not run |
| F15's `PJND_FRAGILITY` at identity | changes 12 slots of every 944 table | small blast radius; the slot is not in the shipped D/B input sets |
| F5's compensated `Σs²` | changes the free-40 slots | no shipped bake reads them yet — **this is the cheapest window to fix it** |

---

## 6. Reproduce

```bash
cd ~/work/zen/zensimx
cargo test --release -p zensim \
  --features training,feature-regime-v2,custom-profiles,classification \
  --test feature_invariants                       # 10 gates

cargo build --release -p zensim \
  --features training,feature-regime-v2,custom-profiles,classification \
  --example feature_invariant_probe
B=target/release/examples/feature_invariant_probe
$B identity     out.tsv    # the 286-slot census, 4 routes
$B determinism  out.tsv
$B engineparity out.tsv    # 33 comparisons
$B degenerate   out.tsv
$B ladder       out.tsv    # includes the CTRL_MSE stimulus control
$B scale        out.tsv
$B depth        out.tsv
$B cliff        out.tsv    # the fabricated-vs-computed identity step
$B widths       out.tsv    # the toggle-shape -> width matrix of §4.1
$B dump         out.tsv    # full to_bits() vectors, for cross-tier diffing
ZEN_FIP_CAP_V3=1 $B dump v3.tsv   # same, dispatch capped at v3/AVX2+FMA
```

`ZEN_FIP_CAP_V3=1` refuses to run rather than mislabel a native pass as capped.
