# G-ADDR — the dial ADDRESSABILITY gate (PRE-REGISTERED 2026-09-04)

**Lane:** `claude-dialgate`.
**User rule (2026-09-04), verbatim spirit:** *"floor and ceiling dial addressability is
crucial … any model that limits dial range cannot ship"*, and *"I want to be able to reuse
ssim2 scores if possible."*
**Owner:** `zensim-validate/src/dial_addressability.rs` (registry
`benchmarks/dial_addressability_floor_2026-09-04.json`, embedded with `include_str!`),
wired into `bake_verdict`'s DIAL panel and emitted at `dial.addressability` in
`--full-json`.

> **§1–§6 are PRE-REGISTRATION.** They were written and committed (`f2eeccf9`) BEFORE any
> era-corrected anchor was fitted. The bars are the SHIPPED dial's own measured values plus
> the already-registered product conventions; nothing here was chosen after seeing a
> candidate's numbers, and nothing here was lowered to let a candidate through. Results are
> §7 onward.

---

> **⚠ SUPERSEDED BARS BELOW — READ [§14 "Re-pin 2026-09-04"](#14-re-pin-2026-09-04--the-bars-are-the-reference-metric-now) FIRST.**
> USER DECISION 2026-09-04: *"I don't think we should pin to B, ssim2 seems a better
> mentor."* Every REGRESSION bar quoted in §0-§11 is the **retired shipped-B** pin set. It
> is still readable (the registry keeps it; `bake_verdict --gaddr-reference shipped_b`
> reproduces this grading exactly) but it is **no longer a bar**. The active bars are
> `peer_ssim2`'s own measured values, and the re-graded candidate table is §14.4.

---

## 0. Headline

**No candidate passes. `B dial-era v2` is NOT proposed.** Three bars block the leading arms,
and all three are now measured rather than argued:

- **A4** (`p5 ≤ 13.645`) is **not attainable by any output spline on B's ordering**. An
  ORACLE arm — the eval grid itself as the anchor with the ssim2 truth as target — reads
  `p5` **21.5–22.8** across a 10× knot sweep, *further* from the bar than every real
  candidate (17.7–18.6). Shipped B clears it by mapping the low band **below** its
  conditional median: on the 221 lowest-truth cells it is +23.27 off the truth, the best
  candidate +22.80, the oracle +27.09. **Profile D reads `p5 = 9.52`** on the same
  instrument, so the bar IS reachable — by different weights, not by a different spline.
- **A1 / A3 / A6** (`max`, `p95`, `dynamic_range`) sit **ABOVE the reference metric's own
  values on the same grid** — truth `max` 98.38 / `p95` 95.46 / DR 85.20 against bars
  99.98 / 99.72 / 86.08. A dial calibrated exactly to the truth fails all three, and both
  other shipped profiles (A and D) do. Those bars encode the incumbent's *stretch* at the
  ceiling, not its reach.
- **C2 ⊻ C6 is a MODEL defect with a proof.** 266 of 4,424 dial-grid cells (6.01 %) have a
  raw prediction ABOVE the identity vector's — B ranks 6 % of lossy output better than a
  perfect copy. Pin identity at 100 and they cap (C2 fails, measured: 267 cells at exactly
  100.00); leave identity below 100 and they out-score it (C5/C6 fail). Monotonicity admits
  no third option. The weights must change.

**What the lane did establish:** two arms (`ne12_ss_unc_id100`, `ss_unc_id100_lowband`) that
are **rank-identical to shipped B**, retain **79 % / 77 %** of the era correction, calibrate
better than shipped on the whole grid (MAE 4.29 / 4.37 vs **5.45**), and **fix four of the
shipped dial's standing contract defects** — the dial goes genuinely negative on
negative-truth input for the first time, identity reads exactly 100, and nothing out-scores
a perfect copy. Plus: a **codec-specific ssim2 reuse rule** (§7), the first measurement of
what the **truth** says the dial's ends should read (§8), and the CID22 contamination audit
of both anchors — **0 hits at d ≤ 10** (§12).

---

## 1. Why a new gate

`benchmarks/imazen26_anchor_2026-09-04.md` established that shipped **B**'s −5..−6 point
dial skew is an **era** term: re-anchoring the SAME 2,000 anchor rows read today recovers
+3.892 (CID22) / +4.798 (KonJND) / +3.864 (AIC-3), 78–82 % of the defect, with SROCC
identical to 5 dp. But every candidate that recovers it **loses reach**:

| | shipped | `B_safesyn_curera` | `B_im26anchor` | `B_im26topdense` |
|---|---:|---:|---:|---:|
| reach | 96.85 | 94.23 | 85.74 | 88.96 |
| p5 | 13.73 | 18.23 | 22.91 | 22.99 |
| dynamic range | 85.99 | 81.33 | 75.53 | 76.13 |

Under the user's rule none of those can ship. **Nothing in the existing eval sees this.**
SROCC is rank-invariant and therefore *structurally* blind to it (all five arms read
0.88212 on CID22). G3 is about ordering. G1 asks only `p5 ≤ 25 ∧ p95 ≥ 85` — a bar every
one of the compressed arms clears comfortably. So a dial can lose 11 points of reach and
every panel in the mandatory two-panel eval will call it fine.

G-ADDR is the missing measurement.

## 2. The two tiers — and why they must not be merged

| tier | bar | question |
|---|---|---|
| **REGRESSION** (`A1`–`A9`) | the SHIPPED dial's own value on the SAME instrument | *is this candidate worse at the ends than the dial users have today?* |
| **CONTRACT** (`C1`–`C6`) | absolute product requirements | *does this dial meet the product contract at all?* |

They are separate because **the shipped dial fails four contract rows today** (§4). Merging
them would make every candidate's report say "FAIL" for reasons it did not cause. Every
report therefore prints an `incumbent` column: what the shipped dial reads on that same
axis. A ship needs **both** tiers PASS.

`—` is **NOT MEASURED**, never a pass. An unregistered dial grid makes the whole regression
tier `NOT MEASURABLE` — a bar you can dodge by choosing a friendlier instrument is not a
bar.

## 3. The bars

Instrument: the **canonical** dial grid
`dial_grid_372col_2026-05-29_quarantined_v2.parquet`, sha `6546c43e6d9572dc…`, 4,424 rows.
(The non-canonical `_quarantined` grid that `imazen26_anchor_2026-09-04.md` measured on is
*also* registered, with its own row, so those published numbers stay checkable against
their own instrument. New work uses the canonical one.)

### REGRESSION tier — bar = shipped B on the same instrument

| id | axis | direction | bar (canonical grid) | bar (doc grid `b5d27f21…`) |
|---|---|---|---:|---:|
| A1 | ceiling — pooled dial `max` | ≥ | **99.98330778475787** | 99.98376794026095 |
| A2 | floor — pooled dial `min` | ≤ | **3.12950123756248** | 3.12950123756248 |
| A3 | robust ceiling — `p95` | ≥ | **99.72170874183841** | 99.71524151863066 |
| A4 | robust floor — `p5` | ≤ | **13.645032446453126** | 13.726111774203428 |
| A5 | `reach` = max − min | ≥ | **96.85380654719539** | 96.85426670269847 |
| A6 | `dynamic_range` = p95 − p5 | ≥ | **86.07667629538528** | 85.98912974442723 |
| A7 | negative-tail probe dial `min` | ≤ | **2.516685884084839** | (probe is grid-independent) |
| A8 | negative-tail probe dial `p1` | ≤ | **3.981383254902343** | — |
| A9 | negative-tail `frac_below_zero` | ≥ | **0.0** | — |

A4's bar on the doc grid (13.7261) and A5/A6 reproduce the imazen-26 record's `p5 13.73 /
reach 96.85 / DR 85.99` exactly, which is the cross-check that this gate measures the same
quantity that record does.

### CONTRACT tier — absolute

| id | axis | bar | provenance |
|---|---|---|---|
| C1 | monotonicity | ≥ 0.93 | registered G3, `docs/EVAL_PANEL_REQUIREMENT.md` |
| C2 | flat/clamp dead-zone | ≤ 0.05 | registered G3 |
| C3 | negative values WORK — `frac_below_zero` on an all-negative-truth probe | > 0 | CLAUDE.md *"NEGATIVE zensim values MUST work … do NOT clamp at 0"* |
| C4 | deepest probe dial | < 0 | same |
| C5 | `dial(ref==dist)` inside `[97.5, 100]` | 0 rows outside | `benchmarks/standard_bake_packing_2026-05-27.md` + the QAT record |
| C6 | dial-grid cells out-scoring a perfect copy | 0 | *"0 above-identity"*, same record |

## 4. The instruments (pinned, content-addressed)

**`negtail_probe_372_2026-09-04.parquet`** — sha `5609d19fa10aef81…`, 2,000 rows,
`/mnt/v/output/zensim/dialgate-2026-09-04/`. Cut from
`canonical-2026-07-15/train/kadis_negrich.parquet` (266,111 rows, 237,675 with
`ssim2_gpu < 0`) by **20 equal-count quantile bins over the negative population, lowest 100
row indices per bin** — deterministic, no RNG. Truth span `ssim2_gpu` −770.62 … −0.33;
**every row's reference metric is negative**, so a correct dial must go below zero somewhere
on it. Current era (extracted 2026-07-15, after both extractor fixes).

**`identity_probe_372_2026-09-04.parquet`** — sha `e6f9096b8e0ebd97…`, 38 rows. All 38
distinct reference images of the canonical dial grid, each paired with **itself**, extracted
by `extract_features_372col` through imazen decoders at this lane's HEAD.

> **MEASURED, and it changed the design: the identity feature vector is the ZERO vector, for
> every image.** All 38 rows come back byte-identical and all-zero — which is what a
> *difference* metric must do, and it makes the identity dial a **scalar property of the
> bake**, `dial(0⃗)`, not a per-image measurement. The probe is kept at 38 rows rather than
> 1 so that property stays gated instead of assumed.

Measured identity dial across the shipped bakes:

| bake | `dial(0⃗)` | inside [97.5, 100]? |
|---|---:|---|
| `v47_strict_qat_native_2026-05-27` (Profile A) | **97.6893** | ✓ |
| `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` (Profile B, SHIPPED) | **96.2412** | ✗ |
| `d_sdr_add156_dense_dial_2026-08-31` (Profile D) | **96.1157** | ✗ |

So `[97.5, 100]` is a **v47-era** property; both shipped LINEAR dials sit ~1.3 points below
it. Registered as specified and **not relaxed** — the incumbent's failure is a finding.

## 5. Where the incumbent stands — the gate's first reading

`bake_verdict --bake <shipped B> --dial-grid <canonical> --negtail-probe … --identity-probe …`

**`NOT SHIPPABLE — regression PASS / contract FAIL`**, 11 pass / 4 fail / 0 not measured.
Regression is a tie by construction (the bars are its own values, and the comparators are
inclusive — a dedicated test pins that). The four contract failures are **standing defects
in the shipped product dial**:

| id | shipped B reads | bar |
|---|---:|---|
| C3 negative values work | `frac_below_zero` **0.0000** on a probe whose every row's truth is negative | > 0 |
| C4 deepest tail dial | **+2.5167** | < 0 |
| C5 identity band | **96.2412** (38/38 rows outside) | [97.5, 100] |
| C6 nothing above identity | **266 of 4,424 cells (6.01 %)** out-score a perfect copy; worst `1a20ecb0c1b92466_1022x818` jxl d=0.05 at **99.9833** vs identity **96.2412** | 0 |

**Read C3/C4 together with the floor mechanism.** Shipped B's dial anchor
(`multiband_anchor_dial100.parquet`) stores `target_score = max(ssim2, 0)`. 147 of its 2,000
rows have a genuinely negative ssim2 — down to **−64.16** — and **every one is stored as
0**. `dial_spline::fit_spline_knots` then collapses that run of `y == 0` bins to a single
bottom knot (the `neg_tail` dedup), so the spline carries **no in-distribution evidence at
all** about how far below zero the dial should go; the whole negative tail is a linear
extrapolation off one knot, and it extrapolates to `+2.52` on inputs whose truth is −770.

That is also why a *pure* re-anchor fails A4 by construction: the era correction is a
near-uniform **+3.9…+4.8** lift, so the floor rises with everything else and the clamp is
the reason nothing pushes back. **The lever is the clamp, not the anchor.**

## 6. Gate implementation — what landed

- `zensim-validate/src/dial_addressability.rs` — the owner: registry reader, `GridMeasure` /
  `NegTailMeasure` / `IdentityMeasure`, `evaluate`, markdown + JSON renderers. 12 tests.
- `benchmarks/dial_addressability_floor_2026-09-04.json` — the append-only registry.
- `bake_verdict` — new `--negtail-probe` / `--identity-probe`; the DIAL panel now emits the
  G-ADDR section on **every** run; `--full-json` carries `dial.addressability` (verdict +
  every raw measurement) and `dial.min` / `dial.max`.
- Test `canonical_dial_grid_has_a_g_addr_floor_row` pins the registry to `bake_verdict`'s
  own `CANONICAL_DIAL_GRID_SHA256`, so rotating the canonical grid without measuring the
  reference on it fails the build instead of silently disarming the gate.

**A one-ULP defect found and fixed on the way.** `serde_json`'s default float parser is not
correctly rounded: the bar written as the shipped dial's own `99.98330778475787` parsed back
as `…788`, and the reference bake **failed its own bar by one ULP** (A1 and A5 read ✗ in the
first run). Fixed by enabling serde_json's `float_roundtrip` feature for `zensim-validate`
(parsing only — serialization already goes through ryu), with
`registry_floats_round_trip_bit_exactly` as the guard. Worth knowing beyond this gate:
**every float bar `freeze_check` reads out of a fulleval JSON had the same hazard.**

---

# RESULTS

## 7. ssim2 reuse across decoder eras — MEASURED, and the answer is CODEC-SPECIFIC

**Question (user):** *"I want to be able to reuse ssim2 scores if possible."* Both anchors'
`target_score` is a **stored** ssim2 computed at the corpus's own decode era, while the
features beside it are decoded today — the "mixed-era caveat" §2c of the imazen-26 record
refused to paper over.

**Method.** Both anchors' pairs re-scored with `zenmetrics batch --metric ssim2` — imazen
`fast-ssim2` through the umbrella's `cpu-ssim2`, decode by imazen codecs — on **today's**
bytes, against the stored value. (`zenmetrics`'s prebuilt CLI had no JXL decoder; it was
rebuilt at its default feature set, which restores jpeg/webp/avif/jxl. Its `--metric ssim2`
help text still says "via the `ssimulacra2` crate" — read from source, the CPU path routes
through `zenmetrics_api`'s `cpu-ssim2` = **`fast-ssim2`**, i.e. imazen. The help text is
stale, the dispatch is not.)

| anchor | n | stored col | median \|Δ\| | p95 | p99 | max | frac \|Δ\| > 0.5 | SROCC | PLCC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| shipped safesyn `multiband_anchor_dial100` | 2,000 | `ssim2_gpu` | 0.0534 | 0.983 | 2.103 | 5.130 | **11.30 %** | 0.99984 | 0.99991 |
| imazen-26 `imazen26_multiband_anchor_dial100` | 4,000 | `score_ssim2` | 0.0043 | 0.241 | 1.750 | 8.375 | **2.50 %** | 0.99989 | 0.99993 |

**Per codec — this is the whole finding:**

| safesyn family | n | median \|Δ\| | frac > 0.5 | mean signed |
|---|---:|---:|---:|---:|
| `zenwebp-default-m4` | 279 | 0.0134 | 0.36 % | +0.032 |
| `zenjxl-e7` | 210 | 0.0394 | 4.76 % | −0.073 |
| `mozjpeg-rs-420-e4` | 481 | 0.0403 | 2.08 % | +0.066 |
| `zenavif-s5-e6` | 372 | 0.0495 | 6.99 % | +0.070 |
| `zenjpeg-420-e2` | 373 | 0.0613 | 3.49 % | +0.093 |
| **`zenjpeg-420-xyb-e2`** | **285** | **0.6370** | **58.25 %** | **+0.632** |

| imazen-26 codec | n | median \|Δ\| | frac > 0.5 | mean signed |
|---|---:|---:|---:|---:|
| **`zenwebp`** | 1,000 | **0.0000** | **0.00 %** | +0.0000 |
| `zenjpeg` | 1,000 | 0.0082 | 0.20 % | −0.002 |
| `zenjxl` | 1,000 | 0.0409 | 3.80 % | −0.069 |
| `zenavif` | 1,000 | 0.0014 | 6.00 % | −0.021 (max 8.37) |

**The reuse rule, as measured.** Against the pre-registered decision rule ("reusable iff the
fraction past 0.5 is 0") the answer is **NO in aggregate** — 11.3 % / 2.5 % of rows move past
the dial's materiality — so both anchors below were **re-scored at their own decode era**,
which is what makes them single-era in features AND target and closes the imazen-26 record's
§2c caveat. But the aggregate hides the structure:

- **`zenwebp` re-scores to 0.0000 on 1,000 of 1,000 imazen-26 rows** — bit-exact. That also
  settles what the stored bigcodec `score_ssim2` column IS: the same CPU `fast-ssim2` path,
  not a GPU column, so for imazen-26 this measurement is a **pure decoder-era** term with no
  CPU/GPU confound.
- The safesyn drift is carried by **one family**: XYB JPEG (`zenjpeg-420-xyb-e2`), median
  0.637 and 58 % past materiality, against ≤ 0.061 and ≤ 7 % for every other family. That is
  the same family the retracted `image::open` probe stumbled over — the difference is that
  zenjpeg's XYB path genuinely evolved, which an imazen-only re-read can measure and a
  third-party decoder cannot see at all.
- The imazen-26 tail is **`zenavif`** (median 0.0014 but 6 % past 0.5, max 8.37), consistent
  with its manifest note that AVIF decodes to `Rgb16` and is flattened.

**Rule to carry forward:** a stored ssim2 target is reusable across decoder eras **per
codec** — bit-exact for zenwebp, immaterial for zenjpeg/mozjpeg/zenjxl — and is **NOT**
reusable for **XYB JPEG** or for **AVIF's tail**. Rank is untouched everywhere (SROCC ≥
0.9998), so a stored target is always safe for anything rank-based; it is the *absolute*
value that moves. Re-scoring 6,000 pairs cost 171 s of CPU, so when the bytes are on disk,
re-score.

## 8. What the truth says the dial should read — the reference nobody had measured

Before judging any candidate, the same instrument was measured with the **reference metric
itself**: `dialcells_ssim2_qv2grid.tsv` (2026-08-31) holds ssim2 for every one of the 4,424
canonical dial-grid cells. The dial's anchor target IS ssim2, so this is what a perfectly
calibrated dial would read.

| | min | p5 | median | p95 | max | reach | dyn. range | frac < 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ssim2 TRUTH** | **−55.35** | **10.26** | 87.28 | **95.46** | **98.38** | **153.73** | **85.20** | **3.48 %** (154 cells) |
| shipped B | +3.13 | 13.65 | — | 99.72 | 99.98 | 96.85 | 86.08 | 0 |

Three things follow immediately, and they reframe two of the pre-registered bars:

1. **Shipped B's floor is not merely compressed, it is absent.** 154 grid cells have a
   genuinely negative ssim2 and B scores **every one of them positive**, bottoming out at
   +3.13 where the truth reaches −55.35.
2. **Shipped B's `dynamic_range` (86.08) is LARGER than the truth's own (85.20).** A dial
   calibrated exactly to the truth would **fail bar A6**. A6 is therefore not a reachable
   addressability bar; it is a bar the incumbent clears by stretching.
3. **Shipped B's p5 (13.65) is 3.4 points ABOVE the truth's (10.26)** — so the incumbent is
   not "more addressable" at p5 either; it is closer to the truth there than the candidates,
   but for a reason §10 shows is miscalibration rather than reach.

**And the same is true at the ceiling.** The truth's `max` is 98.38 and its `p95` 95.46;
shipped B reads 99.98 and 99.72 — **+1.6 and +4.3 above the reference metric**. So bars A1,
A3 and A6 all sit ABOVE the truth's own values on this grid. They encode the incumbent's
*stretch*, not its reach, and **a dial calibrated exactly to the truth would fail all three**.
A2 / A5 / A7-A9 (the floor and tail axes) are the ones where the incumbent is genuinely
short of the truth, by an enormous margin, and those are the axes the candidates fix.

### 8a. Running the gate on the OTHER shipped profiles — the cross-check that settles it

Same grid, same probes, zero extra work (the gate runs on every `bake_verdict` invocation):

| profile | reach | min | max | p5 | p95 | DR | tied | negtail min | frac<0 | identity | above-id | fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **B** (SHIPPED SDR) | 96.85 | 3.13 | 99.98 | 13.65 | 99.72 | 86.08 | .0000 | +2.52 | .0000 | 96.24 | 266 | C3 C4 C5 C6 |
| **A** `v47_strict_qat_native` | 113.90 | −17.11 | 96.78 | 16.67 | 94.51 | 77.85 | .0000 | **−93.90** | **.5575** | **97.69** | **0** | A1 A3 A4 A6 |
| **D** `d_sdr_add156_dense_dial` | 108.25 | −12.20 | 96.05 | **9.52** | 95.28 | 85.77 | .0000 | **−100.00** | **.8580** | 96.12 | **0** | A1 A3 A6 C5 |
| ssim2 TRUTH | 153.73 | −55.35 | 98.38 | 10.26 | 95.46 | 85.20 | — | — | — | — | — | — |

Three things this settles:

- **Profile A is the only bake measured here that passes the ENTIRE CONTRACT tier** —
  identity 97.6893 in band, 0 cells above identity, and a negative tail that actually works
  (55.75 % of the all-negative-truth probe scores below zero, min −93.90). The QAT-era dial
  was right about all four things the shipped SDR dial gets wrong.
- **A4 is reachable — by a DIFFERENT MODEL.** Profile D reads `p5 = 9.52`, comfortably under
  the 13.645 bar and within 0.7 of the truth's 10.26. §10.1's impossibility is therefore
  precisely scoped: it is unattainable *for B's raw ordering*, and the residual is a weights
  limitation, now demonstrated rather than inferred.
- **A1 / A3 / A6 are the mirror image**: both A and D fail all three, and both sit BELOW the
  truth at the top while B sits above it. Only the incumbent clears bars that the reference
  metric itself does not.

## 9. The candidates — all rank-identical, and four of the incumbent's contract defects fixed

Every arm shares **identical weights, scaler and winsor guards**; only the output spline
differs. **CID22 SROCC is 0.88212 on every single arm** (KonJND −0.51938, AIC-3 0.76501, TID
0.77852, KADID 0.80847) — rank invariance verified, not assumed. Chain control: rebuilding
`B_safesyn_curera` through this lane's driver reproduces the imazen-26 lane's bake
**BYTE-IDENTICALLY** (sha `c414b3f91da83e69…`).

Measured on the canonical dial grid + both probes (bars in §3; **bold** = fails):

| arm | reach | min | max | p5 | p95 | DR | mono | tied | negtail min | negtail p1 | frac<0 | identity | above-id | fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SHIPPED B | 96.85 | 3.13 | 99.98 | 13.65 | 99.72 | 86.08 | .9792 | .0000 | 2.52 | 3.98 | .0000 | 96.24 | 266 | C3 C4 C5 C6 |
| `ctl_curera` (era only) | **94.23** | **5.73** | **99.96** | **18.10** | **99.57** | **81.47** | .9799 | .0000 | **5.02** | **6.68** | .0000 | 95.85 | 266 | A1-A8 C3-C6 |
| `ss_cur_rescored_clamped` | **94.23** | **5.75** | **99.97** | **18.05** | **99.66** | **81.61** | .9794 | .0000 | **5.04** | **6.69** | .0000 | 96.48 | 266 | A1-A8 C3-C6 |
| **(a)** `ss_cur_rescored_unclamped` | 98.01 | 1.96 | **99.97** | **18.05** | **99.66** | **81.61** | .9794 | .0000 | 0.35 | 3.95 | .0000 | 96.48 | 266 | A1 A3 A4 A6 C3-C6 |
| **(b)** `ss_..._unclamped_id100` | 98.12 | 1.88 | 100.00 | **17.92** | 100.00 | **82.08** | .9808 | **.0567** | 0.28 | 3.87 | .0000 | 100.00 | 0 | A4 A6 C2 C3 C4 |
| **(b′)** `ne12_ss_unc_id100` | 102.70 | −2.70 | 100.00 | **18.63** | 100.00 | **81.37** | .9794 | **.0567** | −4.37 | −0.52 | .0135 | 100.00 | 0 | **A4 A6 C2** |
| **(b″)** `ss_unc_id100_lowband` | **109.39** | **−9.39** | 100.00 | **18.64** | 100.00 | **81.36** | .9771 | **.0567** | **−11.68** | **−6.45** | **.0660** | 100.00 | 0 | **A4 A6 C2** |
| **(c)** `im26_..._unclamped_id100` | **88.09** | **11.91** | 100.00 | **22.11** | 100.00 | **77.89** | .9789 | **.0567** | **11.64** | **12.42** | .0000 | 100.00 | 0 | A2 A4-A8 C2 C3 C4 |
| `mix_unc_id100_lowband` | 101.30 | −1.30 | 100.00 | **21.47** | 100.00 | **78.54** | .9775 | **.0567** | −3.15 | 1.09 | .0045 | 100.00 | 0 | **A4 A6 C2** |
| **ORACLE** (train-on-test) | 98.62 | −0.85 | **97.77** | **21.75** | **97.13** | **75.38** | .9801 | .0000 | −2.45 | 1.23 | .0040 | 95.25 | 266 | A1 A3 A4 A6 C5 C6 |

**What each mechanism bought, isolated:**

- **Unclamping the anchor target** (keeping the 145 genuinely-negative rows instead of
  `max(ssim2, 0)`) moves the fitted spline's bottom from `y = 0.0` to `y = −18.2` and is the
  whole of the floor fix: grid `min` 5.75 → 1.96 (**A2 passes**), `reach` 94.23 → 98.01
  (**A5 passes**), negtail `min` 5.04 → 0.35 (**A7**), `p1` 6.69 → 3.95 (**A8**). It changes
  nothing else — same `p5`, same `p95`, same mono.
- **Anchoring the identity rows at 100** puts the spline's top knot at `(raw(0⃗), 100)`:
  `max` and `p95` reach exactly 100 (**A1, A3 pass**), the identity dial becomes **100.0000**
  (**C5 passes**) and **0** cells out-score a perfect copy (**C6 passes**) — the first time
  either has held. It costs `tied` 0.0000 → **0.0567**, which fails C2 (see §10.3).
- **Coarsening the knot grid to `n_edges = 12`** deepens the tail further (`min` 1.88 →
  −2.70, negtail `frac<0` 0 → 1.35 %, **C3 and C4 pass**) — the first arms in this whole
  lineage where a negative-truth input actually scores negative.
- **A low band** (1,000 kadis-negative rows, quantile-stratified and **disjoint from the
  negtail probe**, targets unclamped) deepens it much further: negtail `min` −11.68, `p1`
  −6.45, `frac<0` **6.60 %**, grid `min` −9.39, `reach` **109.39**.
- **imazen-26 as the anchor corpus REGRESSES the floor badly** — grid `min` +11.91, negtail
  `min` +11.64, `p5` 22.11 — and mixing it in drags the safesyn arms' floor back up. The
  imazen-26 record's own reach warning is confirmed and localised: it is a *low-band
  coverage* deficit, not a top-end one.

**Era correction retained.** Mean per-pair dial shift vs shipped B (100 % of pairs move by
more than 0.5 on every corpus):

| arm | CID22 | KonJND | AIC-3 |
|---|---:|---:|---:|
| `ctl_curera` (the record's arm) | +3.923 | +4.826 | +3.892 |
| **(b′)** `ne12_ss_unc_id100` | **+3.947** | **+4.498** | **+3.912** |
| **(b″)** `ss_unc_id100_lowband` | +3.729 | +4.744 | +3.686 |
| `mix_unc_id100_lowband` | +3.368 | +3.955 | +3.439 |

against the −4.977 / −5.857 era defect — i.e. the leading candidates still recover
**79 % / 77 %** of it, the same as the plain re-anchor, while also fixing four contract rows.

## 10. Why NO candidate passes — the precise impossibility, per axis

Three bars remain unmet by the best arms: **A4** (`p5 ≤ 13.6450`), **A6** (`dynamic_range ≥
86.0767`) and **C2** (`tied ≤ 0.05`). Each is now measured, not argued.

### 10.1 A4 — unattainable by any output spline ON B'S ORDERING (a different model reaches it)

An **ORACLE** arm was built to bound the question: the dial grid *itself* as the anchor, with
the **ssim2 truth** as the target — i.e. train-on-test, the best a monotone re-map of B's own
ordering can do. It is a diagnostic and can never ship; its only job is to bound A4.

| n_edges | 12 | 18 | 30 | 60 | 120 |
|---|---:|---:|---:|---:|---:|
| candidate `ss_unc_id100` p5 | 18.63 | 17.92 | 17.73 | 18.17 | 18.17 |
| **ORACLE p5** | — | **21.75** | **22.53** | **22.77** | **22.66** |

**The oracle is FURTHER from the bar than every candidate, at every knot count.** A monotone
map must assign one dial value per raw value, so the best it can do in a bin is that bin's
conditional median — and in the bin holding the 5th percentile of B's predictions that median
is ~22. The knot sweep (candidate d) moves `p5` by 0.9 points across a 10× range of knot
counts; it is not the lever.

The reason is visible in the low band. On the 221 lowest-truth cells (**mean truth −11.30**):

| arm | mean dial there | bias vs truth | MAE (low 5 %) | MAE (all 4,424 cells) |
|---|---:|---:|---:|---:|
| ssim2 truth | −11.30 | — | — | — |
| shipped B | +11.97 | **+23.27** | 23.31 | **5.45** |
| `ctl_curera` | +15.39 | +26.68 | 26.69 | 4.35 |
| `ne12_ss_unc_id100` | +13.66 | +24.95 | 25.00 | **4.29** |
| `ss_unc_id100_lowband` | **+11.50** | **+22.80** | **23.08** | 4.37 |
| **ORACLE** | +15.80 | **+27.09** | 27.13 | **4.17** |

**Read the last two columns together.** The oracle has the best whole-grid calibration
(MAE 4.17) *and* the worst low-band bias (+27.09) — because the model's raw prediction does
not separate those cells at all, so any honest monotone map hands them the conditional
median. `ss_unc_id100_lowband` reaches +22.80, **better than shipped B's +23.27 and better
than the oracle's**, which is as far as an output spline can go. Every candidate beats
shipped B on whole-grid MAE (4.29–4.37 vs **5.45**).

So shipped B's `p5 = 13.65` is not addressability — it is the low band mapped *below* its
conditional median. **A4 as pre-registered rewards that.** It is unattainable by a dial
change *on B's ordering*, and the residual is a **model (weights) limitation** — which §8a
demonstrates directly: **Profile D (`d_sdr_add156_dense_dial`) reads `p5 = 9.52`**, under the
bar and within 0.7 of the truth's 10.26, on the same grid with the same gate. A4 is a
reachable bar; B cannot reach it by re-splining.

### 10.2 A6 — the bar exceeds the truth's own dynamic range

`dynamic_range = p95 − p5`. With identity anchored at 100 the ceiling is exactly 100, so
A6 reduces to `p5 ≤ 13.923` — i.e. **A6 is A4 with a 0.28-point discount** and falls with it.
Independently: **the ssim2 truth's own dynamic range on this grid is 85.196, below the
86.077 bar.** A dial calibrated exactly to the truth fails A6. The bar measures stretch, not
reach.

### 10.3 C2 ⊻ C6 — a hard either/or created by the MODEL, not the dial

**266 of 4,424 dial-grid cells (6.01 %) have a raw prediction ABOVE the identity vector's** —
B ranks 6 % of *lossy* codec output better than a perfect copy. An output spline is monotone,
so it cannot reorder them. Exactly two outcomes exist:

- **Identity pinned at 100** (arms b, b′, b″): those cells hit the [0,100] cap. Measured:
  **267 of 4,424 cells sit exactly at 100.00** in both id100 arms — matching the 266 count —
  and the adjacent-rung ties they create are the entire `tied = 0.0567`. C5 and C6 pass;
  **C2 fails.**
- **Identity below 100** (shipped B, arms a, oracle): those cells spread above identity.
  `tied = 0.0000`; **C5 and C6 fail** (266 cells above identity, worst
  `1a20ecb0c1b92466_1022x818` jxl d=0.05 at 99.98 against identity 96.24).

Monotonicity forces `dial(identity) ≤ dial(cell)` whenever `raw(identity) ≤ raw(cell)`, so no
choice of knots satisfies both. **C2 and C6 cannot both pass while those 266 raw inversions
exist.** The fix is the weights, not the dial. (This is also the first time the defect has
been *counted*: 6.01 % of the grid, on the shipped product dial, today.)

## 11. Recommendation

**No candidate passes the full G-ADDR gate, so under the user's rule none may ship, and none
is proposed for the board.** `B dial-era v2` is NOT proposed. That is the honest answer and
the gate did its job on its first use.

What the lane established, in the order it matters:

1. **The bars that block the leading candidates are, in part, not addressability bars.**
   A4 is unattainable by any monotone dial *on B's ordering* (though Profile D reaches it —
   §8a — so it is a weights problem, not an unreachable bar); and **A1, A3 and A6 all sit
   ABOVE the reference metric's own values on the same grid** (truth `max` 98.38 / `p95`
   95.46 / DR 85.20 vs bars 99.98 / 99.72 / 86.08). A dial calibrated exactly to the truth
   fails all three, and both other shipped profiles do. **This is a finding about the
   pre-registration, not a request to relax it** — the bars stay exactly as written, no
   candidate was graded against anything softer, and the arms are reported as failing. The
   user's decision is whether to REPLACE the four *ceiling/spread* bars with ones a
   correctly-calibrated dial can meet: truth-referenced (`|dial end − truth end| ≤ δ`) or
   calibration-referenced (low-band and whole-grid MAE against the reference metric), both of
   which **every candidate already beats the incumbent on** (whole-grid MAE 4.29-4.37 vs
   5.45). The FLOOR bars (A2, A5, A7-A9) need no revision — they are the axes where the
   incumbent is genuinely short of the truth, and the candidates fix them.
2. **`ne12_ss_unc_id100`** (sha `2deeae9c…`) and **`ss_unc_id100_lowband`** (sha `ef4298ef…`)
   are the leading arms: rank-identical to shipped B, era correction retained (+3.9 / +4.5 /
   +3.9 and +3.7 / +4.7 / +3.7), whole-grid calibration better than shipped (MAE 4.29 / 4.37
   vs 5.45), **and they fix four of the incumbent's standing contract defects** — the dial
   goes genuinely negative on negative-truth input for the first time (C3, C4), identity
   reads exactly 100 (C5), and nothing out-scores a perfect copy (C6). They fail only A4, A6
   and C2.
3. **C2 vs C6 is a model defect with a proof.** 266 lossy cells out-rank a perfect copy in
   B's RAW space; no output spline can hold both bars. It needs a weights change, and it is
   now counted and named rather than latent.
4. **imazen-26 is the wrong anchor for the FLOOR.** Every imazen-26 arm regresses grid `min`
   to +11.9 and negtail `min` to +11.6. Its deficit is low-band coverage in prediction space,
   which is fixable by adding a low band — not by top-densification, which the imazen-26
   record had already found could not move `p5`.

**A note on other regimes.** A 720/944-input bake reads `regression NOT MEASURABLE` and
`contract INCOMPLETE`: its default dial grid is not in the registry and the 372-wide probes
refuse to score against it (both refusals print loudly on stderr). That is correct and
deliberate — registering a 944 grid needs a **shipped 944-class reference dial** to measure
the floor from, and none exists. Verified on `c_sdr_purity944_2026-08-29`: 2 pass, 0 fail,
13 not measured, no silent pass anywhere.

**Registered, not run:** (i) a non-kadis low band — `ss_unc_id100_lowband`'s low rows come
from `kadis_negrich`, disjoint from the negtail probe by construction but same-distribution,
so a ship-grade version wants a low band from the anchor's own corpora; (ii) wiring
`dial.addressability` into the gauntlet board (the JSON is emitted; nothing reads it yet);
(iii) the A4/A6 replacement above, which is a user decision.

## 12. CID22 contamination audit of the anchors — CLEAN at the strict threshold

The brief asked whether an audit of the imazen-26 anchor's origins against CID22's 49
validation references exists. **It did not.** `benchmarks/imazen26_holdout_audit_2026-08-25.json`
audits imazen-26 *ids* (verdict for `ext_cid22`: "clean — different id namespace") and
`benchmarks/imazen26_dhash_audit_2026-08-27.md` runs dHash against the picker corpus and
synthetic-v2, explicitly registering the cross-corpus dHash check as a follow-up. Run here,
for both anchors, with the owner:

```
check_holdout_overlap --cid22-refs /mnt/v/dataset/cid22/CID22_validation_set/original \
    --training-csv <anchor refs> --threshold 10
```

| anchor | distinct refs | flagged d ≤ 10 | d ≤ 16 (screen) | closest |
|---|---:|---:|---:|---|
| imazen-26 anchor | 1,224 | **0** | 67 rows / **9 origins** | d=12 `o_1442` ~ CID22 `2887497.png` |
| shipped safesyn anchor | 1,495 | **0** | **0** | d=17 `09928cea53e5d8c9_1022x818` ~ `2887497.png` |

**Neither anchor has a single reference within the strict d ≤ 10 contamination threshold of
a CID22 validation reference.** The 9 imazen-26 origins in the d ≤ 16 *screening* tier are
reported, never quarantined, per the 2026-05-14 policy — a side-by-side review page (no image
processing, the originals as-is) is at
`/mnt/v/output/zensim/dialgate-2026-09-04/overlap-review/index.html`
(`http://localhost:3300/zensim/dialgate-2026-09-04/overlap-review/`). No action is proposed
and none should be taken without the user's eye pass.

## 13. Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict --bin bake_dial_refit
D=/mnt/v/output/zensim/dialgate-2026-09-04

# gate any bake (the G-ADDR section prints on every dial-panel run)
./target/release/bake_verdict --bake <bake.bin> \
    --dial-grid /mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet \
    --negtail-probe  $D/negtail_probe_372_2026-09-04.parquet \
    --identity-probe $D/identity_probe_372_2026-09-04.parquet \
    --corpora cid22,konjnd,kadid,tid,aic3 --full-json verdict.json

# build + gate one candidate arm (shared-anchor -> add-winsor -> extend-top)
scripts/dialgate_arms.sh <label> <anchor.parquet> [n_edges] [extend_top_anchor]

# the ssim2 era measurement
~/work/zen/zenmetrics/target/release/zenmetrics batch --metric ssim2 \
    --pairs $D/build/safesyn_anchor_pairs.tsv --output $D/build/safesyn_anchor_ssim2_today.tsv

# the CID22 contamination audit
./target/release/check_holdout_overlap \
    --cid22-refs /mnt/v/dataset/cid22/CID22_validation_set/original \
    --training-csv $D/build/im26_anchor_refs_for_overlap.csv --threshold 10 \
    --out-tsv $D/build/overlap_im26_vs_cid22.tsv
```

Artifacts: `/mnt/v/output/zensim/dialgate-2026-09-04/`
— `ref/` (the shipped-B floor verdicts), `build/` (pairs, re-scored ssim2, per-cell dial
dumps, overlap TSVs), `anchors/` (every candidate anchor parquet, sha-listed in §9's arms),
`arms/` (every bake + `--full-json` verdict), `overlap-review/` (the d ≤ 16 screening page),
and the two pinned probes at the top level.

**LAN mirror** (the probes are load-bearing — the registry references them by path AND
sha256, and a missing probe degrades the gate to NOT MEASURED):
`s3://zentrain/dialgate/2026-09-04/` holds both probes, every candidate anchor and every
arm bake (73 objects; `source ~/tmp/_lan_env.sh` for `$ZEN_S3_ENDPOINT`).

Key shas: `ne12_ss_unc_id100.bin` `2deeae9ce7da9cc2…`, `ss_unc_id100_lowband.bin`
`ef4298ef4d938be6…`, anchor `ss_cur_rescored_unclamped_id100.parquet` `6ce2c32971a34791…`,
anchor `ss_unc_id100_lowband.parquet` `a91a676156d13b08…`, chain control
`ctl_curera.bin` `c414b3f91da83e69…` (= the imazen-26 lane's `B_safesyn_curera`, byte-identical).

---

# 14. Re-pin 2026-09-04 — the bars are the REFERENCE METRIC now

**USER DECISION, verbatim:** *"I don't think we should pin to B, ssim2 seems a better
mentor."*

**Landed:** owner `zensim-validate/src/dial_addressability.rs`, registry
`benchmarks/dial_addressability_floor_2026-09-04.json` (append-only), driver
`scripts/dialgate_arms.sh score`.

## 14.1 Why the incumbent was the wrong mentor

The gate's own first run measured its pins to be defective — not merely strict, but
pointing the wrong way. Both findings are in §8 and §10 and neither was known when the
bars were pre-registered:

1. **A1 / A3 / A6 sat ABOVE what the reference metric itself reaches on the same grid.**
   Truth `max` 98.3766 / `p95` 95.4593 / DR 85.1960 against bars 99.9833 / 99.7217 /
   86.0767. A dial calibrated *exactly to the truth* failed all three, and so did both
   other shipped profiles. Those bars encoded the incumbent's **stretch**, not its reach.
2. **A4 was met by B only through a −23-point low-band bias.** On the 221 lowest-truth
   cells B reads +11.97 where the truth is −11.30, and the train-on-test ORACLE — the
   ceiling for any monotone re-map of B's ordering — reads `p5` 21.5–22.8. B's low `p5`
   is the low band mapped *below* its conditional median; the old A4 rewarded exactly that.

A gate pinned to the incumbent therefore **barred candidates for being closer to the truth
than the incumbent is**. That is the failure the re-pin fixes.

## 14.2 What changed, precisely

Registry rows are now keyed **`(instrument, reference)`**, both halves load-bearing. The
`peer_ssim2` pin set was appended; the `shipped_b` set is **retained, printed as
`incumbent`, and never a bar** — labelled biased, with the two measured reasons above.
Pre-2026-09-04 rows carry no `reference` field and a serde default supplies one, so those
rows are **byte-untouched**, which is what append-only requires.

Direction semantics are unchanged in form and sharper in meaning: **a candidate must
address at least the range ssim2 addresses** (`max`/`p95`/`reach`/`DR`/`frac<0` ≥ ssim2's;
`min`/`p5`/negtail `min`/negtail `p1` ≤ ssim2's). Every report prints both columns —
**`bar (vs ssim2)`** and **`incumbent (shipped B)`** — so "worse than the mentor" and
"worse than what shipped" can never be confused.

**The mentor's own values** (full f64, straight out of `dial_addressability::to_json`; no
percentile math was re-implemented beside the owner):

| axis | ssim2 (THE BAR) | shipped B (retired bar, now `incumbent`) |
|---|--:|--:|
| A1 `max` ≥ | **98.376644** | 99.98330778475787 |
| A2 `min` ≤ | **−55.354544** | 3.12950123756248 |
| A3 `p95` ≥ | **95.45929934999998** | 99.72170874183841 |
| A4 `p5` ≤ | **10.26332105** | 13.645032446453126 |
| A5 `reach` ≥ | **153.731188** | 96.85380654719539 |
| A6 `dynamic_range` ≥ | **85.19597829999998** | 86.07667629538528 |
| A7 negtail `min` ≤ | **−770.619744** | 2.516685884084839 |
| A8 negtail `p1` ≤ | **−187.13142578999998** | 3.981383254902343 |
| A9 negtail `frac_below_zero` ≥ | **1.0** | 0.0 |
| — mono | 0.99235757295044 | 0.9791570171375636 |
| — tied | 0.0 | 0.0 |
| — identity dial | **100.0** (all 38 refs) | 96.24115978721524 |
| — cells above identity | **0** of 4,424 | 266 of 4,424 |

**A9's `1.0` is DEFINITIONAL, not a discovery** — the probe's population was selected by
"ssim2 < 0", so the reference metric is below zero on all 2,000 rows by construction. A9
therefore asks for perfect sign agreement with ssim2 on an all-negative population. That
is the strictest honest reading of "address at least the range ssim2 addresses", and it is
stated here so nobody mistakes it for an empirical bar.

## 14.3 The re-pin is NOT a relaxation — measured, 70 against 9

Re-grading all 17 candidates under **both** pin sets, same measurements, different bars
(`--gaddr-reference shipped_b` reproduces the retired grading through the same owner, so
this is a measurement rather than a hand-derived table):

| candidate | fails under RETIRED shipped-B pins | fails under ACTIVE ssim2 pins | flipped FAIL→PASS | flipped PASS→FAIL |
|---|---|---|---|---|
| SHIPPED B | C3 C4 C5 C6 | A2 A4 A5 A7 A8 A9 C3 C4 C5 C6 | — | A2 A4 A5 A7 A8 A9 |
| Profile A `v47_strict_qat_native` | A1 A3 A4 A6 | A1 A2 A3 A4 A5 A6 A7 A8 A9 | — | A2 A5 A7 A8 A9 |
| Profile D `d_sdr_add156_dense_dial` (=ADD156) | A1 A3 A6 C5 | A1 A2 A3 A5 A7 A8 A9 C5 | A6 | A2 A5 A7 A8 A9 |
| `ctl_curera` | A1 A2 A3 A4 A5 A6 A7 A8 C3 C4 C5 C6 | A2 A4 A5 A6 A7 A8 A9 C3 C4 C5 C6 | A1 A3 | A9 |
| `ss_cur_rescored_clamped` | A1 A2 A3 A4 A5 A6 A7 A8 C3 C4 C5 C6 | A2 A4 A5 A6 A7 A8 A9 C3 C4 C5 C6 | A1 A3 | A9 |
| (a) `ss_cur_rescored_unclamped` | A1 A3 A4 A6 C3 C4 C5 C6 | A2 A4 A5 A6 A7 A8 A9 C3 C4 C5 C6 | A1 A3 | A2 A5 A7 A8 A9 |
| (b) `ss_..._unclamped_id100` | A4 A6 C2 C3 C4 | A2 A4 A5 A6 A7 A8 A9 C2 C3 C4 | — | A2 A5 A7 A8 A9 |
| (b′) `ne12_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| `ne30_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| `ne60_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| `ne120_ss_unc_id100` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| (b″) `ss_unc_id100_lowband` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| (c) `im26_..._unclamped_id100` | A2 A4 A5 A6 A7 A8 C2 C3 C4 | A2 A4 A5 A6 A7 A8 A9 C2 C3 C4 | — | A9 |
| `mix_unc_id100` | A2 A4 A5 A6 A7 A8 C2 C3 C4 | A2 A4 A5 A6 A7 A8 A9 C2 C3 C4 | — | A9 |
| `mix_unc_id100_lowband` | A4 A6 C2 | A2 A4 A5 A6 A7 A8 A9 C2 | — | A2 A5 A7 A8 A9 |
| ORACLE | A1 A3 A4 A6 C5 C6 | A1 A2 A4 A5 A6 A7 A8 A9 C5 C6 | A3 | A2 A5 A7 A8 A9 |
| ORACLE id100 | A1 A3 A4 A6 C5 C6 | A1 A2 A4 A5 A6 A7 A8 A9 C5 C6 | A3 | A2 A5 A7 A8 A9 |

**70 cells flipped PASS → FAIL; 9 flipped FAIL → PASS.** The re-pin moved the difficulty
from the ceiling to the **floor**, which is correct: ssim2 reaches −55.35 on this grid
where shipped B stops at +3.13, and its negative-tail probe is 100 % below zero against
B's 0 %. Shipped B itself goes from **0** regression fails (it *was* the bar) to **6**.

**Round-trip control:** grading `peer_ssim2` against its own freshly-registered pins reads
**SHIPPABLE — regression PASS + contract PASS, 15 / 15**, with every bar tied bit-exactly.
That is the check that the pins were derived through the owner and not re-typed.

**One honest limit of the reach reading.** A1/A3 are `≥`, so a dial that *overshoots* the
truth passes — shipped B's `max` 99.98 clears A1 while sitting 1.6 above the reference
metric. G-ADDR is an **addressability** gate (how much range is reachable), not a
calibration gate, and over-reach is not an addressability failure. The alternative
**calibration-referenced** reading (`|dial end − truth end| ≤ δ`, or low-band / whole-grid
MAE against the reference metric) that §11 registered is still a separate, unimplemented
user option; this re-pin implements the reach reading only.

## 14.4 Every candidate, re-graded against the mentor

Measured values (identical to §9 — only the bars moved):

| candidate | reach | min | max | p5 | p95 | DR | mono | tied | ntl min | ntl p1 | frac<0 | identity | above-id | REGRESSION | CONTRACT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| **ssim2 (THE BAR)** | 153.731 | −55.355 | 98.377 | 10.263 | 95.459 | 85.196 | .9924 | .0000 | −770.620 | −187.131 | 1.0000 | 100.0000 | 0 | *(is the bar)* | **PASS 6/6** |
| SHIPPED B | 96.854 | 3.130 | 99.983 | 13.645 | 99.722 | 86.077 | 0.9792 | 0.0000 | 2.517 | 3.981 | 0.0000 | 96.2412 | 266 | FAIL | FAIL |
| Profile A `v47_strict_qat_native` | 113.897 | -17.112 | 96.784 | 16.666 | 94.511 | 77.845 | 0.9782 | 0.0000 | -93.897 | -59.171 | 0.5575 | 97.6893 | 0 | FAIL | PASS |
| Profile D `d_sdr_add156_dense_dial` (= ADD156) | 108.252 | -12.204 | 96.049 | 9.517 | 95.284 | 85.767 | 0.9847 | 0.0000 | -100.000 | -87.306 | 0.8580 | 96.1157 | 0 | FAIL | FAIL |
| 944 flagship `c_sdr_purity944` | 88.365 | 11.635 | 100.000 | 33.189 | 100.000 | 66.811 | 0.9932 | 0.0376 | — | — | — | — | — | NOT MEASURABLE | INCOMPLETE |
| ADD156 raw lasso `add156_n156` | 1.225 | -0.165 | 1.061 | 0.502 | 0.973 | 0.471 | 1.0000 | 0.0000 | — | — | — | — | — | FAIL | INCOMPLETE |
| `ctl_curera` (era only) | 94.232 | 5.731 | 99.963 | 18.103 | 99.573 | 81.470 | 0.9799 | 0.0000 | 5.023 | 6.680 | 0.0000 | 95.8517 | 266 | FAIL | FAIL |
| `ss_cur_rescored_clamped` | 94.228 | 5.745 | 99.972 | 18.047 | 99.658 | 81.611 | 0.9794 | 0.0000 | 5.037 | 6.693 | 0.0000 | 96.4843 | 266 | FAIL | FAIL |
| (a) `ss_cur_rescored_unclamped` | 98.012 | 1.960 | 99.972 | 18.047 | 99.658 | 81.611 | 0.9794 | 0.0000 | 0.347 | 3.954 | 0.0000 | 96.4843 | 266 | FAIL | FAIL |
| (b) `ss_..._unclamped_id100` | 98.118 | 1.882 | 100.000 | 17.919 | 100.000 | 82.081 | 0.9808 | 0.0567 | 0.276 | 3.865 | 0.0000 | 100.0000 | 0 | FAIL | FAIL |
| (b′) `ne12_ss_unc_id100` | 102.695 | -2.695 | 100.000 | 18.633 | 100.000 | 81.367 | 0.9794 | 0.0567 | -4.374 | -0.519 | 0.0135 | 100.0000 | 0 | FAIL | FAIL |
| `ne30_ss_unc_id100` | 100.967 | -0.967 | 100.000 | 17.725 | 100.000 | 82.275 | 0.9801 | 0.0567 | -3.596 | 2.135 | 0.0030 | 100.0000 | 0 | FAIL | FAIL |
| `ne60_ss_unc_id100` | 102.123 | -2.123 | 100.000 | 18.168 | 100.000 | 81.832 | 0.9801 | 0.0567 | -2.650 | -1.304 | 0.0215 | 100.0000 | 0 | FAIL | FAIL |
| `ne120_ss_unc_id100` | 106.999 | -6.999 | 100.000 | 18.170 | 100.000 | 81.830 | 0.9810 | 0.0567 | -10.060 | -3.045 | 0.0210 | 100.0000 | 0 | FAIL | FAIL |
| (b″) `ss_unc_id100_lowband` | 109.390 | -9.390 | 100.000 | 18.640 | 100.000 | 81.360 | 0.9771 | 0.0567 | -11.676 | -6.451 | 0.0660 | 100.0000 | 0 | FAIL | FAIL |
| (c) `im26_..._unclamped_id100` | 88.088 | 11.912 | 100.000 | 22.106 | 100.000 | 77.894 | 0.9789 | 0.0567 | 11.642 | 12.418 | 0.0000 | 100.0000 | 0 | FAIL | FAIL |
| `mix_unc_id100` | 92.716 | 7.284 | 100.000 | 20.547 | 100.000 | 79.453 | 0.9787 | 0.0567 | 6.371 | 8.425 | 0.0000 | 100.0000 | 0 | FAIL | FAIL |
| `mix_unc_id100_lowband` | 101.300 | -1.300 | 100.000 | 21.465 | 100.000 | 78.535 | 0.9775 | 0.0567 | -3.154 | 1.088 | 0.0045 | 100.0000 | 0 | FAIL | FAIL |
| ORACLE (train-on-test) | 98.617 | -0.853 | 97.765 | 21.747 | 97.127 | 75.380 | 0.9801 | 0.0000 | -2.446 | 1.225 | 0.0040 | 95.2510 | 266 | FAIL | FAIL |
| ORACLE id100 | 98.864 | -0.866 | 97.998 | 21.517 | 97.261 | 75.743 | 0.9794 | 0.0000 | -2.454 | 1.205 | 0.0040 | 95.3393 | 266 | FAIL | FAIL |

Per-row verdict, **both tiers**:

| candidate | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 | A9 | C1 | C2 | C3 | C4 | C5 | C6 | pass/fail/NM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| **ssim2 (THE BAR)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 15/0/0 |
| SHIPPED B | ✓ | **✗** | ✓ | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 5/10/0 |
| Profile A `v47_strict_qat_native` | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 6/9/0 |
| Profile D `d_sdr_add156_dense_dial` (= ADD156) | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ | 7/8/0 |
| 944 flagship `c_sdr_purity944` | — | — | — | — | — | — | — | — | — | ✓ | ✓ | — | — | — | — | 2/0/13 |
| ADD156 raw lasso `add156_n156` | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | — | — | — | ✓ | ✓ | — | — | — | — | 3/5/7 |
| `ctl_curera` (era only) | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 4/11/0 |
| `ss_cur_rescored_clamped` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 4/11/0 |
| (a) `ss_cur_rescored_unclamped` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | **✗** | **✗** | **✗** | **✗** | 4/11/0 |
| (b) `ss_..._unclamped_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | 5/10/0 |
| (b′) `ne12_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| `ne30_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| `ne60_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| `ne120_ss_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| (b″) `ss_unc_id100_lowband` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| (c) `im26_..._unclamped_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | 5/10/0 |
| `mix_unc_id100` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | ✓ | ✓ | 5/10/0 |
| `mix_unc_id100_lowband` | ✓ | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | **✗** | ✓ | ✓ | ✓ | ✓ | 7/8/0 |
| ORACLE (train-on-test) | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | **✗** | **✗** | 5/10/0 |
| ORACLE id100 | **✗** | **✗** | ✓ | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | ✓ | ✓ | ✓ | ✓ | **✗** | **✗** | 5/10/0 |

Distance to the mentor on each regression axis (negative = short of the bar):

| axis | dir | ssim2 bar | (b″) lowband | gap | Profile D | gap | Profile A | gap | shipped B | gap |
|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A1 | ≥ | 98.377 | 100.000 | OK | 96.049 | -2.328 | 96.784 | -1.592 | 99.983 | OK |
| A2 | ≤ | -55.355 | -9.390 | -45.964 | -12.204 | -43.151 | -17.112 | -38.242 | 3.130 | -58.484 |
| A3 | ≥ | 95.459 | 100.000 | OK | 95.284 | -0.175 | 94.511 | -0.948 | 99.722 | OK |
| A4 | ≤ | 10.263 | 18.640 | -8.377 | 9.517 | OK | 16.666 | -6.403 | 13.645 | -3.382 |
| A5 | ≥ | 153.731 | 109.390 | -44.341 | 108.252 | -45.479 | 113.897 | -39.835 | 96.854 | -56.877 |
| A6 | ≥ | 85.196 | 81.360 | -3.836 | 85.767 | OK | 77.845 | -7.351 | 86.077 | OK |
| A7 | ≤ | -770.620 | -11.676 | -758.944 | -100.000 | -670.620 | -93.897 | -676.723 | 2.517 | -773.136 |
| A8 | ≤ | -187.131 | -6.451 | -180.680 | -87.306 | -99.825 | -59.171 | -127.961 | 3.981 | -191.113 |
| A9 | ≥ | 1.000 | 0.066 | -0.934 | 0.858 | -0.142 | 0.557 | -0.443 | 0.000 | -1.000 |

## 14.5 Is there a `B dial-era v2`? — **NO**, and for better reasons than before

**No candidate passes the re-pinned gate, so none is proposed for the board and no
discussion set is published.** But the *blocking axes changed completely*, and the new ones
are all real:

- Under the retired pins the leading arms failed **A4 / A6 / C2** — two of which §8 had
  already shown were bars the reference metric itself could not meet.
- Under the mentor's pins they fail **A2, A4, A5, A6, A7, A8, A9 and C2** — the **floor and
  spread** axes, every one of them a genuine shortfall against a scorer that demonstrably
  reaches further.

The magnitude is not marginal. The best arm, `ss_unc_id100_lowband`, is **45.96 short on
A2**, **44.34 short on A5**, **758.94 short on A7**, **180.68 short on A8** and reaches
**6.6 %** of the negative tail where the mentor reaches **100 %**. Re-splining B cannot
close that: the whole B-lineage spread on A9 across every arm built in this lane is
0.0000 → 0.0660.

**C2 ⊻ C6 still blocks B, and it is still a WEIGHTS defect.** §10.3's proof is untouched by
the re-pin: 266 of 4,424 dial-grid cells (6.01 %) carry a raw prediction above the identity
vector's, so a monotone spline must either cap them (C2 fails, `tied` 0.0567 — measured on
every `id100` arm) or let them out-score a perfect copy (C5/C6 fail). No output spline
satisfies both. The fix is in the weights.

## 14.6 Then which dial should be the default? — a proposal for the user

Measured on the same instrument, same probes, same gate:

| | contract | regression | rank (CID22 / \|KonJND\| / AIC-3 / TID / KADID) |
|---|---|---|---|
| **Profile A** `v47_strict_qat_native` | **PASS 6/6** | 0/9 | 0.86606 / 0.44313 / 0.77039 / 0.79264 / 0.79378 |
| **Profile D** `d_sdr_add156_dense_dial` (= ADD156) | 5/6 — **only C5** | **2/9** (A4, A6) | 0.86338 / **0.53319** / **0.77734** / **0.82348** / **0.80822** |
| shipped B | 2/6 | 3/9 | **0.88212** / 0.51938 / 0.76501 / 0.77852 / 0.80847 |

**Profile A is the only bake measured anywhere in this lane that passes the entire CONTRACT
tier** — identity 97.6893 in band, 0 cells above identity, and a negative tail that works
(55.75 % of the all-negative probe below zero, min −93.90). If a contract-passing dial
default has to be named **today**, it is A.

**But D is the better candidate, and its single contract failure is one dial edit away —
provably without B's either/or.** D fails only **C5**: its identity dial is 96.1157, which
is **1.384 below** the `[97.5, 100]` band. Two measured facts make that fixable in a way
B's is not:

1. **D's grid `max` is 96.049 — strictly BELOW its own identity of 96.1157.** Every one of
   the 4,424 cells already scores below a perfect copy (`above-identity = 0`). So the 266
   raw inversions that force B's C2 ⊻ C6 either/or **do not exist for D**, and anchoring
   D's identity at 100 cannot pile cells at the cap the way it does for every `id100` arm
   in §14.4 (`tied` 0.0567).
2. **D is already the closest thing to the mentor's floor that exists.** A9 `0.858` against
   the bar's `1.000` — a gap of 0.142, against 0.934 for the best B-lineage arm and 1.000
   for shipped B. On A8 it is 99.8 short where the best arm is 180.7 short.

D also beats A on **four of five** rank corpora (and is 0.0027 behind on CID22), and passes
**A4 and A6**, the only two regression axes any shipped bake passes.

**Proposal, for the user to decide:**

- **(i)** Name **Profile A** the contract-passing dial default *only if* one must be named
  before any new build. It is contract-clean today and nothing else is.
- **(ii)** Otherwise, build **`D-id100`** first — Profile D re-anchored with identity at
  100 — and re-gate it. Fact 1 above says the C2 cost that blocks the B lineage cannot
  arise. ~~**REGISTERED, NOT RUN:**~~ **RUN 2026-09-04 — see
  [`d_id100_2026-09-04.md`](d_id100_2026-09-04.md).** Fact 1 held: `tied` stays 0.0000 and
  `above-identity` stays 0. **`D-id100` reads CONTRACT 6/6 + REGRESSION 4/9** (A1 96.049 →
  99.380, A3 95.284 → 95.518, A4, A6) and **`D-id100-negrich` reads CONTRACT 6/6 +
  REGRESSION 7/9** — only A7 and A9 fail, both for measured structural reasons. Two of this
  bullet's premises were **overturned by measurement**: the pin is **not** deliverable by
  the fit (eight real re-fits folding identity into the Gram at 0.1 %–20 % of the data mass
  move the identity dial +0.0055 and cost −0.0125 CID22 — the identity Gram's `S`/`s`/`q`
  are all exactly zero, so it is nearly the bias offset that is a provable no-op), and it
  therefore **is** a re-spline: the winning arms carry weights **byte-identical** to shipped
  D (sha `330d8c09…` after stripping spline + repro), with zero pair-order flips on all 14
  corpora and `product_composite` byte-identical.
- **(iii)** shipped **B stays the rank leader on CID22** (0.88212) and is not displaced by
  either on that axis. The dial and the ranker are not the same decision.

## 14.7 What was added to the owners

`bake_verdict` gained three flags, all of which existed to make this measurable rather than
to make it pass:

- **`--negtail-peer-scores` / `--identity-peer-scores`** (`entry⇥pred`) — a reference
  metric has no bake, so before this its floor axes could not be measured at all, and a
  peer run silently reported *the peer's* grid reach beside *the bake's* tail depth under
  one headline, counting "cells above identity" across two different scorers. Peer mode is
  now **all-or-nothing per axis**: an unsupplied probe is NOT MEASURED, never filled in
  from `--bake`. Refused loudly.
- **`--gaddr-json <path>`** — the G-ADDR block alone at full f64, stamped with which scorer
  it describes. Peer-safe (unlike `--full-json`/`--fulleval`, which stay refused in peer
  mode). This is how a pin set is derived without duplicating percentile/tail math outside
  its owner, and it also closes §11's registered "the JSON is emitted; nothing reads it".
- **`--gaddr-reference <name>`** — grade against a named pin set. Default is
  `ACTIVE_REFERENCE` = `peer_ssim2`; `shipped_b` reproduces the retired grading through the
  same comparators, which is what makes §14.3 a measurement.

`scripts/dialgate_arms.sh` gained a **`score`** mode so re-grading an existing bake and
building a new arm end in the same `grade` function — a re-grade after a bar change cannot
accidentally be a different measurement from the build-time one.

**Tests** (`cargo test -p zensim-validate --lib dial_addressability`, 17 pass): both
directions of disagreement are covered by fixtures built from real measured values —
ssim2's own values fail the retired B bars on A1/A3/A6, B's own values fail the ssim2 bars
on A2/A5; A4 specifically is pinned to stop rewarding the low-band bias while still
accepting Profile D's real 9.52; the mentor's contract PASS is pinned; and the retired
shipped-B rows are asserted to survive. **Negative control:** deleting the `peer_ssim2`
rows from the registry fails **12 of the 17**.

## 14.8 Does the mentor itself pass the CONTRACT tier? — **YES, all six**

The question "what does *as good as the mentor* mean at the ends" has a measured answer.
`peer_ssim2`, graded by the same instrument on the same grid and probes:

| id | axis | bar | ssim2 measured | shipped B | verdict |
|---|---|--:|--:|--:|:--:|
| C1 | monotonicity | ≥ 0.93 | **0.99235757295044** | 0.9791570171375636 | ✓ |
| C2 | flat/clamp dead-zone | ≤ 0.05 | **0.0** | 0.0 | ✓ |
| C3 | negative values WORK (`frac<0` on an all-negative probe) | > 0 | **1.0** | 0.0 | ✓ |
| C4 | deepest probe dial < 0 | < 0 | **−770.619744** | +2.516685884084839 | ✓ |
| C5 | `dial(ref==dist)` ∈ [97.5, 100] | 0 rows outside | **100.000000** (min = med = max, n=38) | 96.24115978721524 | ✓ |
| C6 | cells out-scoring a perfect copy | 0 | **0** of 4,424 | 266 of 4,424 | ✓ |

Three notes on how those were obtained, because the identity row is the one that could
have been assumed instead of measured:

- **C5 was MEASURED**, not taken from SSIMULACRA2's definition:
  `zenmetrics batch --metric ssim2` (the imazen CPU implementation) over the 38 dial-grid
  references paired **with themselves** reads exactly `100.000000` on all 38 — one distinct
  value, min = median = max. The grid truth is `ssim2_gpu`; identity is 100 by
  SSIMULACRA2's construction, which is why the cross-implementation question does not arise
  at this point.
- **C6 follows from two registry rows** with no extra measurement: the mentor's grid `max`
  (98.3766) is below its identity (100.0), so nothing can out-score a perfect copy. Pinned
  as a test.
- **C1 is the quiet one.** ssim2 is *more* monotone on this grid than the shipped dial
  (0.9924 vs 0.9792) with the same zero tied rate — so C1/C2 are not rows the mentor needed
  help with, and a candidate that beats ssim2 on monotonicity has to clear 0.9924, not the
  0.93 floor.

So the mentor is clean at both ends, and the four contract rows shipped B fails are
**not** rows that are unreachable in principle — the reference metric meets all of them,
and so does Profile A.

## 14.9 Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict
D=/mnt/v/output/zensim/dialgate-2026-09-04
G=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet

# the mentor's pins (this is how they were derived; --gaddr-json carries full f64)
./target/release/bake_verdict --bake <any 372 bake> --corpora cid22 --dial-grid $G \
    --negtail-probe  $D/negtail_probe_372_2026-09-04.parquet \
    --identity-probe $D/identity_probe_372_2026-09-04.parquet \
    --dial-peer-scores     peer_ssim2=/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv \
    --negtail-peer-scores  peer_ssim2=$D/repin/negtail_peer_ssim2.tsv \
    --identity-peer-scores peer_ssim2=$D/repin/identity_peer_ssim2.tsv \
    --gaddr-json $D/repin/peer_ssim2_gaddr.json --output $D/repin/peer_ssim2_gaddr.md

# ssim2's identity, measured with the imazen CPU implementation
~/work/zen/zenmetrics/target/release/zenmetrics batch --metric ssim2 \
    --pairs $D/build/identity_pairs.tsv --output $D/repin/identity_ssim2_cpu.tsv

# re-grade any candidate (score mode = no rebuild); add --gaddr-reference shipped_b
# to reproduce the retired grading
scripts/dialgate_arms.sh score <label> <bake.bin> [372|720|944]
```

**As-run artifacts** (block storage, not git): `$D/repin/` (peer cell tables, the mentor's
G-ADDR json + markdown, the ssim2 identity measurement), `$D/arms/gaddr_R_*.json` (every
candidate under the active pins), `$D/arms/bpins/gaddr_R_*.json` (every candidate under the
retired pins), `$D/arms/verdict_R_*.json` (full verdicts with the rank panel).

---

## 15. Board coverage — every fair cell graded, and the one instrument that is missing

The fair-gauntlet record (`benchmarks/fair_gauntlet_2026-09-04.md` §4) closed with
*"the board still does not badge NOT-SHIPPABLE on a CONTRACT failure, because no cell has a
CONTRACT-tier measurement."* That is now false: **97 of 97 fair cells were graded through
the owner, under BOTH pin sets, and 96 carry the verdict on the board.** 47 of them fail a
contract row and are NOT SHIPPABLE by measurement.

### 15.1 What was run

`bake_verdict --dial-grid <the cell's OWN grid> --negtail-probe … [--identity-probe …]
--gaddr-json …`, twice per cell — once under the ACTIVE `peer_ssim2` pins and once with
`--gaddr-reference shipped_b` — for all 97 VERIFIED-FAIR + FAIR-NOTED rows of
`benchmarks/fairness_tiers_2026-09-04.pointer.md`. Ensembles went through `--ensemble` with
their `member_names`; the 3 canonical-372 cells got both registered probes. Nothing was
re-derived anywhere: the board renders the owner's own `checks` array.

As-run: `/mnt/v/output/zensim/gaddr-board-2026-09-04/{active,bpins,verdict,logs}/`.

### 15.2 The result

| | n | CONTRACT 6/6 | ≥1 CONTRACT row FAIL | INCOMPLETE (no fail, ≥1 unmeasured) |
|---|--:|--:|--:|--:|
| VERIFIED-FAIR | 42 | 0 | **17** | 25 |
| FAIR-NOTED | 55 | **1** | **30** | 24 |
| **all fair** | **97** | **1** | **47** | 49 |

Per contract row, over all 97:

| row | what | fail | pass | NOT MEASURED |
|---|---|--:|--:|--:|
| C1 | monotonicity ≥ 0.93 | 1 | 96 | 0 |
| C2 | flat/clamp dead-zone ≤ 0.05 | **23** | 74 | 0 |
| C3 | negative values WORK (`frac<0` > 0) | **39** | 45 | 13 |
| C4 | deepest probe dial < 0 | **39** | 45 | 13 |
| C5 | `dial(ref==dist)` ∈ [97.5, 100] | 1 | 2 | **94** |
| C6 | nothing out-scores a perfect copy | 2 | 1 | **94** |

**The shippable set under the user's rule is ONE cell: `v47_strict_QAT_native@cur372`**
(Profile A), the only bake with all six contract rows measured AND passing — which
independently reproduces §14.6's claim from a different lane, a different invocation and
a different binary. Its regression tier is 0 of 9 under the mentor's pins, so it is not
*shippable*; it is the only thing that is *contract-clean*.

Two more independent reproductions fell out of the both-pin-sets read, both exact:
shipped **B** reads **9/9 PASS** under the retired `shipped_b` pins and **3 pass / 6 fail**
(A2, A4, A5, A7, A8, A9) under `peer_ssim2` — §14.3's "0 regression fails → 6" — and
**BHdr** and **v47** both fail exactly A1/A3/A6 under the retired pins, the three axes
`_schema.reference_sets` calls BIASED.

**C3/C4 are the headline.** 39 of the 84 cells whose tail could be measured never emit a
value below zero on a probe where *every row's reference metric is negative* — the shipped
dial's own defect (§5), reproduced across most of the 944 population. C2 fails on 23 more.

### 15.3 The instrument gap, and why it was not papered over

`bake_verdict` scores a probe only when the probe's column count equals the bake's caller
width, so the two registered 372-wide probes reach **3** of the 97 cells. Two 944-wide
negative-tail probes were therefore cut **by the registered rule, from in-era sources**,
and they are what makes C3/C4 measurable on 81 more cells:

| probe | rows | source (same `build_commit` as the grid it serves) | sha256 |
|---|--:|---|---|
| `negtail_probe_944_2026-08-01era.parquet` | 2,000 | `kadis-944-2026-08-01/kadis_negrich_944.parquet`, `ec3bdd6a…` — the same commit that built `dial_grid_944col_2026-08-01` | `42f93e61c6e5f562…` |
| `negtail_probe_944_era2r4_foldapp2.parquet` | 2,000 | `ext944-era2r4-2026-09-01/foldapp2_views/ext_kadis.parquet`, `75c09149…` | `b73ce10655cb1c16…` |

Both live in `/mnt/v/output/zensim/dialgate-2026-09-04/probes944/`. Truth spans
−702.313191 … −0.123406 and −1.0 … −0.000372; every row negative in both, which is the
probe's defining property. **The cut has a committed owner now** —
`scripts/cut_gaddr_negtail_probe.py` — and a control: re-cutting the ORIGINAL 372 source
with it reproduces the stored `negtail_probe_372_2026-09-04.parquet` truth column
**exactly**, 2,000 of 2,000 rows, max abs diff 0.0. So these are the same instrument at
another width, not a lookalike.

**They are deliberately NOT in the floor registry.** No reference scorer has been measured
on them, so registering them would either invent a bar or silently borrow `shipped_b`'s
bar from a different instrument. Unregistered means A7–A9 read NOT MEASURED on them while
C3/C4 — which have absolute bars and take no reference — read normally. That is the
registry's own rule ("a bar you can dodge by choosing a friendlier instrument is not a
bar") applied to instruments this lane created.

**Three gaps stay open, each for a measured reason:**

1. **No 944 IDENTITY probe exists, and one cannot be faked.** At 372 the identity feature
   vector is the ZERO vector (§4), which makes `dial(ref==dist)` a scalar property of the
   bake. **MEASURED 2026-09-04: that does NOT extend to 944.** Extracting the 38
   `ref == dist` pairs through `sdr944_extract` gives **190 of 944 slots non-zero**, and
   they vary per image (row-to-row spread 0.594 on those slots, max |value| 1.0). So a 944
   identity read is an extractor-era-dependent measurement, not an algebraic constant, and
   no in-era extraction exists for any of the three 944 eras on the board. C5/C6 are
   therefore NOT MEASURED on 94 cells — never zero, never a pass. Closing it needs one
   38-pair extraction per era at that era's build commit (`ec3bdd6a`, `75c09149`, and the
   POOLS root), which is seconds of compute behind an era-matched extractor build.
2. **The POOLS era has no negative-truth rows at all.** `wlin7-pools944-2026-08-30`'s
   `tbig`/`tsafesyn` legs are clamped at `human_score >= 0` (measured: min 0.0, 0 negative
   rows in 319,237), so no in-era negative-tail probe can be cut for the 9 cells on
   `dial_grid_944col_POOLS_2026-08-30`. C3/C4 stay NOT MEASURED there.
3. **The REGRESSION tier is NOT MEASURABLE on 94 of 97 cells** — their dial grids
   (`dial_grid_944col_2026-08-01`, `…_POOLS_2026-08-30`, `…_foldapp2_2026-09-01`) are not
   in the floor registry, because the mentor has never been measured on them. Only the 3
   canonical-372 cells have a graded regression tier. **A `peer_ssim2` row for the 944
   grids is the single highest-value registry append available**, and it needs one
   `zenmetrics batch --metric ssim2` pass over each 944 grid's cells plus a
   `--dial-peer-scores` derivation run — the exact recipe §14.9 already documents.

### 15.4 Two defects found on the way

**(a) `bake_verdict` has TWO percentile implementations and they disagree by 1 ULP.**
`bake_verdict::percentile` (bake_verdict.rs:3213) interpolates
`sorted[lo]*(1-frac) + sorted[hi]*frac`; `dial_addressability::pct`
(dial_addressability.rs:244) interpolates `sorted[lo] + frac*(sorted[hi]-sorted[lo])`.
Same sorted vector, same process, mathematically identical, numerically not: `--full-json`
wrote `dial.p5 = 27.289567929893384` while `--gaddr-json` wrote
`measured.grid.p5 = 27.28956792989338` for the same cell. Every stored board cell carries
the first form; **A4 and A6 are graded on the second**. This is the same hazard class as
§6's `serde_json` one-ULP bar failure, and it is a no-duplication-rule violation:
`percentile` should call the library's `pct`. **NOT fixed here** — collapsing them moves
`pct`, which would move the committed registry bars and the tests pinned to them, so it is
the gate owner's call, not a graft's drive-by. The graft gate allows ≤ 4 ULP on the
interpolated dial scalars *only*, records every slack it used in
`dial_gaddr_source.dial_scalar_ulp_slack`, and still refuses anything larger (a read on a
different grid differs by orders of magnitude, and one did — see (b)).

**(b) The same-grid gate caught two real mismatches, which is the point of having it.**
`HYA_w084` refused at equal ensemble weights (`mono_pct` 0.99404 vs the board's 0.99298);
re-running at **0.84 / 0.16** reproduced the board's dial **byte-exactly** and 0.16 / 0.84
did not — so the ensemble's weights are now a measured fact rather than an assumption, and
the other 14 ensembles' equal weights are confirmed the same way (they grafted on the first
try, which only happens if the weights match). `ebothg_m504` refused outright
(`mono_pct` 0.7754 board vs 0.9273 fresh) — it is the known wrong-root cell, its board row
was read on an instrument this lane cannot identify, and it is the **one** fair cell with
no G-ADDR block on the board. Its own G-ADDR read exists as an as-run artifact and is
**not** citable as that row's behaviour.

### 15.5 What the board does now

`gauntlet.py` prefers the cell's grafted `dial.addressability` over the six axes it can
re-derive from stored dial scalars: the G-ADDR column reads `pass/15` instead of `pass/6`,
the tooltip prints both tiers with the owner's headline, per-row bars, states, the
shipped-B context column and the NOT-MEASURED reason verbatim, and a red **NOT SHIPPABLE**
badge rides the bake name wherever a CONTRACT row measurably FAILED — 46 cells on the fair
board. **An INCOMPLETE contract never draws the badge**: an unmeasured row is not a fail.
Grafting is `promote_fulleval.py --graft-gaddr` (sha-gated on the scorer bake's own bytes,
same-grid gated on every stored dial scalar, provenance in `dial_gaddr_source`); nothing
was hand-edited.

Boards regenerated + gated (`scripts/v_next/gauntlet_gates.sh`, both PASS):
`summer_gauntlet_fair.html` **8,205,798 B (7.8 MiB, under the 12 MB cap)**, 97 rows;
`summer_gauntlet.html` **21,003,539 B (20.0 MiB, over cap, reported not trimmed)**,
433 rows.

### 15.6 Reproduction

```sh
# grade one cell, both pin sets (the runner does all 97)
BV=target/release/bake_verdict
D=/mnt/v/output/zensim/dialgate-2026-09-04
$BV --bake <bake.bin> [--ensemble a.bin,b.bin] --dial-grid <the cell's own grid> \
    --regime 944 --corpora cid22 \
    --negtail-probe $D/probes944/negtail_probe_944_2026-08-01era.parquet \
    --gaddr-json out.json                      # add --gaddr-reference shipped_b for the retired pins
# graft it onto the board cell (refuses on a sha or same-grid mismatch)
python3 scripts/promote_fulleval.py --graft-into <cell>.fulleval.json --graft-gaddr out.json
# cut a negative-tail probe at another width, by the registered rule
python3 scripts/cut_gaddr_negtail_probe.py <src.parquet> <neg-truth col> <out.parquet>
```
