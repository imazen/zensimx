# FEATURE REVISION 3 — the stable SSIM moments, and what a refit needs

Sibling of [`PLAN_FEATURE_REV2_2026-09-05.md`](PLAN_FEATURE_REV2_2026-09-05.md),
same lane, same gates. Executes [issue #61](https://github.com/imazen/zensimx/issues/61).

Source of the defect: [`benchmarks/nonmax_diagnosis_2026-09-08.md`](../benchmarks/nonmax_diagnosis_2026-09-08.md).
Source of the mechanism and the integration record:
[`benchmarks/stable_ssim_kernel_2026-09-08.md`](../benchmarks/stable_ssim_kernel_2026-09-08.md).

---

## 0. What "revision 3" is

> **Framing (user, 2026-09-09):** zensim is *"a speedy and consistent dial and
> steering metric with useful spatial steering across both sdr and hdr"*, and
> *"bounded error is fine, speed above minor flaws"*. Revision 3 as first
> integrated (an exact f64 second pass) restored locality but cost +27-87% of
> extraction and flipped zensim from beating fast-ssim2 to losing to it. It was
> replaced, before any data was extracted at it, by the FUSED form below:
> bounded error, no second traversal. The exact kernel survives only as the
> reference the bounds are measured against.


Revision 2 plus ONE change: the per-pixel v1 SSIM dissimilarity is formed from
a DIRECT error moment. The existing H pass already accumulated three of the four
moments the stable form needs (`Σa`, `Σb`, `Σ(a²+b²)`); its fourth plane, `Σab`,
has no consumer under revision 3, so it now carries `Σ(a−b)²` instead (the same
two FMAs on `a−b`). The V pass forms
`loss + (1−loss)·E_err/(var1+var2+C2)` from the same four f32 planes it always
V-blurred. Nothing is traversed twice.

Revision 3 INHERITS revision 2 (`SsimLumaForm::Clamp`, `paired_global_contrast`,
`NestedSqrt`, `PureRust`, `REV2_HFGAIN`). It is not an alternative to revision 2;
it is revision 2 with the SSIM signal computed accurately.

### Why the old form loses precision

The shipped signal recovers variance and covariance by SUBTRACTION from f32 raw
moments:

```text
var1+var2 = E[a²+b²] − E[a]² − E[b]²      (sigma_sq − mu1² − mu2²)
cov       = E[ab]    − E[a]·E[b]          (sigma12 − mu1·mu2)
d_raw     = 1 − num_m · (2·cov + C2) / (var1 + var2 + C2)
```

Two defects compound, both upstream of that expression — which is why clamping
the result cannot recover anything:

1. **Catastrophic cancellation.** On flat high-value content — paper white in a
   document, a screenshot background — `E[a²]` is ~0.25–1.0 while the true
   variance is ~1e-6. Subtracting two nearly equal f32 quantities annihilates
   most of the significant bits, so the RELATIVE error of the recovered variance
   goes to O(1). Near identity both the numerator and denominator are dominated
   by their own cancellation residue.
2. **Path-dependent running-sum drift.** The box blur is a sliding recurrence
   (`sum = sum + entering − leaving`) in f32, and that accumulation is
   history-dependent. Two windows containing IDENTICAL samples, reached by
   different add/remove histories, produce different sums. This is what breaks
   locality: a pixel whose own window contains no changed sample still moves,
   because the running sum that reached it passed through the changed region.

### What the reference kernel does (`ssim_form::stable_ssim_plane`, not served)

f64 products and f64 window accumulation; a FOURTH moment `(a−b)²` so the error
variance is accumulated directly rather than recovered as `var1 + var2 − 2·cov`;
a guard that skips an add/remove pair that cancels, so an exact no-op window
acquires no drift; one reflect-101 box; one rounding to f32 at the end.

Algebraically the same form —
`1 − num_m·(2cov+C2)/(var1+var2+C2) ≡ loss + (1−loss)·ve/(var1+var2+C2)` —
verified against an independent direct-window f64 reference.

---

## 1. Owner map

| owner | what revision 3 changes there |
|---|---|
| `ssim_form::stable_ssim_plane` | the exact f64 REFERENCE kernel. Not on the served path; the bounded-error tests measure against it. |
| `ssim_form::SsimSplats16/8::direct`, `ssim_direct*` | NEW. The direct-error form in f32 SIMD/scalar, mirroring the reference `finalize`. |
| `blur::fused_blur_h_ssim` (+`ssim3`, all tiers) | under revision 3 the fourth plane is `Σ(a−b)²`, read once per call from `active_revision`, loop-unswitched in every tier. `sigma12` keeps its name; its meaning is revision-dependent and documented. |
| `ssim_form::check_route` | NEW. Explicit refusal of routes the revision does not serve. |
| `fused::fused_vblur_features_ssim` | under revision 3 forms `d_raw` with `ssim_direct*` from the four V-blurred planes; the f64 second pass and its thread-local scratch are gone. Legacy `ssim_dissim*` is bypassed, not modified. |
| `simd_ops::ssim_signal_inline_both` / `_mask` / `_iw` | NEW. Weighted pools over the RETAINED signal. Weights, `.max(0)`, tiers and accumulation order copied verbatim from the legacy trio; the legacy trio is byte-for-byte unchanged. |
| `streaming::process_strip_channel` | retains the band's inner signal before the activity work reuses `temp_blur`, and pools from it. Under revision 3 the two sigma V-blurs are NOT run — running them would hand the weighted pools a different `d_raw` than the basic pools already consumed. |
| `feature_v2::fold_v1_one_band` | same retention/pooling for the folded band replay. |
| `attribution::process_channel_banded` | unchanged: it consumes `fused_vblur_features_ssim`'s `sd` side-output, so it inherits the corrected signal. |
| `metric/bake.rs::check_pixel_revision` | compares the REVISION, not only the luminance form it selects. |
| `feature_defs::REV_SSIM_STABLE` | the `v1ssimstable` era entry. `Proposed`: shipped bytes are unchanged. |
| `zensim-validate::feature_set` | `ADMITTED_FORMULA_REVISIONS` — one list, replacing three `matches!(n, 1 | 2)` spellings. |

**Not changed, deliberately:** `sigma12` keeps covariance semantics (real
consumers remain); `fast-ssim2` keeps its own arithmetic and stays the
independent SSIMULACRA2 judge; `SHIPPED_REVISION` stays `Rev1`.

---

## 2. Route contract

Revision 3's support is exactly ONE reflect-101 box of `blur_radius`. Therefore
`blur_passes != 1` — which selects `process_strip_channel`'s separate
blur+reduce fallback, whose halo is `passes * radius` — is NOT served.
`ssim_form::check_route` returns `ZensimError::ModelForwardFailed` naming both
the route and the revision, from every fallible entry that builds a
`ZensimConfig`: profile-based entries in `metric`, `attribution`, `diffmap` and
`corruption_head` (mapped to `NotServable` there), plus the two `pub fn`s that
take a caller-built config. Not a panic; not a silent fall-through to legacy
arithmetic.

**Observed while auditing that gate, NOT acted on:** the `blur_passes != 1`
fallback calls `box_blur_1pass_into` — one pass — while `process_scale_bands`
sizes its halo at `passes * radius`. Whatever `blur_passes = 2` or `3` means in
the streaming path today, it is not "two or three box passes". A pre-existing
question for whoever owns that route; this lane refuses it rather than
inheriting it.

---

## 3. Measured evidence

Integrated banded walk, retained `AttrScaleRetention::sd` at four scales and
three channels; deterministic document/screenshot fixture at 192×288 (scale 0
spans three 128-row strips). A rectangle of the distorted image is replaced with
REFERENCE pixels; every signal whose own window contains no changed sample must
be unchanged.

| revision | out-of-support signals moved | peak abs delta |
|---|---:|---:|
| 1 (shipped) | 8,293 | 4.886e-4 |
| 3, exact f64 second pass (superseded) | 0 (serial and rayon) | 0 |
| 3, fused (served) | 11,163 (serial and rayon) | **4.277e-6** (bound 2e-5) |

The fused form is BOUNDED, not exact: its f32 sliding sums stay
path-dependent, so out-of-support signals still move, but the direct error
moment removes the cancellation and the peak drops ~114×. Retained planes vs
the whole-plane exact f64 kernel: 220,320 signals, worst |delta| **3.150e-4**
(bound 1e-3). All-equal windows: worst residue **3.689e-6** (bound 1e-5).

Cost, single thread, 2048², paired A/B on one binary: `fold944_full` 263.3 ms
(rev 1) → 259.9 ms (fused rev 3), anchor −1.0% — parity; the superseded exact
form measured 336.5 ms (+27%). `fold944_full` at revision 3 is under
`fast_ssim2` (288.6 ms) again. Full table: benchmark record, "Fusion".

The revision-1 row is a committed test of its own, so the revision-3 row cannot
pass on an inert fixture.

---

## 4. Blast radius

`v1ssimstable` is registered against the same signals as `v1ssimcap` (F4): every
SSIM-derived slot, because both change the same per-pixel `d`. In the 372
layout that is **132** slots — basic 36, peaks 24, masked 36, IW 36.

Unlike `v1ssimcap`, revision 3 is NOT bit-identical to revision 1 on ordinary
content. Clamp only bites where `D² > 1`; precision bites everywhere. **Every
one of those 132 columns moves on every image.**

`rev3_moves_exactly_the_registered_slots` checks this by cross-revision
re-extraction, in both directions: a slot that moves and is not registered
fails, and a registered slot that does not move fails. The second direction is
the one that catches a consumer still reading legacy moments.

---

## 5. Recalculation manifest

Same shape and the same rules as revision 2's §5, with one difference that
changes the ordering: **revision 3 moves its 132 columns on all content**, so no
table can be carried forward on an "it does not move here" argument.

Every table gets a `_MANIFEST.json` carrying `build_commit`, `formula_revision:
3`, `feature_set_id` (the `v1ssimstable` era token), decoder era per format, and
per-file sha256; LAN store + Tower mirror; a `DATA_PROVENANCE.md` and
`docs/DATASET_HISTORY.md` row.

Reproducible extraction is TWO things, and the second is the one that bites.

**1. The arithmetic switch is the environment:**

```sh
ZENSIM_FORMULA_REV=3 <the existing extractor invocation, unchanged>
```

`ZENSIM_FORMULA_REV` is read once per process into a `OnceLock`, so it cannot be
changed mid-run and cannot be set per-request; one process extracts one era.
`=1`, `=2` and `=3` are the same byte length on purpose — an environment block's
size has moved a 2304² timing ~10 % here
(`benchmarks/era2_perf_break_2026-08-31.md` §22.5), so an A/B that varies the
value must not vary the length.

**2. The DECLARATION is not derived from it.** VERIFIED by reading the
producers, 2026-09-09: nothing stamps `formula_revision` into a manifest from
the active revision. `scripts/v_next/build_corruption_corpus.py` takes a
hand-authored `--producer-json`, `require`s
`producer["formula_revision"] == 1 and producer["root_form"] == "libm"`, and
only then sets `ZENSIM_FORMULA_REV=1` itself — the declaration and the
environment are cross-checked against each other, not inferred one from the
other. `zensim_mlp_train` likewise stamps `zentrain.formula_revision` into the
bake from the TABLE's admitted revision, not from the process it runs in.

Consequences for a revision-3 wave, all of them actionable:

- That corpus script is pinned to revision 1 **by design** — it reproduces a
  registered revision-1 corpus. A revision-3 corpus needs its OWN registered
  producer declaration and its own script entry point. Do not edit the `1` to a
  `3` in place: that silently re-labels a registered corpus recipe.
- Every new table's `_MANIFEST.json` / `PRODUCER.json` must carry
  `formula_revision: 3` explicitly, written by the operator, and the extraction
  process must be pinned to match. Admission enforces agreement
  (`feature_set::ADMITTED_FORMULA_REVISIONS` now accepts 3, and a declared
  revision conflicting with the registered era is an error, not a default) —
  but agreement between two things you wrote is not the same as derivation, so
  a wave that forgets the stamp produces revision-3 bytes labelled revision 1
  and admission cannot tell.
- The cheapest guard is the one this lane already added: extract a fixture at
  both revisions and diff. If the 132 registered columns did not move, the
  process was not pinned.

Fleet: zenfleet only (`zenfleet-ctl declare` + `zenmetrics jobexec`), LAN +
tower only. Bake the revision-3 extractor into the canonical worker image under
a NEW TAG, never a new package name. First-cell gate before scaling.

**Admission will reject a mislabelled table**, in both directions:
`feature_set::ADMITTED_FORMULA_REVISIONS` accepts `3`; a table whose manifest
says `3` but whose registered era says otherwise is a conflict, not a default;
and `bake_verdict` refuses a bake whose declared revision disagrees with the
cached table's (`--cross-regime` remains explicit historical replay only).

---

## 6. Refit, and the two things that must not happen

The shipped **D** lineage and the fast-class campaign's best servable recipe
refit on revision-3 legs; board cells land under a distinct era suffix, never
mixed with revision-1 or revision-2 cells. Ship rule unchanged: install into
`ZensimProfile::D` only on the full gate AND CID22 ≥ today's D with CI.

**Do not relabel.** A revision-1 bake with its metadata edited to `3` is refused
by `check_pixel_revision`, deliberately — the refusal is the point, not an
obstacle. The two revisions that share the `Clamp` luminance form (2 and 3) are
distinguished by the REVISION comparison specifically so this cannot slip
through.

**Do not mix columns.** A training table with revision-1 rows and revision-3
rows is not a mixed-quality dataset, it is two different quantities in one
column. The manifest is what prevents it; check it, do not infer era from
column count or a familiar path.

### Replaying an existing candidate before any refit

Issue #61 asks for the existing candidate surface to be run on the corrected
path before new fitting. That is a cross-era measurement by construction, so it
needs both switches:

```sh
cargo build --release -p zensim --features cross-revision-diagnostic,...
ZENSIM_FORMULA_REV=3 ZENSIM_CROSS_REVISION_DIAGNOSTIC=1 <tool>
```

The cargo feature is off by default, so a product build does not CONTAIN the
bypass; the environment variable is required on top of it; and every affected
process prints a stderr line naming both revisions. A number produced this way
measures the extraction change against fixed coefficients. It is not model
quality and must never be reported as a score.

---

## 7. What is NOT done

- **No model is trained on corrected features.** Every stored table and every
  shipped bake is a revision-1 artifact.
- The `ZENSIM_FORMULA_REV` research pin does not make a BUILT-IN profile's own
  bake revision-checked — that contract lives in `BakeScorer`, where a bake
  declares its own revision and a mismatch is refused. A built-in profile
  declares nothing, so it IS revision 1: pinning revision 3 and reading its
  score prices revision-1 coefficients against another era's features. That
  cannot be refused here without breaking extraction (the same call emits the
  features a research run exists to collect), so it WARNS once per process on
  stderr instead. Use the pin to EXTRACT; use `BakeScorer` to score.
- HDR coverage at revision 3 is untested.
- Nothing here establishes codec RD gains, target-controller behaviour, or
  product qualification. The existing scorecard remains the release contract.
