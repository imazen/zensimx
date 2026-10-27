# Codec-target metric — the one user-facing quality dial

**Research status, October 27:** the active Mlic/AIC panel study has not changed
the serving profile. A completed TRAIN fit or in-sample gain is not adoption.
The [current queue](../SESSION-RESUME.md#current-work--october-27-2026) retains
runtime, steering, color/precision and production qualification before stable
adoption. The seamless HDR frontend follows that adoption, with the SDR fast
path preserved and PU21 retained as a comparison control.

**Current integration guide, checked against source on 2026-10-19.**
`ZensimProfile::codec_target()` and `latest_preview()` return **B**.
D is available explicitly; the September 5 change replaced D’s calibration,
not the `codec_target()` alias. The implementation in
[`profile.rs`](../zensim/src/profile.rs) determines routing and embedded bytes.

**User ruling, later 2026-09-07:** no consumers have calibrated to B, C or D;
all may be improved/replaced. The mapping below records today's implementation,
not a frozen score contract. New models must be fully executable and servable
in Rust through a zensim surface API, and evaluation must use that API,
including all heads, corruption gates, splines and final composition.

## What the user controls

A user chooses a target such as **80**. The encoder chooses its own quality
parameter, measures the decoded result, and corrects it within its work budget.
A picker may choose the codec as well. JPEG quality, WebP quality and JXL
distance are internal controls with different scales and directions; users
should not need to translate between them or select a research model.

The product objective is small output at the requested perceived quality, with
low total latency and memory use. An undershoot loses quality; an overshoot
spends extra bytes. A fast metric matters on every correction pass, and a good
starting estimate saves the entire encode/decode/score pass. A fast model
forward alone does not establish a fast encoder loop.

The intended score contract is:

- **100 represents identity**, and a distortion must not score above a perfect
  copy. Test the model path as well as the API’s identical-input shortcut.
- **Negative scores are valid.** Keep them in scoring, reports and chart axes.
  There is no prescribed negative depth such as −50. Each codec’s lowest
  configurable, distinct outputs must remain representable.
- The usable range must retain resolution at both ends, especially near
  lossless. A monotone calibration cannot repair an inversion in raw predictions.
- A given target should imply consistent perceived quality across images,
  content types and codecs. Human near-threshold evidence and independent
  judges test that claim; agreement with a teacher metric alone does not.
- A steering map must help the encoder improve the same scalar it targets.
  Its value is measured in real bytes, achieved quality and passes.

- Planned (owner decision 2026-10-11): zensim reports a declared JND estimate
  beside the dial now; once AIC2026 human labels arrive, a new profile puts the
  dial at `100 − 10 · JND` (90 = 1 JND) above a smooth join to the lower range.
  See [JND_DIAL_PLAN_2026-10-11.md](JND_DIAL_PLAN_2026-10-11.md).

These are requirements, not a claim that every available profile meets them.
In particular, **“60 = visually lossless/JND” is an old calibration convention,
not a validated universal promise for the current profiles**. The current
exam is [`MODEL_SELECTION_SCORECARD.md`](MODEL_SELECTION_SCORECARD.md), including
G-ADDR. Its floor rule is named in every comparison; the September ladder
ruler uses `resolvable` with margin 0.5 against SSIMULACRA2’s per-codec result.
See the [ladder measurement](../benchmarks/board_ladder_ruler_2026-09-06.md).

## Current variant → backing bake mapping

All filenames below are relative to `zensim/weights/`. Hashes are SHA-256
prefixes; the B rotation was checked on 2026-10-19. Each bake has a same-stem
TOML under [`weights/manifests/`](../zensim/weights/manifests/).

| Public variant | Current bake | Role |
|---|---|---|
| **B** (`zensim-b`) | `b_sdr_diff_frag_820_rev5_2026-10-19.bin` · `89cb4861…` | Default SDR scorer; `codec_target()` and `latest_preview()` |
| **BHdr** (`zensim-b-hdr`) | `bhdr_linear_shaped_cvvdpmix_byid_2026-09-06.bin` · `8461ac55…` | Default HDR route through the absolute-luminance entry points |
| **C** (`zensim-c`) | `c_sdr_purity944_byid_2026-09-07.bin` · `996dfbb1…` | Available SDR candidate; not the default |
| **CHdr** (`zensim-c-hdr`) | `c_hdr_l1t1944_byid_2026-09-07.bin` · `3ea640d3…` | Available HDR candidate; BHdr remains the default |
| **D** (`zensim-d`) | `d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin` · `cd1098b4…` | Fast SDR profile; 28 declared feature IDs, 1,420-byte bake |
| **A** (`zensim-a`) | `v47_strict_qat_native_byid_2026-09-06.bin` · `de0ddb3d…` | Deprecated profile |
| `PreviewV0_1`, `PreviewV0_2` | In-source weight arrays | Historical compatibility profiles |

B now means the frozen **diff+frag-820** model, with model-bound Rev5
extraction, no attached corruption head and no JND map. Its runtime is present
in minimal builds. HDR still routes to BHdr. This intentionally changes B's
SDR scores and retains known severe color-noise inversions and incomplete
qualification. See the [promotion record](PROFILE_B_DIFF_FRAG_820_2026-10-19.md).

C/CHdr/D use the default-on `candidate-profiles` feature; A uses
`deprecated-profiles`. The September 6 explicit-ID conversion of A/B/BHdr/D
preserved served scores relative to their prior bakes. See
[dense bake flip](../benchmarks/dense_bake_flip_2026-09-06.md) and the
[build-feature serving fix](../benchmarks/dense_serving_ungate_2026-09-06.md).

**C/CHdr now declare their consumed IDs and use canonical training activity
semantics.** The September 7 correction changes their pixel scores; it is not
a score-preserving conversion relative to the old mismatched serving path.
The full consumed-feature census, dense/wide gates and HDR evidence are in
[the planner correction record](../benchmarks/feature_plan_cleanup_2026-09-07.md).

Do not identify feature semantics by vector width. Bakes declare IDs; producer
and consumer must also agree on feature revision and decoder/extraction
provenance. [`FEATURE_SET_IDS.md`](FEATURE_SET_IDS.md) owns that contract.

## Integration and real callers

For ordinary SDR integration, construct the metric with
`Zensim::new(ZensimProfile::codec_target())` and score source and reconstruction
using the same pixel/color/alpha interpretation. Use the HDR absolute-luminance
API for HDR; sending HDR through an SDR pixel path does not establish a common
SDR/HDR scale.

For a dynamic candidate, use `BakeScorer` for both the scalar and the map.
With `custom-profiles` and `feature-regime-v2`, an SDR codec loop can reuse
its parsed model, reference cache and session:

```rust
let mut scorer = zensim::BakeScorer::new(&model)?;
let reference = scorer.precompute_reference(&source)?;
let mut session = zensim::Fused944Session::new();
// Repeat for each decoded reconstruction against this source.
let scored = scorer.compute_with_ref_and_attribution(
    &source, &reference, &reconstruction, Some("jxl"), &mut session, 8,
)?;
let achieved_score = scored.result().score();
let first_block_gain = scored.refinement_gain(0, 0, 8, 8);
```

The complete configured model supplies the sensitivities, including heads,
splines, ensemble and corruption gating. `refinement_gain` adds finite max
removal to the additive density. Inspect `unsupported_refinement_feature_ids()`
and `has_corruption_gate()` before using it. The query is non-additive: two
rectangles can each leave a tied maximum unchanged while their union removes
it. Coarse/reflected footprints are explicit; actual pixel edits also change
blurred neighborhoods and can cross nonlinearities.

`attribution().query_rect()` remains the additive density-only route; its
`unsupported_feature_ids()` report still includes max terms. Both routes
include L8 moment-removal linearization and its curvature limitations.
Masked/IW coverage remains unsupported. Complete local coverage does not prove
useful encoder steering; the [finite-query record](../benchmarks/max_attribution_2026-09-08.md)
retains the current pixel-screen failures and their non-max diagnosis.
The [September 8 binding record](../benchmarks/candidate_attribution_serving_2026-09-08.md)
includes exact score/feature gates and full coherence grids, with their tails.
Those checks do not establish codec improvements or model qualification.

Existing callers and their algorithm owners, checked in the sibling source:

| Consumer | Entry point / owner | What it controls |
|---|---|---|
| zenjpeg | `Quality::Zq` / `ZqExplicit`; `zenjpeg/zenjpeg/src/encode/zq.rs` | Starting quality, correction passes and optional block-artifact bound |
| zenwebp | `LossyConfig::with_target_zensim`; `zenwebp/src/encoder/zensim_target.rs` | Content-specific starting quality, secant correction, asymmetric tolerances |
| zenavif | `TargetMetric::Zensim`; `zenavif/src/target_quality.rs` | Encode/decode/score target search using `codec_target()` |
| jxl-encoder | `jxl-encoder/jxl-encoder/src/vardct/zensim_loop.rs` and `perceptual_loop.rs` | Scalar target and quantization-field refinement; pin the configured driver when comparing models |
| zensim-target | [`target_search`](../zensim-target/src/lib.rs) | Cross-codec search with a bounded iteration budget and best-probe result |

The codec-native JPEG/WebP paths use a quality floor and separate overshoot
and undershoot handling. Their defaults permit best-effort output; inspect
achieved quality or request the codec’s strict failure policy when needed.
The `zensim-target` helper instead reports symmetric target tolerance and a
`converged` result. Do not silently equate those success conditions in an eval.

**CLI default aligned during the September 7 cleanup:** `zensim-target`'s
library, CLI default and `--profile default` all use `codec_target()`.
Use `--profile tuner-v4` to reproduce the previous CLI default. See its
[checked usage](../zensim-target/README.md). This change does not establish
that every sibling consumer uses the same model.

The product loop is: predict an initial setting → encode/reconstruct → score →
accept or correct. Reuse source-side work where the actual scoring API supports
it. Keep the best valid candidate and record failures to reach the target;
an exhausted search budget is not evidence of success. Lossless codecs do not
have a lossy quality dial.

In-encoder per-block RDO is a separate cadence from an image-level correction
loop. Do not insert a full image comparison into thousands of RDO decisions;
see the dated [feasibility study](RDO_LOSS_FEASIBILITY_2026-05-24.md).

## Evaluate the delivered output

For every candidate, report target error (including undershoot separately),
target-hit rate, bytes, number of passes, total encode/decode/score time and
peak memory, per codec and content class. Include tails and failure examples;
a pooled median curve can hide an individual source’s inversion.

Compare bytes at equal **independently judged** quality. A model’s own score
cannot establish its RD improvement. Keep human holdouts distinct from
teacher-agreement corpora, and grade the high-fidelity range independently.
The [scorecard](MODEL_SELECTION_SCORECARD.md),
[wave playbook](WAVE_PLAYBOOK.md), and
[reproducibility contract](REPRODUCIBILITY.md) own the evaluation workflow.

**Research selection is not release qualification.** `freeze_check --select`
can select a recipe with incomplete G-ADDR coverage or failed codec floors.
The addressability owner’s `Verdict::shippable()` requires both tiers to pass;
other product gates must pass too. The September 6 board record demonstrates
this distinction on the selected fast-class recipe. D has the strongest
recorded five-codec floor result among the compared zensim families, while
stronger-rank MLP candidates retain floor and/or near-threshold tradeoffs.
No new common-dial winner is declared by this guide.

## Reproducibility and chronology

A public profile name is not a frozen model hash. Pin the bake, extractor
revision, decoder versions, calibration and target-predictor lineage for a
reproducible experiment. A dial calibration change may require rescoring data
and retraining/recalibrating experimental starting-quality predictors even
when rankings are unchanged. This is an experiment-reproducibility concern;
the user confirms there is no deployed consumer calibration to B/C/D to preserve.

- **May 2026:** Balanced/Compression/Tuner experiments and provisional JND/JOD
  anchors; those variant names and integration examples are historical.
- **July 12:** the codec-target alias became B.
- **August 29:** C rotated to the purity-trained bake; CHdr was added.
- **September 5:** D’s id100+negative-tail calibration replaced its prior dial;
  the proposed peaks model did not ship.
- **September 6:** A/B/BHdr/D switched to explicit-ID bakes without changing
  served scores; the board adopted the floor-dense ladder ruler.

Training lineage remains in [B’s methodology](../benchmarks/profile_b_methodology_2026-07-12.md),
[BHdr’s record](../benchmarks/bhdr_improvement_split_lineage_2026-07-12.md),
[C’s purity wave](../benchmarks/sdr_pure_retrain_wave_2026-08-28.md),
[CHdr’s wave](../benchmarks/hdr944_retrain_wave_2026-08-28.md), and
[D’s dial change](../benchmarks/d_ship_flip_2026-09-05.md).
The older version of this guide is retained in Git history; its May numbers
must not be treated as current product guarantees.
