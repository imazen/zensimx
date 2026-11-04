# zensim-viz implementation plan

The brief and visual design are `/home/lilith/tmp/zensim-viz-brief/BRIEF.md`
and `DESIGN.md`. The supplied `design/` directory is copied verbatim under
`web/design/`. This standalone tool is an inspection instrument; exported
numbers must come from the existing Rust owners.

## Existing owners and map contract

`research::Request::for_slots` and `research::extract` run the deterministic
fold and return pooled values with registry provenance. `feature_v2.rs`
merges strip partials in order and finalizes with
`finish_channel_scale_at_revision`; `feature_v2_stream.rs` supplies the
strided strip planes. The active process must explicitly select revision 7.

`attribution.rs` retains basic, peaks and supported v2/append signals.
`BakeScorer::prepare_steering` and `ScoredAttribution::refinement_gain` expose
score-oriented density and region sensitivity, not raw pre-pooling maps.
Their density normalization, p-pool derivative factors and coarse spreading
must never be presented as pre-pooling feature maps. Unsupported maps must
be named explicitly, without substituting a generic difference image.

Add an export-only fold observer under a disabled-by-default feature if the
retained signals cannot be accessed through an existing supported surface.
The observer copies kernel terms after their calculation, without modifying
accumulation order, padding, SIMD dispatch, or finalization. Global ratios
need their actual numerator/denominator fields and an honest global label.
The atlas export will compare every pooled f64 bit with a separate
`research::extract` call on the same pair, including odd sizes and strided
rows. Maps remain on native scale grids, at most 512 pixels on the long edge;
encoding and image ingress use Imazen codecs only. Each bundle records input
hashes, revision, source and map range; binary bundles stay outside git.

## Ordered milestones

1. Implement the standalone Rust exporter, first the per-weight audit on F
   seed 0. Decode ZNPR with zenpredict, inspect declared model inputs and
   scaler signs before assigning monotonicity constraints. Audit every layer
   weight and bias/unit; retain row/column median-MAD statistics, including
   an explicit zero-MAD convention. Bank diagnostics distinguish exact
   deadness, corpus constants, duplicate inputs and distribution drift.
   Pin F bytes and measured flag counts in the regression test.
2. Export one real pair with bit-exact pooled values and actual pre-pooling
   maps. Build the atlas with supplied tokens, preview/pins, family matrix,
   keyboard navigation, scale/channel/fragility/pooling/bank filters and
   common-range pinned comparisons. Expand to 6–10 content-varied pairs.
3. Export PDP/ICE (at least 200 observed rows), range separation, identity
   family/input ablations, hidden-unit summaries, bake comparisons and
   monotonicity flags. All displayed scores call BakeScorer directly.
   Test curve points against independent direct score_features calls.
4. Complete the MLP lens: concern table, architecture, flagged weight
   matrix, PDP/ICE, quantile whiskers and family attribution. Provide
   keyboard/table equivalents and tooltips. Verify browser behavior and
   screenshots in the prescribed output directory.
5. Write README and REPORT with measured E/F audit counts, actual coverage,
   unresolved design questions and the verified network URL. Land verified
   milestones through scripts/safe_push.sh and verify remote ancestry.

## Execution and data protections

All edits are confined to this jj workspace. Commands repeated during
implementation get `just viz-*` recipes. Builds/tests queue behind
`~/tmp/heavy.lock` and run-heavy (16G); never bypass active training or a
queued higher-priority run. Initial inspection found two training flock
processes using that lock. Full logs live in `~/tmp/zensim-viz/`.

Read only admitted TRAIN and explicitly supplied HDR banks. Do not read
AIC-2026 tables or export their statistics. Bank loading must project only
required columns, reject missing/nonfinite inputs, and preserve row identity.
No training data, generated images, large JSON or bake bytes enter git.
No metric arithmetic, public API, established expectation or frozen artifact
is changed as a side effect of this tool.

## Verified model correction (native certificate, 2026-11-04)

The brief's unqualified nonnegative-encoder description is incomplete. The
canonical `bake_dial_refit certify-monotone` certifies F with one live max pair
and E with three; both have zero free ramps and zero free amplitude. In a max
pair, `c*(ReLU(a*x)+ReLU((b-a)*x-t)) = c*max(a*x,b*x-t)` on nonnegative inputs.
Negative entries in the second column can therefore be legal. The audit must
show paired/gated witnesses, use the canonical certificate, and flag failures
of the actual constraints rather than flag every negative encoder entry.
The native certificate's source owner is
`zensim-validate/src/mlp_train/mod.rs::certify_monotone_bake`.

## Design review 1 — verified implementation

Apply the coordinator's ten decisions from the supplied review after the verified
atlas count replay (landed `fbf2feb7`). Preserve all measured native flags and
zero-tolerance checks. New exports use fresh directories; old data is retained.

1. Magnitude-based rise severity; structural concerns critical. Group flags with
   count/worst witness; severity-first order; 50-row paging.
2. Add exact all-zero incoming-row `dead-input` findings and truthful readouts.
   Verified correction: F0 input 371 has 33 nonzero weights, not an all-zero row.
   Its zero MAD must not be mistaken for deadness.
3. Export native interquartile/centered-ICE statistics; render at most 40 ICE
   traces while retaining every one of the 200 original curves and all flags.
4. Native row/column orders, 1:1 minimap, selected-region zoom, named signed
   top-32-input/top-16-unit strips. Preserve stored input/unit identities.
5. Keep native human units in a separate aligned panel (Q2 confirmed); human
   marks use the requested series-2 point/whisker encoding and half-bin offset.
6. Derive signal groups from canonical registry names, use them for atlas rows
   and ablation ranking, and label top inputs with short names beside IDs.
   Retain the original broad-family curves as well as new signal-group curves.
7. Value-dependent overlay alpha; opaque raw map-only mode; pooling grid in legend.
8. Default percentile colors; divergent value colors only for bank ranges that
   cross zero. Export bank min/max alongside the existing percentiles.
9. Keep 14px cells/2px gaps, expand hover targets invisibly, move global-map
   affordances into the global column without dropping their individual values.
10. Keep a visible empty pin strip with the Enter affordance.

Regression gates: every original raw/broad-family ICE point and flag remains
bit-identical on the same bank; full F census retains its existing counts with
an additive exact-zero finding; atlas parity still covers all 504 populated
slots and all 720 layout positions. Exercise grouping, paging, centered view,
minimap/zoom identity mapping, overlay/map-only alpha, signed gating, global
cells, empty pins, both themes and bank scope in a real browser. Capture fresh
four view/theme images and record the landed commit in REPORT and chat.

Review completion: all ten decisions are implemented and checked against complete
native bundles. Six audit and six curve preservation checks retain the original
measurements; current dark/light screenshots and detailed coverage are in REPORT.md.

## Design 2 — viewport, references, then cursor terms

The coordinator's DESIGN-2.md supersedes page layout. First replace the frame
with a 44px header, virtual feature tree, dominant stage, reference pane and
56px filmstrip. Retain every native concern, curve, display identity and map.
Only one lens exhibit is visible. Gutters and all controls support keyboards.

Extract bibliography entries and family descriptions from a pinned, read-only
zenpapers revision. Its current draft uses Typst and four split bibliography
files, without fenced LaTeX blocks. Label code-derived revision-7 equations,
and explicitly mark signals lacking paper descriptions. Join all 984 model
inputs to reference entries; report coverage and source hashes.

After the frame works, export basic-family term planes from retained native
fold values, on the same sampled grid as maps. Preserve full f32 precision for
the cursor demo; never infer terms from quantized PNGs. Compare all pooled
bits against extraction without the hook, including odd sizes and row padding.
Preserve existing exports and flags in fresh artifact directories.

Acceptance: both views and themes at 1600×1000 and 1280×800 (eight captures),
document height equal to viewport, stage at least 55% width and 70% height at
1600×1000, preloaded hover below 50ms, complete stored-ID/reference joins,
and native term/parity checks. Record evidence and open questions in REPORT
and chat; push each verified milestone to main.

Design 2 completion: the fixed frame, all 36 code-derived equations, paper
coverage labels, 984-row virtual join and native basic cursor terms are
implemented. Eight viewport/theme/view captures and exact cursor, map, pooled
bit, original-weight-field and review-1 regressions pass. Current coverage and
source commits are recorded at the top of REPORT.md and in design2-evidence.json.
