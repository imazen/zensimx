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
