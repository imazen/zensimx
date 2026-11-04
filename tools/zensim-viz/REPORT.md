# zensim-viz report

## Coverage and current limits

Neither visualizer is qualified yet (0/2 complete). Missing: compiled F/E
weight-audit counts and regression pins; real atlas bundles and bit-parity
results; numerical lens tests; browser checks with real data; screenshots.
The first build is queued behind the shared training lock. No audit census
or numerical/pixel parity result is claimed before that build executes.

## Verified findings

The canonical native monotonicity certificate certifies F seed 0 with one
live max pair, E with three, and both with zero free ramps / free amplitude.
A negative encoder entry inside a max-pair witness is permitted. The audit
must test the coupled constraint rather than assert every encoder entry is
nonnegative. Source: `mlp_train::certify_monotone_bake`; command:
`bake_dial_refit certify-monotone --bake BAKE`.

The supplied design directory matches `web/design/` byte for byte. The plan
and assets landed on main in `5789a78f081f`. The app serves successfully over
the network (HTTP 200), and real Chromium checks pass for view switching,
both themes and empty states without uncaught page errors. Those checks do
not validate data-backed interactions.

## Implementation awaiting compilation

Standalone Rust commands: `audit`, `mlp-lens`, `compare`, `atlas`.
The audit delegates overall certification to the existing native certifier,
locates paired/gated per-weight witnesses, and emits flags plus full weights.
`VIZ_CERTIFIER` names that binary. Bank diagnostics explicitly use a recorded,
deterministic observed sample; they are not whole-bank certification.

The private, disabled-by-default `viz-export` feature retains fold planes
and serializes shared detector terms. Default extraction code is unchanged.
Atlas export launches instrumented and control processes and refuses any
pooled-bit difference. PNG/JPEG ingress and PNG output use Imazen codecs.
Global moment/scale ratios carry explicit no-map labels. Native grids are
nearest-subsampled only when their long edge exceeds 512 pixels.

PDP/ICE and ablations call BakeScorer directly. Compare exports align input
identities and report norm drift; stored hidden-unit indices are not neuron
alignment. Weight drift alone does not establish functional change.

## Audit census

| Bake | Native certificate | Live max pairs | Individual flag counts |
| --- | --- | ---: | --- |
| F seed 0 | certified | 1 | not yet measured |
| E | certified | 3 | not yet measured |

## Design questions for the coordinator

- The 14-pixel matrix cells with 2-pixel gaps cannot each have a disjoint
  24-pixel hit target. Preserve the specified dense grid with keyboard/table
  equivalents, or increase cell pitch? Current draft preserves the grid.
- Mean/MAD flags are diagnostic notes; a high weight is not intrinsically a
  defect. The concern table labels this distinction and reserves critical
  severity for failed numerical/sign contracts.
- Raw human scores may use different units from served scores. If present,
  the range plot must state their native units; a separate aligned panel is
  preferable to an unexplained overlay. CID22 oracle labels are not shown as
  human responses.

## Local preview

http://dev.lan:3310/

This is currently the interface frame; measured bundles are pending.
