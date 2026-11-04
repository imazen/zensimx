# zensim-viz report

## Coverage and current limits

Neither visualizer is qualified yet (0/2 complete). Missing: regression replay; real atlas bundles and bit-parity results;
numerical lens tests; browser checks with real data; screenshots. F and E
audits have completed on the whole admitted CID22 TRAIN teacher fit bank.
The pinned regression and map/PDP oracle tests are now running.

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
| F seed 0 | certified | 1 | 0 sign; 32,344 row outliers; 2,275 column outliers; 64 near-zero outgoing rows; 0 constant inputs; 95 inactive/flat hidden units; 54 correlated input pairs |
| E | certified | 3 | 0 sign; 20,564 row outliers; 3,375 column outliers; 122 near-zero outgoing rows; 0 constant inputs; 93 inactive/flat hidden units; 46 correlated input pairs |

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

## First F audit (completed)

The packed F seed-0 bake has 126,080 weights (984 × 128 plus 128 × 1).
The audit read every one of the 12,163 rows of the admitted CID22 TRAIN
oracle-teacher fit bank. It found zero sign/scaler violations and zero
constant inputs. One paired witness is hidden units 126/127.

There are 95/128 hidden units in their inactive ReLU region on this bank;
this does not establish deadness on the full training mixture. The 64
near-zero outgoing rows are in layer 1 (hidden→output), not dead raw inputs.
There are 54 input-pair flags at |Pearson| > 0.999. Row-MAD flags: 32,344;
column-MAD flags: 2,275. Zero-MAD deviations are explicitly undefined z
scores and diagnostic notes. No drift bank was supplied for this first run.

The complete local audit is under
`/mnt/v/output/zensim/viz/2026-11-04/audit-f-s0/`; the compact census and
input hashes are in `f-audit-census.json`. Regression replay is pending.
The first successful build and audit logs are preserved. Audit resource
receipt: `run-heavy: done rc=0 9s | peak-RSS 0.29GiB | min-avail 38062MiB | peak-load 4.57`.

## E comparison census (completed)

E has 105,088 weights and was assessed on the same 12,163 CID22 TRAIN teacher
rows and table hash as F. It has zero sign/scaler violations; 93/128 hidden
units are inactive on this bank; 122 outgoing rows are within 1e-6; zero
inputs are constant; 46 input-pair correlations exceed 0.999 in magnitude.
Its 20,564 row and 3,375 column outlier flags are diagnostic notes. Counts
and hashes are in `e-audit-census.json`; full JSON/TSV are in the local
`audit-e/` bundle. No drift comparison has run yet. Resource receipt:
`run-heavy: done rc=0 7s | peak-RSS 0.25GiB | min-avail 38097MiB | peak-load 2.25`.
