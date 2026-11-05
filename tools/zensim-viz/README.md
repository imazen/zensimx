# zensim-viz

A native revision-7 feature atlas and MLP lens. Hover detector maps, inspect
individual weights and follow their served-score effects. [REPORT.md](REPORT.md)
records measured coverage, limitations and the completed design review.

## Inspect

The app uses one viewport: a virtual feature tree, the image or one lens exhibit,
and a paper reference pane. Search with `/`, walk with arrows, and pin with
Enter. Release the pin to resume hover previews. Wheel/drag zoom and pan the
image; arrows on the focused stage pan, and double-click resets. Gutters also
resize with arrows. Pins and the active legend stay in the bottom strip.

The photographic base becomes grey while a detector is shown; difference mode
keeps its native difference image. Nonzero map pixels use the selected opacity
times `0.25 + 0.75·t`; zero remains transparent. The bottom range control defaults
to native p2–p98 of the sampled, encoded map pixels, cached per shared map.
`own` uses the complete map range; `bank` uses the feature's whole-bank p1–p99.
Map-only stays opaque. These are display ranges; stored detector maps, pooled
values and flags remain unchanged.

The reference's compact model strip keeps total |weight|, ablation rank, reading
units and flag counts visible. Weights has a 128-cell capacity strip. Its native
rule is: dead = all-zero encoder column and bias ≤ 0; residual = non-dead and
|output coefficient| below 1e-3 times the largest; live = the rest. This differs
from inactivity measured on one bank. Click a capacity cell or use arrows to
select its stored unit; the coefficient and bias are in its tooltip.

MLP input names in concerns, ablations, incoming-weight strips and coefficient
tables link to the feature's formula, detector explanation and paper citations.
Reading-unit links open its paired or individual function. Copied links restore
the bake, bank and raw/fragility identity.

The **Function** exhibit shows the stored head as weighted max pairs plus an
explicit linear background and any remaining hinges. Select a pair or unit,
hover its native samples, or switch between its contribution, the complete
network and the complete served score. All coefficients and input scales remain
available in linked tables, including the summed linear-background coefficients.
The 65-point curves are mathematical raw-feature basis slices with all other
raw slots/fragility zero; they are not observed images or partial dependence.
Output calibration is shown separately with its native knots.

On the certified nonnegative standardized-input domain, all three F seeds have
one max pair, with at most two affine forms before calibration. E has three
pairs, giving at most eight forms. Always-active nonnegative residual units
contribute to the linear background: F0 has 31, F2 one. F1's 31 otherwise nondead
residual units and E's 29 have exactly zero output weights. These are structural
bounds, not a measured number of occupied regions or a proof about memorization.
No empirical memorization or generalization experiment is performed here.

Reference entries distinguish paper prose from revision-7 code-derived
formulae. KaTeX 0.19.1 loads once from cdnjs. Build the read-only paper join with
`just viz-reference "$output/audit/weights.json" "$output/reference.json"`.
The paper commit, bibliography and code hashes travel with the result.

Basic local SSIM, added-edge, lost-detail and MSE maps expose exact f32 terms
at the cursor. Their binary planes share the map sampling grid. Global HF
ratios and other families explicitly have no local term demo. The native
revision-7 direct-error form uses an error moment instead of reconstructing
covariance. RGB8 ingress and the other limits in REPORT still apply.

## Build and export

From the repository root, set absolute paths for `bake`, `comparison_bake`,
`bank`, `reference`, `distorted` and `output`, then:

```sh
just viz-build
export ZENSIM_FORMULA_REV=7
export VIZ_CERTIFIER=/absolute/path/to/bake_dial_refit
just viz-audit "$bake" "$bank" "$output/audit"
just viz-atlas "$bake" "$bank" "$reference" "$distorted" "$output/pair"
just viz-bank "$bake" "$bank" "$output/lens" "$comparison_bake"
just viz-compare "$bake" "$comparison_bake" "$bank" "$output/compare" 0
```

This is an unpublished standalone Cargo root. Recipes serialize native work
through the resource lock. Use a **new output directory** for each export;
preserve failed runs before retrying. The certifier must support
`certify-monotone --bake`. Atlas ingress supports opaque RGB8/gray PNG and RGB8
JPEG. Alpha and 16-bit PNG are refused; native HDR pixel maps are unsupported.

Banks need a producer sidecar `<table>.parquet.manifest.json` or parent
`_MANIFEST.json`, matching SHA-256, revision 7, canonical feature-set identity
and populated consumed slots. AIC tables are refused before table I/O.
Audit, percentiles and ablations use the whole bank by default; curves retain
200 observed rows at 21 points. Scoring geometry is recorded as 512×512.
`compare` defaults to 512 observed rows; the demo comparisons explicitly use
all CID22 rows. `--max-rows` selects another recorded diagnostic population.

For one curve use `just viz-lens "$bake" "$bank" "$output/curve" 100`.
The CLI accepts `--family` on `mlp-lens`, repeated `--percentile-bank` on
`atlas`, and `--training-bank` on `audit`. Raw damage interventions recompute
their fragility products. The lens draws at most 40 ICE traces; native bands,
means and the numeric table retain all 200.

`curve-audit LENS_DIRECTORY --out NEW_DIRECTORY --baseline OLD_LENS_DIRECTORY`
verifies every original native point and flag and reports exact maximum rises.
`audit --baseline-audit OLD_AUDIT_JSON` verifies original flags independently
of approved display-severity changes. No tolerance is introduced.

## Serve and add pairs

Keep full bundles outside git; link `web/data` to that output root.
`just viz-bundle PLAN_JSON` runs a [plan](examples/plan.json), freezes the binary,
records PIDs/full logs, and marks completion only after exit and artifact checks.
Identical completed jobs are reusable; changed arguments/binary need new paths.

`just viz-assemble CONFIG_JSON` uses an [assembly config](examples/assemble.json)
to publish completed, matching bundles into `data/manifest.json`. Native
`mlp-bank/index.json` supplies per-input and family curve paths. Existing
manifests are preserved. All web URLs are relative to `web/`.

To add a pair, select existing training reference/distortion files, add its atlas
job and assembly entry, export, then assemble. The exporter records paths/hashes,
compares every populated slot and full layout position against a control
extraction, and refuses any changed bit. Maps have at most a 512-pixel long edge;
global ratios are explicitly labeled without a spatial map.

```sh
just viz-assemble /absolute/path/to/assemble.json
just viz-serve
```

The static server binds all interfaces on port 3310; use the host's network DNS
name. `just viz-progress` observes this task's dated local runs and recorded PIDs;
it does not renew checkout locks or infer progress from an agent's status.

## Verify

```sh
VIZ_F_BAKE="$bake" VIZ_F_BANK="$bank" just viz-test-real
just viz-clippy
just viz-library-check
just viz-format-test
just viz-web-check
VIZ_TEST_COMPLETE=1 just viz-browser
just viz-review-browser
just viz-design2-browser
just viz-review2-browser
```

Real-bake tests require the frozen F0 bake/CID22 bank identified in
`f-audit-census.json` and `VIZ_CERTIFIER`. Browser checks need `VIZ_PLAYWRIGHT`
(installed module) and `VIZ_CHROMIUM` (executable). `VIZ_CAPTURE_DIR` saves eight fresh view/theme/viewport images and
`browser-review.json`; existing files are refused. `VIZ_TEST_SCOPE=audit` checks only the audit portion.

See the report for diagnostic definitions: zero MAD differs from an all-zero
row; inactive units are bank-scoped; drift is descriptive; ablations overlap;
weight indices do not align neurons across independent bakes. All strict
score rises remain recorded, with the approved magnitude-based severity.

`zensim-viz-export weights BAKE --out NEW_DIRECTORY` exports the same native
weight document with per-input total |weight| and an identity receipt. Set
`weights_directory` on a bake in the assembly config to use it; assembly verifies
its bake against each bank audit. This does not replace the bank audit.
`just viz-design2-verify OUTPUT_ROOT NEW_RECEIPT_JSON` compares complete Design 2
bundles against the retained review-1 exports, including every prior feature
field and map byte.

Review-2 captures use `viz-review2-browser`, which holds the shared heavy lock
and runs through run-heavy. Set `VIZ_REVIEW_COMMIT` to the verified source SHA
and `VIZ_CAPTURE_DIR` to a fresh directory: it saves
both sizes/themes/views, a delta view, two capacity-strip views, and
`browser-review.json`, `design2-gates.json` and an `index.html` gallery.
Older bundles need a fresh native atlas export for the
new map/bank ranges, and a fresh weights export for capacity metadata. Keep the
previous bundle and compare through
`just viz-review2-verify OUTPUT_ROOT NEW_RECEIPT_JSON review2-final-` for the
published final bundle prefix; all prior fields and raw PNG bytes must agree.

Function exports use `zensim-viz-export function BAKE --out NEW_DIRECTORY`,
with `VIZ_CERTIFIER` set. Add a function job to the existing bundle plan and set
`function_directory` on each bake in the assembly config; assembly checks its
bake identity against the audit. The current export supports certified,
untransformed, scale-only, single-hidden-ReLU/scalar-output bakes and explicitly
refuses unsupported heads or preprocessing. It reads model bytes without a bank.

`just viz-function-browser` runs the retained review-2 gates and the linked
feature/function checks under the shared heavy lock. The same capture/source
variables save the retained views plus all four function exhibits and affine
background examples; `function-index.html` links the complete gallery.
