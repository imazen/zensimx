# zensim-viz

A native revision-7 feature atlas and MLP lens. Hover detector maps, inspect
individual weights and follow their served-score effects. [REPORT.md](REPORT.md)
records measured coverage, limitations and the completed design review.

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
```

Real-bake tests require the frozen F0 bake/CID22 bank identified in
`f-audit-census.json` and `VIZ_CERTIFIER`. Browser checks need `VIZ_PLAYWRIGHT`
(installed module) and `VIZ_CHROMIUM` (executable). `VIZ_CAPTURE_DIR` saves fresh
view/theme images. `VIZ_TEST_SCOPE=audit` checks only the audit portion.

See the report for diagnostic definitions: zero MAD differs from an all-zero
row; inactive units are bank-scoped; drift is descriptive; ablations overlap;
weight indices do not align neurons across independent bakes. All strict
score rises remain recorded, with the approved magnitude-based severity.
