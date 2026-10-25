# Evaluate metric implementations on explicit datasets

`panel evaluate` runs dataset-independent diagnostics using the same statistics
as `panel` and `bake_verdict`, all owned by zenmetrics-stats (the `zenstats`
dependency; zenmetrics `docs/EVALUATION_METHODS.md`). It accepts stored scores, keyed score tables,
external executables, batch executables, and the `zenmetrics score` CLI. It
writes JSON, Markdown, HTML, scored rows, and retained executable stdout/stderr.
It neither fits a metric nor discovers or opens default corpora.

```sh
cargo build --release -p zensim-validate --bin panel
target/release/panel evaluate --manifest examples/metric-evaluation/suite.json \
  --output /path/to/new-report-directory
```

The output directory must be new. Exit 0 means the requested computation
completed and any declared requirements passed; exit 1 means a scorer,
instrument or requirement failed, or a requirement was unmeasured; exit 2 means
invalid input or an execution/setup error. Without requirements the report is
diagnostic, not a pass certificate. Read coverage and individual criterion
states. Neither this report nor an attached instrument establishes production
qualification automatically.

The checked-in synthetic suite is self-contained. `just metric-evaluate-tests`
exercises real CLI invocations, including direction, keyed joins, Parquet,
external command failures, Zenmetrics multi-output selection and stale evidence.

## Dataset and metric contract

The manifest is `metric-evaluation-v1`, with nonempty `datasets` and `metrics`.
Paths resolve relative to the manifest; image paths in a table resolve relative
to that table. Executables use explicit paths, not PATH lookup. Each dataset
pins its exact file SHA256 and declares its role (`synthetic`, `train`, `eval`,
or `public_test`). Those declarations do not replace the dataset owner's
admission or authorize protected reads. Use the existing split/exposure rules.

CSV (including quoted fields), TSV and Parquet support named column mappings.
Only explicitly mapped Parquet columns are read. `id` is required and unique;
row order is preserved. Map arbitrary source headings to these names:

| Canonical column | Enables / meaning |
|---|---|
| `target` | Full rank panel and scatter; requires `target_kind` (`human`, `objective`, `synthetic`) |
| `reference_id` | Within-reference rank; separate from an image filename if needed |
| `content`, `codec`, `band` | Separate ranking panels |
| `sigma` | Positive per-stimulus standard deviation for normalized error |
| `quality`, `codec`, `reference_id` | Adjacent-rung dial diagnostics; dataset must declare `quality_direction` |
| `reference`, `distorted` | Original files for per-pair executable adapters |
| `pixel_sha256` | Optional known identical-pixel rungs; absence never means pixel equality |
| `right_id`, `choice`, `preference_group`, optional `weight` | Preference responses; current row is left, `right_id` names the other row. `choice` names the **more distorted** side (`left`, `right` or `notsure`), matching the forced-choice owner `zenstats::forced_choice` (zenmetrics-stats). `notsure` counts half for each side (owner decision 2026-10-11); tokens are parsed by `Choice::parse`, as `panel --pairwise` does. Rows must already be screened by the dataset's own rule (AIC-3: keep batches with at least 70% correct on their 10 trap questions): the evaluator sees no worker ids and cannot screen. The report adds the cross-fitted ceilings `ceiling_loo` and `observer_agreement` beside the in-sample `ceiling`. Empty right/choice cells have no response. |
| `severity_code`, `reference_id` | Existing five-level KADIS ramp owner: `type*10+level`, including its signed-type folding. This is a specific instrument encoding, not a generic severity scale. |
| `corruption_label` | Existing legacy corruption-ordering instrument: `ref__family__region__severity__corruption`, with corresponding `__q20` and `__q10` anchors. This is not an integrity detector qualification. |

Any additional mapped name can hold a metric's stored score. Missing/invalid
scores remain recorded and fail the assessment; no survivor-only correlations
are emitted. Invalid labels or ambiguous/duplicate keys are errors. A table can
have no target: human agreement then remains unmeasured rather than using a
quality setting or another metric as a human label.

Metric `direction` is required (`higher` or `lower`). `target_direction` defaults
to `higher`; set it to `lower` for DMOS/distortion targets. Signed quality-aligned
Spearman, Kendall, fitted Pearson and raw Pearson remain visible alongside the
legacy magnitude statistics. Direction is declared before scoring, never inferred
from observed correlation. `implementation` and `input_contract` describe the
build/model and pixel/color/display interpretation. `artifacts` optionally pins
model, script and configuration dependencies by path and SHA256; the executable
itself is hashed automatically. A wrapper must declare dependencies it uses.

The canonical merged quality-band method needs explicit `target_range: [lo,hi]`.
Its existing sample-size and target-span floors remain unchanged. The raw
scatter diagnostic uses the complete population; no plotting sample determines
tail statistics. `difference_threshold` optionally enables the DS-AUC
proxy (`zenstats::pairs::ds_auc`) in target units; it is not a measured human JND threshold. That owner
retains its deterministic pair subsampling at large populations.

`ladder_epsilon` belongs to each metric and is in that metric's native score
units. It must be explicit for dial diagnostics. The same rung classifier is
used by `bake_verdict`. These generic ladders report single-reference ordering,
range and dead zones. They do not infer encoder-attributed reversals or apply
Zensim's 0–100 calibration and mentor-floor thresholds to arbitrary metrics.

Full PWRC has quadratic cost. `max_panel_rows` defaults to 10000: larger rank
panels remain explicitly unmeasured. Increase it deliberately with an adequate
memory budget; this option never samples or truncates rows. Within-reference,
scatter, score retention and other applicable diagnostics still operate on all
rows. Run substantial evaluations through the normal resource-limited runner.
The evaluator parallelizes across every core (Rayon) unless `RAYON_NUM_THREADS`
caps it: paired comparisons repeat the full panel per bootstrap draw, and an
uncapped run of nine collections took 32 threads on dev (2026-10-25).

## Scoring adapters

Stored column, in the dataset's canonical column mapping:

```json
{"kind":"column", "column":"ssim2"}
```

Separate CSV/TSV/Parquet table, joined by exact row ID rather than position:

```json
{"kind":"table", "path":"scores.csv", "sha256":"<file SHA256>",
 "id_column":"image_name", "score_column":"score"}
```

Extra/duplicate IDs refuse; missing/invalid IDs produce failed score rows.

External executable: stdout is one JSON value; stderr is retained separately.
Arguments are passed literally without a shell. Available placeholders are
`{id}`, `{reference}`, `{distorted}`. Input files are hashed and kept unchanged;
the adapter owns decoding and its declared input contract.

```json
{"kind":"command", "program":"/path/to/metric-adapter",
 "args":["--reference","{reference}","--distorted","{distorted}"],
 "json_pointer":"/score", "timeout_seconds":120}
```

A numeric JSON root uses an empty pointer. A nonzero process exit, missing score,
nonfinite score, timeout or excessive output fails the row. Output limit is
1 MiB per stream; the timeout kills the direct process. Adapters must wait for
and clean up their own subprocesses. Per-process wall times include startup,
decoding and instrumentation; they are not qualified metric latency.

The Zenmetrics adapter uses its existing JSON score protocol:

```json
{"kind":"zenmetrics", "program":"/path/to/zenmetrics",
 "metric":"ssim2", "args":[], "timeout_seconds":120}
```

For multi-output metrics set `score_column` to the exact emitted key. Ambiguous
outputs refuse. Additional `args` preserve implementation-specific options such
as display selection. An available metric depends on the supplied binary's build;
use its `list-metrics` command. No metric names, versions or color conversions are
substituted. For native HDR use an adapter around the appropriate existing HDR
owner and declare its actual input contract; an SDR score invocation is not HDR
qualification.

For expensive initialization, use `batch_command` once per dataset:

```json
{"kind":"batch_command", "command":{
 "program":"/path/to/batch-adapter", "args":["{dataset}"],
 "timeout_seconds":600}}
```

It returns `{"scores":[{"id":"row-key","score":42.0}, ...]}`. Joins obey the
same strict keyed-table rules. The adapter reads the source table according to
its documented contract; it must not infer target labels or use them for fitting.
For large outputs, precompute a keyed score table instead of exceeding the JSON
output limit.

## Specialized instruments and requirements

Spatial intervention, RD, targeting, integrity, HDR, runtime and correctness
need information beyond image-pair scalar scores. Each appears as
`not_measured` unless a corresponding instrument is supplied. Reuse the owners
in [the playbook](WAVE_PLAYBOOK.md#one-owner-per-task); an arbitrary metric needs
an adapter implementing that instrument's capability.

An `instruments` entry selects `metric`, `dataset`, `criterion`, and either:

- `kind: "artifact"`, `path`, `sha256`, or
- `kind: "command"`, `command` (same executable protocol).

An instrument command receives placeholders `{dataset}`, `{scores}`,
`{metric_sha256}`, `{dataset_sha256}`, `{dataset_contract_sha256}`, `{criterion}`. Its JSON must be:

```json
{
  "schema":"metric-instrument-v1", "criterion":"targeting",
  "metric_sha256":"<run binding>", "dataset_sha256":"<run binding>",
  "dataset_contract_sha256":"<run binding>",
  "n":100, "measurements":{"p95_absolute_error":2.3}
}
```

The binding is recorded in `report.json`. The metric hash covers its complete
configuration, executable identity and declared dependencies. The dataset hash
covers the exact input table; the dataset-contract hash also binds mappings,
direction, ranges and role. Per-image input hashes accompany scored rows;
inputs are checked again after per-pair execution.
Stale bindings refuse. Artifact hashes protect bytes, not the scientific validity
of a claim: retain the measuring owner's underlying evidence and contract. A
claimed `pass` field is never imported as a decision.

Acceptance is explicit in `requirements`; no global composite invents weights
or hides missing capabilities:

```json
{"metric":"candidate", "dataset":"test", "criterion":"rank",
 "pointer":"/srocc_signed", "min":0.90}
```

Pointers are relative to `measurements`. Supply `min`, `max`, or both. Undefined,
missing or unmeasured values produce `incomplete`, never pass. Registered product
requirements must retain their existing values; generic thresholds do not replace
[the production contract](MODEL_SELECTION_SCORECARD.md).

Optional `comparisons` entries name `dataset`, `a`, `b`, `bootstrap_resamples`
and `seed`. Since 2026-10-11 a comparison resamples **references**, not rows:
`zenstats::resample::cluster_bootstrap` with `Design::OneWay(reference_id)` draws
whole references with replacement (a reference drawn twice counts as two), and
both metrics' panels are computed on the same draw. `delta` holds, in
`delta_order`, A − B for each panel statistic with its percentile interval
(Hyndman–Fan type 7), bootstrap standard error and two-sided `p_zero`;
`bootstrap` records the unit, reference count, B, seed, level and rule. OR and
Z-RMSE are lower-is-better. A dataset without `reference_id` reports the
comparison as not measured. With fewer than about 30 references the percentile
interval under-covers: read it as descriptive. For several metrics, apply Holm
(`zenstats::multiple`) to the paired `p_zero` values.

`legacy_row_bootstrap` keeps the § A.9 `decisive` rule (MRR tests plus a row
bootstrap of the panel deltas) under that label. Its bootstrap unit is a row, so
it is not confirmation; its bounds are type 7 percentiles since the
zenmetrics-stats pin. An inverted/undefined signed correlation refuses the
polarity-tolerant comparison rather than turning reversed quality into a
competitive model.

## AIC2026 preliminary release

[The dataset](https://doi.org/10.18419/DARUS-6156) provides decoded images,
bitstreams, cropped stimuli and objective metric tables. Its
[paper](https://arxiv.org/html/2607.22783v1) describes subjective scores as future
work. `JND_CVVDP` and the other metric-derived JND columns are **objective
estimates**, not human responses. Keep human-ranking capability unmeasured
unless an independently supplied subjective table is identified and admitted.

`scripts/inspect_metric_dataset.py ROOT --extract --report NEW.json` extracts
completed ZIP files into separate directories, validates CRCs, refuses differing
existing files, and inventories every Markdown/CSV while preserving hyperlinks.
Incomplete downloads are untouched. It discovers new completed ZIPs between
passes; rerun with a new report after a later download finishes. Inspection and
extraction confer no training permission.

Prepare a suite beside your own report directories:

```sh
python3 scripts/prepare_aic2026_evaluation.py /path/to/AIC2026 \
  --output /path/to/suite.json
target/debug/panel evaluate --manifest /path/to/suite.json \
  --output /path/to/new-report
```

This selects supplied CVVDP, SSIMULACRA2 and proposal-Butteraugli columns,
keeping cropped/full-resolution tables separate. It declares no target and
uses epsilon zero for strict ladder direction, not a perceptual materiality
threshold. Add `--objective-target CVVDP` when preparing a *separate* suite to
measure agreement with that objective teacher; the teacher itself is excluded
from the candidate list. These correlations are not human agreement.

`examples/metric-evaluation/imagemagick_psnr.py` is a small real external
adapter. Invoke it through an explicit Python executable, with arguments
`["/path/to/imagemagick_psnr.py", "--magick", "/path/to/magick",
"--reference", "{reference}", "--distorted", "{distorted}"]` and
`json_pointer: "/score"`. Pin the script and ImageMagick binary in `artifacts`.
It checks dimensions, accepts ImageMagick's differing-image exit status,
and refuses infinite PSNR. It uses ImageMagick's default pixel interpretation;
it does not reproduce AIC2026's supplied PSNR-Y implementation.

The [AIC2026 verification record](../benchmarks/metric_evaluate_aic2026_2026-10-09.md)
records the completed archive inventory, input hashes, measured examples and
remaining capability gaps.

## E34 paired seed comparisons

`panel e34 seed-parity --input <E33_SUMMARY.json> --output <NEW.json>`
recomputes the historical ten-seed means, sample SEs and C-vs-A Student tail
from the hash-pinned summary without prediction reads. The independent Python
check is `just e34-stats-reference`, followed by `just e34-stats-parity` and
`just e34-stats-check` (fresh receipt destinations).

`panel e34 evaluate --input <PAIRED.json> --output <NEW.json>` owns the E34
nine-test family. Schema `e34-paired-seed-v1` supplies four ordered datasets
(`kadid`, `tid2013`, `konfig`, `cid22_a25`), aligned target/reference/codec
vectors, fixed orientation (+1 or -1), and ten prediction vectors per model.
Reference IDs are namespaced by dataset; a shared seed factor resamples the
arm and control at the same index. The statistic recomputes signed SROCC per
seed copy and dataset using expanded rows, then averages seeds and datasets.
The frozen E33 KonFiG keys distinguish parta/partb; both views of one source
must receive the same reference ID (seven pictures, not fourteen views).

The owner uses zenmetrics-stats `Design::TwoWay`, B=10,000, base seed
20261011, registered offsets and type-7 intervals. It reports reference-only
and historical seed-unit components, the directional bootstrap p, corrected
Student p at df 9 and their maximum, dataset/W2 guards, and one Holm family
of nine including p=1 for incomplete entries. Test order and model names are
fixed in the owner. Eligibility-only pair classes use `PairTable` under
`OneWay(reference)`, fixed orientation, target tie zero, no family labels;
seed means, seed-unit SE and excluded target ties are reported.

`panel e34 noise` accepts the two pre-freeze A/C-vs-control comparisons and
reports Q9 arithmetic without a Holm or adoption verdict. These commands do
not grant data access: callers must admit and ledger their explicit input
before opening prediction/target payloads. They do not replace the separate
registered label-free/serving gate owners or qualify a model.

E34 reports the inherited E33 mean-delta and seed-unit W2 point guards beside
the registered two-way guards. Primary, dataset, W2 and reference-only results
record their explicit bootstrap seed. Reference-only Student intervals use the
reference level count; the pooled two-way decision retains G=10 and df=9.
