# Shaping-study inference cost

This specifies the outstanding cost measurements required by
[the completion protocol](SHAPING_COMPLETION_PROTOCOL.md). It does not change
training, selection, convergence, or evaluation gates. The six prescribed image
extraction measurements are recorded in
[the extraction results](../benchmarks/shaping_extraction_cost_2026-11-09.md).
Finalist vector timing remains outstanding.

## Models and evidence

Under the owner-approved reduced scope, measure the actual 48 outer-fit models:
six finalist roles in each of three outer folds, retaining three paired seeds
where registered and one deterministic shaped-additive fit per fold. Resolve
roles, settings and exact cell coverage through `shaping_outer_evidence.primary_cells`
and the three existing outer admissions. The omitted full-FIT retuning and refit
stages are not prerequisites. Require completed fits, final reports,
native qualification, and reproducible model hashes before timing. Use the exact
representation that supplied each final report, including any qualified max-pair
re-export. Preserve original failed gates and convergence qualifications.

Report each model separately. Do not select a fast seed, average model parameters,
prune coefficients, quantize weights, or substitute a smaller tree. Timing does
not change the scientific shortlist. Record native model bytes, trainable
parameter counts, expanded stored coefficients or tree/node counts, and the
training-exposure/resource receipts. Storage size and coefficient counts are not
latency measurements.

## Native scoring

Use one fixed 40-row workload per outer fold: eight rows from each of the four
admitted human populations and CID22-SSIM2. All sixteen model arms in that fold
share those rows. Select from that fold's TRAIN complement only, by ascending original
row index within population, before reading predictions. Preserve exact f32
training-table values and the original feature-ID mapping. Pin the chosen indices,
source tables, declaration, exported rows and reference predictions. Freeze all
three row selections before opening predictions, as `prepare_outer` does.
Workloads differ across folds; report paired model costs within each fold.
Do not consume outer-test, development or untouched final-holdout rows for this
workload. This retains the existing TRAIN-only workload owner.

The neural ZNPR arm uses `BakeScorer::score_features`, including declared input
gathering, learned shaping, scaling and final score transformation. The tree arm
gathers the same raw feature IDs into the opaque vector layout, invokes
`CorruptionHead::decision_function`, and multiplies by 100, matching
`shaping_controls.py`. The tree control is trained on raw features, not damage
products. Include the tree gather in its timed region. Do not time its logistic
probability or corruption-gating score: neither is this study's regressor output.

The shaped-additive control uses its registered `zentrain.research_basis`
transform and native `Predictor`, exactly as the fitting owner does. Include raw
feature gathering and basis construction in its timed region. Share that basis
implementation with the fitting owner rather than duplicating it. It is an
opaque-vector model, not a pixel-serving ZNPR profile. Preserve any existing
dense-input conversion used by the evaluator and reproduce its output bytes;
do not introduce a new pruning or conversion policy for the benchmark.

Use the same ordered rows for every model. Validate outputs against each model's
qualified native predictions before timing, then repeat the comparison after
timing. Require exact score bits on the same runtime and dispatch; a mismatch
requires investigation, not a new tolerance. Preallocate work buffers and load
models outside the timed region; include the per-call work required by the native
scoring API. Report the 40-row batch latency and its per-row arithmetic mean,
clearly distinguished from individual-call latency. Record a separate fresh
process maximum-RSS measurement with the full model and workload loaded.

## Image extraction and qualification limits

Measure the native revision-7 extraction of the union of required raw feature IDs
separately at 256×256, 1024×1024 and 4096×4096, with one and four threads. Freeze
three deterministic nonidentical RGB8 reference/distorted pairs per dimension,
their generation procedure and byte hashes before timing. Verify strided and
tightly packed extraction parity outside the timed region. Record requested IDs,
layout width, decoder-free RGB8 input scope, and actual CPU dispatch.

Extraction timing excludes decode and model scoring. Vector timing excludes image
extraction. Report both directly; do not add medians and label the sum measured
end-to-end latency. The tree export is qualified for opaque-vector arithmetic
only. No pixel-serving claim follows from timing it. A future end-to-end result
requires an independently qualified complete adapter and an actual measurement.

## Timing and resource admission

Run only on an eligible non-training benchmark host, never dev or r7900x. Use
zenbench admission and randomized interleaved arms, with explicit CPU dispatch,
affinity, thread count, source/binary/model hashes and resource-wrapper limits.
Use the highest common supported dispatch for all models in a comparison; report
its exact name and host CPU. Do not use `target-cpu=native`.

For each comparison retain the first 32 clean paired rounds from at most 64,
with 100 ms warmup per arm. A round rejected by zenbench is excluded for every
arm together. Fewer than 32 clean rounds is an incomplete measurement; preserve
it and retry in a new directory. Keep raw samples, gate flags, iteration counts,
units and the zenbench reliability result. Report medians and paired uncertainty
from retained rounds, without treating repeated timing rounds as model seeds.

Maintain the 30 GiB SSD reserve, account for active and pending writes, and retain
full stdout/stderr plus the resource wrapper's peak-RSS line. Checksum every
artifact after reading it back from durable storage. Native instruments exist
as `zensim-validate/examples/shaping_cost.rs` and `shaping_extract_cost.rs`.
The saved vector qualification covers eight fixture models and forty rows;
it is arithmetic qualification, not finalist timing. Finalist availability,
qualification against the actual selected models and measured timing remain
completion gates.
