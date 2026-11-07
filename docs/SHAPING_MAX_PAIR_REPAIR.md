# Native max-pair rounding investigation and re-export

The frozen study's unchanged order gate found epoch-119 violations in completed
ReLU runs. The original runs, bakes, predictions and failed receipts remain
preserved. Scalar fused float32 arithmetic reproduces the native violation;
independent f64 algebra preserves ordering. This is an export/evaluation issue,
not evidence that the learned real-arithmetic function is nonmonotone.

The native runtime proposal is [zenanalyze PR 94](https://github.com/imazen/zenanalyze/pull/94),
dependent on [PR 93](https://github.com/imazen/zenanalyze/pull/93). It directly
evaluates declared tied pairs as maxima, with native structural admission.
Wire code 6 ensures older runtimes reject the new representation. No existing
model's execution changes unless it is explicitly re-exported.

`bake_dial_refit repair-research-max-pairs` accepts a frozen original plus a pair
declaration in stored-unit order. It preserves all weights, biases, scaler and
unit order, verifies those bits after serialization, and refuses to overwrite
an output. It rejects optional decision sections and non-f32 layers. Native
parsing independently validates pair indices, detector signs, bias constraints,
tied downstream coefficients and input domain. Python's diagnostic weight
inspection can propose only uniquely matched signed pairs; ambiguity is a
failure requiring investigation, not permission to guess or drop terms.

Qualification requires native replay of the same probes, original-vs-repaired
prediction comparisons on TRAIN and inner validation, identity, finite/extreme
inputs and baseline parity for unmarked models. The original failed gates do
not become passes. Any qualified repaired export gets a separate model hash,
binary identity and report; no checkpoint or outer score selects the repair.
Apply the qualified representation rule consistently to eligible candidates,
including controls, and measure its inference cost. The frozen fitting matrix,
training exposure, selection boundaries and numerical thresholds stay unchanged.

At initial implementation, runtime/baker tests, no-std checks and strict clippy
passed. Research library and trainer tests also passed with the new dependency.
The first failed checkpoint (`6df732ab294270460360`, epoch 119) was re-exported
without changing weight, bias or scaler bits. Its repaired SHA256 is
`59822e442fcf7c3340c103a659da6af6ebc48219a9abd065550925a12ccd4bcf`.
The same 128 paired probes now have zero violations at the unchanged 1e-5 gate,
identity is exactly 100, and the extreme-ray scores are finite. The old reader
rejects the new wire code.

Across all five TRAIN and five inner-validation population tables, the new runtime
with the original unmarked bake reproduces every original prediction exactly.
The repaired bake's largest absolute score difference is 0.000133514404296875.
These are rounding comparisons, not evidence of improved predictive quality.
The comparison wrapper reported 4 seconds, peak RSS 0.18 GiB, minimum available
39995 MiB and peak load 17.88; these are resource observations, not isolated
inference benchmarks. Targeted CLI strict clippy also passed.

Evidence is under `max-pair-qualification/` in the shaping-completion artifact
root. All three frozen native controls (`new-single`, `deep-relu`, `deep-smooth`)
replayed with exactly matching canonical hashes. That wrapper reported 12 seconds,
peak RSS 0.16 GiB, minimum available 40949 MiB and peak load 18.56.
The qualification archive was reread and verified against manifest
`10916b0792372a1c8354d952e4319a7bef55622fccc4170a85e797329884a62a`.
Full repaired evaluation across all eligible cells and inference-cost measurements
remain required.

Supplemental native evaluation now has a separate owner:
`just shaping-completion-reexport STUDY RUN BINARIES INSPECTOR EPOCH`.
It requires a completed registered inner ReLU fit, verifies the original training
identity and evaluator binary hashes, preserves the original inspection and native
repair receipt, and evaluates through the existing panel/probe owner. Reports go
under `study/reexport_max_pairs/`; they do not overwrite original reports or enter
base selection. The same original TRAIN label maps and memberships are used.
The first epoch-119 replay completed all population panels and unchanged gates;
its human inner composite was 0.9126419709659228. This diagnostic checkpoint is
not a selected candidate. Resource wrapper: 3 seconds, peak RSS 0.23 GiB,
minimum available 39878 MiB, peak load 20.96.

The first uniform replay wave covers every completed ReLU base fit on inner-0-0
(12 settings/seeds) and both completed fresh-head controls: 104 final/checkpoint
reports, all passing the unchanged native gates. All 14 have separately stored
convergence receipts. Each full-exposure base fit triggers the registered loss
investigation; five also encounter one surviving output coefficient. Two have
same-cycle human improvement above 0.002. Neither completed fresh-head control
triggers these criteria. These are within-fold diagnostics, not a family ranking.

Repaired convergence binds every original checkpoint, repair receipt, model and
evaluator hash. Inner selection can explicitly use this representation for ReLU,
but requires the same representation and evaluator across all paired settings,
seeds and folds, and agreement between final and diagnostic reports. The actual
outer-0 selection attempt correctly refuses because the other inner fold is
incomplete; no selection receipt was emitted. Original failed reports remain
untouched. The rescue protocol and unchanged numerical gates still apply.

## Softplus first-layer repair qualification

The same first-layer max-pair rounding reversal was reproduced for Softplus
cell `6e988eec6f23b804ff31`. The repair changes only the marked first-layer
representation; the later anchored Softplus construction and all parameter bits
remain preserved by the existing native admission/export owner. The final and
all seven diagnostic checkpoints pass unchanged probe gates. Full supplemental
evaluation is now collected uniformly for every completed Softplus inner fit.

For this cell's final checkpoint, all five TRAIN and five inner-validation
populations reproduce stored original predictions with the original runtime.
The new runtime with the unmarked model reproduces those float32 predictions
bit for bit. The repaired model's maximum absolute difference is
0.0002899169921875 across these tables. This is a rounding comparison, not an
improvement claim. The separate `softplus-runtime-comparison` archive passed
complete destination checksum verification. Selection support and consistent
later-stage representation admission are implemented below; complete paired
scientific coverage remains unfinished.

## Uniform Softplus selection and later-stage admission

Following the numerical compatibility qualification above, use the direct-max
first-layer representation for the entire Softplus family, as for ReLU. This
is not an option searched per setting, seed or checkpoint. Base selection must
have every paired final panel and diagnostic receipt in this representation
with one pinned evaluator. Rescue comparisons, triggered extensions, outer
refits, learning curves and full-data fits inherit that representation through
the existing admission chain. No original failed gate is converted to a pass.

The ordinary SiLU diagnostic and monotone SiLU family are not admitted by this
qualification. All source boundaries, fitting budgets, human-only selection
composites, finite-score requirements, convergence triggers and native probe
thresholds remain unchanged. The native repair still validates each model's
first-layer structural requirements; family membership alone cannot qualify it.

The supplemental archive supervisor covers base, fresh-head, rescue and
registered exposure-extension checkpoint panels and repaired bakes. Extensions
must reproduce their registration through the existing selection resolver before
any copy. Every final and scheduled diagnostic report, source checkpoint and
conversion artifact remains hash-checked before cold archive readback. This does
not yet provide automatic archive coverage for outer or full-data stages.
