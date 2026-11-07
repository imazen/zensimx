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
