# Optimized trainer: completed monotone-SiLU control

Five other neural control classes remain in progress. No corrected-condition
scientific fit is established by this result.

Control `e3c79ecb9ba271091aa1` (inner partition 1-1, setting 0, seed 0)
completed 240 epochs of 50,000 pair draws on r3800x using runtime source
`e643c8834d6378e4e19c803bff482d0b4d7fd38a` and BUILD
`56a05880e05dcfa78d28207a88df6f605888814e47707e216e5941d743ac0cff`.
Its original frozen BUILD was
`d013c2741eb282eb792231ec5da954fd58c205cc0ee5e653e11c56f07cbb61f2`.

The existing native qualifier's final canonical hashes both equal
`1448005ce698e6db3144633ae1f61c1c3f9cffcd99224ec72376a70bd111b852`.
The separate trajectory owner at source `063095607e92` then compared **240 of
240 finalized checkpoints**. Every canonical bake matched exactly, after
removing only `zentrain.repro` and `zentrain.monotone_research` metadata.
Weights, scalers, operator parameters and the remaining bake contents were
retained in the comparison. This establishes this specific control's parity;
it does not qualify every activation, source condition or CPU tier.

The canonical comparison completed in the wrapper's recorded six seconds,
with peak RSS 0.03 GiB. This is qualification elapsed time, not an inference
benchmark or a claimed trainer speedup. The training wrapper reported:

```text
run-heavy: done rc=0 8333s | peak-RSS 1.28GiB | min-avail 10067MiB | peak-load 13.93
```

The source engineering service had a verified 3 GiB MemoryMax and zero swap
allowance. Its native trainer finished successfully; no fit was restarted.
Two retained full-state snapshots contain 14,232,084 and 14,991,973 bytes.
Snapshot CPU/runtime identity remains binding; arbitrary host transfer is not
qualified by this result.

Evidence on dev:
`/mnt/v/output/zensim/shaping-completion-20261107/optimized-runtime-neutral-qualification-20261108/verified-evidence-v1`.
The `trajectory-check-20261108/RESULT.json` ledger preserves every input and
canonical checkpoint hash. Its README records the shared input and native-tool
dependencies. Intermediate canonical model copies were not uploaded; their
source bakes and comparison ledger remain available.

The 255 evidence artifacts, 87,324,482 bytes, were copied to the corresponding
`optimized-runtime-neutral-qualification-20261108-evidence-v1` directory under
the study's durable mirror and reread with complete SHA-256 verification.
Mirror receipt SHA-256:
`b1ad40a490b50da4218386c5223989812cda79f58e03687c22cd46b68ce8f395`.

Native command and input identity are in `native-fit/RUN.json`; the comparison
was invoked through `just shaping-completion-neutral-trajectory` using that
completed control, the new qualification directory and the pinned runtime.
