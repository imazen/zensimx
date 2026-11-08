# Rescue execution on replacement workers

`shaping_rescue_worker.py` transports a fully admitted registered rescue. It
captures the files read by the existing selection, trigger and reduced-scope
owners, then adds the original native template, TRAIN tables, family metadata
and qualified executable closure. Every imported project module is hash-bound.
Workers reproduce the existing rescue resolution before claiming a compute slot;
the original execution owner rechecks it again before fitting. A request is a
transport manifest, not an alternate scientific admission rule.

Research paths remain identical across hosts. Exported source documents and the
single frozen derived-input declaration may relocate only with identical hashes.
The native command relocation refuses an existing declaration whose bytes drift;
a missing old source path can use the shipped frozen declaration. No worker repo
is overwritten. Existing research files must already match; rsync uses
`--ignore-existing`, then all destination hashes are checked before fitting.
Transfer preserves file permissions, including executable bits, without changing
the ownership or timestamps of shared ancestor directories.

Each dependency manifest has at most 4,000 files and 384 MiB. Incoming rsync runs
inside the shared SSD reservation owner and a 512-MiB memory scope. Training uses
the existing aggregate CPU admission, a two-GiB memory scope and original phase
budgets. Inputs are shared rather than copied per seed. Keep the 30-GiB SSD
reserve and account for pending output reservations. Existing attempts, including
failures, refuse another launch.

An explicit, hash-bound `TRAINING_CPU_BUDGET.json` can set aggregate training
counts independently of per-job thread pools. Its owner approval must preserve
RAM/storage guards and per-job thread caps; observed CPU availability must match
the approved host record. Missing configuration retains historical admission.
Existing frozen executables and loaded orchestration exports remain unchanged;
new owners use the approved budget, with queued-request transfers recorded
explicitly before any replacement controller starts.

`shaping_queue_transfer.py` can move unopened requests from an initial wave.
It excludes the first worker slots, including threads waiting for admission,
and refuses any completed wave member or already-started transfer candidate.
During the short CPU/storage admission transaction it suspends only the Python
queue, preserves each original request under a transfer suffix, and resumes the
queue in `finally`. Separate native controllers/trainers continue throughout.
The old queue later records explicit missing-request errors for transferred
proposals; these are superseded orchestration entries, not failed native fits.
Each new owner gets a fresh pinned request, and only one fit may exist per cell.

A wave manifest pins each request and permits at most four controller threads.
Every child still uses shared CPU/SSD admission. Finished and failed children are
recorded as they exit; failures never retry. Size the aggregate memory scope from
the actual candidate family and observed footprints before launching a wave.

Controller and native phase receipts/checkpoints are retained together. Collection
checks every expected phase, exposure and checkpoint before publication. Native
evaluation stays on the approved worker with the explicitly matching rescue
binary set. Final and diagnostic report transport preserves all numerical gates,
failed gates and convergence classifications. No outer or final-holdout response
is used by this transport.

## Deployment observation — November 8, 2026

The approved aggregate ceilings are 5, 11, 14 and 14 trainers on r3500,
r5600g, i134 and r3800x respectively. One live observation confirmed all
44 native trainers; a later observation found 43 during result finalization.
Per-job thread pools, memory limits and SSD reservations remain enforced.

All four coordinator collection observers now use the rescue-capable source.
Their replaced observers were quiesced only after active child operations had
finished; native training was not interrupted. The evaluation observer also uses
the rescue binary directory explicitly. Its first sweep must finish before its
status receipt establishes readiness. The dashboard now reads the replacement
worker status paths rather than the retired observers' files.

These observations establish placement and supervision, not completion of the
new rescue fits or their numerical evaluation. Outer/variant/bootstrap transport
still needs qualification and deployment.

`shaping-completion-qualify-worker` places the existing full native neutral
control on an approved worker, sharing CPU claims and a one-GiB pending-output
reservation with scientific fits. It refuses an existing attempt and never
retries a native failure. A supplied snapshot store enables the qualified
trainer's bounded crash recovery. The control still compares canonical native
weights against the completed frozen fit; it is not a selection candidate.

Active native epoch dumps precede CLI metadata stamping and derived-input
compaction. A finalized original checkpoint and an active raw dump therefore
cannot use the finalized-bake byte comparison directly. Keep both forms and
perform that comparison after successful CLI finalization. An absent metadata
key in an early diagnostic is not evidence of a changed weight trajectory.

For system-service launches, set finite `MemoryMax` and `MemorySwapMax=0` on
the service too. A worker's older wrapper can lack a usable user-manager
environment in this context; a printed fallback is not a hard memory cap.
