# Native HDR features: stopped before bank qualification, 2026-10-29

MISSING: a valid published native feature bank, 4Kdtb source clearance,
complete input/recipe qualification, and P2/P3 fits/evaluations. No model adopted.
The owner requested closing the current step and stopping; no new run was started.

Artifacts checked on dev: `/var/tmp/byc-hdr-20261029/`.
UPIQ native rows and metadata are also retained under
`/mnt/tower/output/zensim/byc-hdr-20261029/upiq-retained-native-r1/`.
Other artifacts are not claimed fully mirrored. No R2 mirror was created.

UPIQ native extraction computed all 380 admitted rows. Its first transport
failed after computation because it checked `build_commit` rather than the
research manifest's `zensim_build_commit`. The publication-only recovery
reported DONE, but readback found a staging envelope instead of persisted
sidecars: that bank is INVALID. No rows were used in a fit.
The worker's shared staged-publication dispatch omitted the new `hdr:` namespace.
The isolated fix routes both worker execution paths through the existing
staging owner and refuses opaque native output. Its regression checks
sidecars before primary publication; end-to-end recovery is still unverified.

- Native TSV: SHA256 `668f726d4906a20c179219c3b04cf752222d961f3b670fac7fe9701af2566df8`.
- First run: `upiq-first-r1/` on the extraction host; copied ledger/log in the dev artifact root.
- Invalid recovery: `upiq-recovery-r1/` on the extraction host; envelope retained as `UPIQ_FEATURE_RESULT.json` on dev.
- Immutable binaries, source patch and lockfiles: `verified-bin/` on dev.
- Replay control and its limits: [record](hdr_training_control_2026-10-29.pointer.md).

Native UPIQ Rust tests (8) and release build passed; the worker publication
regression and release build passed. The PQ/RGBE prototype's 3 tests and
immutable-dependency release build passed, but runtime/source qualification
and broad dependency compatibility remain incomplete; its source is preserved
as unpublished WIP. Full worker tests and strict clippy were not run.
Earlier failed builds and an excluded build against dirty dependency sources
are retained, not used as feature-bank evidence. No expectation was weakened.

Next step, only if resumed: build an image containing the verified worker fix
and corrected transport, recover the stored native TSV without recomputation,
and verify the primary object plus every referenced object before accepting
the bank or scaling. Then complete source screening, remaining feature legs,
and pinned recipes before fitting. HDR-VDP-3 students' UPIQ results are not
independent because that teacher was calibrated on UPIQ.
