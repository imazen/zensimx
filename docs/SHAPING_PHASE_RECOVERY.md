# Bounded state for future native phases

New transported outer, variant and exposure-extension owners reserve one
snapshot store per registered native phase before creating an attempt. A
two-phase pretrained fit reserves both slots up front, so its second phase
cannot be crowded out by unrelated admissions. Existing claims and running
fits are preserved. The existing corrected-inner path remains unchanged.

The shared native store keeps its existing 4 GiB bound, 63 identities, and
64 MiB per-identity allowance for two retained states plus a simultaneous
write. Admission counts closed first-phase directories, active native locks,
bound phase stores and not-yet-created phase reservations. Completed stores
continue to consume capacity until archived and retired through the existing
readback/open-file gates. New phased owners also reserve four GiB above the
30 GiB SSD floor for the shared state pool.

The phase owner opts into the already qualified native full-state machinery.
Each phase binding checks its unchanged parent RUN hash, cell, binary identity,
registered phase length and actual native command. Ordinary and fresh phases
keep their existing seed streams and schedules. The native payload retains
full-precision weights, Adam, sampler cursor/RNG, learned shapers, head state
and checkpoint receipts. The two states rotate on the native 60-second and
epoch boundaries. No published model or objective changes.

The observer retains its parent-run identity for fit progress and separately
identifies the native phase for snapshot binding. A phase may use 480 epochs
only when that exact length comes from its parent schedule; legacy unphased
bindings retain their previous 120/240-epoch contract.

This makes exact continuation possible through the existing native CLI. It
does not automatically replay failed attempts or authorize recovery without
reconciling the failure and original scientific identities. One-hidden-layer
original-A training remains outside this native snapshot implementation.

Unit coverage exercises slot exhaustion across a completed first phase,
480-epoch parent authority, actual-command mismatches and existing retention
contracts. The native state engine's prior mechanical tests cover mid-epoch
and epoch-boundary recovery for plain, Softplus, frozen and joint shaping.
The frozen `0dea5f546a6e` orchestration source was deployed to the six eligible
workers and nine future extension, outer and variant owners. Each replacement
owner was verified childless and waiting before replacement, then verified live
against its exact command afterward. No active scientific fit was signaled or
replayed. The unchanged registration and selection gates still control admission.

Four completed paired-gradient engineering stores have also been retired through
the existing snapshot owner. Their full recovery states had complete durable
readback under `paired-gradient-fullstate-archive-20261109`. Retirement reread
the durable state hashes, checked the original successful bindings and native
store locks, and refused active file users. It removed sixteen worker files:
eight global state files and their eight staged duplicates, measuring
165,860,564 bytes before removal, and released four store reservations.
The durable payload and coordinator copy remain available. No training signal
was sent and no state from an active or interrupted fit was removed.
The retirement process reported `done rc=0 2s | peak-RSS 0.02GiB |
min-avail 16145MiB | peak-load 4.48`; this is metadata-cleanup resource use.
Its collected journals and removal inventory passed durable checksum readback
under `paired-gradient-state-retirement-20261109` beside the study archives.
Actual bindings for these future scientific phases remain execution evidence to
collect after admission; deployment does not establish that those fits have run.

`shaping_phase_snapshot_watch.py` retires only successful states whose parent
fits have completed immutable worker collection. It checks worker metadata
against those collected hashes, stages at most 4 GiB, and uses the existing
durable mirror with complete cold readback. Retirement checks native locks,
all state hashes and open users. A private read-only archive mount namespace
lets a worker read the actual archive while preserving its ordinary mount view;
the caller's finite memory cap, nice level and idle I/O class remain enforced.

The worker removes global and staged state copies together after the same
cold verification. The coordinator separately removes its verified staged
payload and keeps manifests, bindings, locations and retirement journals.
Only one archive batch runs at a time. A successful cycle therefore keeps
the native pool within 4 GiB per worker and each staging workspace within
4 GiB, rather than accumulating completed state payloads. Interrupted or
uncollected runs remain protected. The watcher can also retire earlier
completed one-phase stores through the same unchanged provenance gates.

The first six-worker cycle archived and retired 154 completed one-phase stores.
It also removed 3,153,443,999 bytes of coordinator staging payload after complete
durable readback; those bytes were temporary staging, not a reduction in the
coordinator's pre-cycle stored results. Metadata and retirement journals remain.
The actual future two-phase retirement path has not yet executed.

## Qualified memory sizing for future streaming refits

The completed 480-epoch ReLU, identity-start joint product and identity-start
joint separate fits each reported 0.11 GiB native peak RSS and 0.17 GiB for
the whole worker receipt. The sampled Softplus fit reported 2.42 GiB. The
resource qualification reuses these existing complete runs and their exact
native BUILD identities; no control trajectory is replayed for memory sizing.

New outer/variant requests may attach the hash-bound
`STREAMING_MEMORY_QUALIFICATION.json` profile for those three measured families
with exactly one 480-epoch phase. Their parent scope, admission allowance and
native wrapper all use 2 GiB. Allocation admission retains an extra two GiB
until the owning worker observes its first fit checkpoint, then clears only
its own warmup flag. Other workers' claims stay untouched. The existing
aggregate CPU, RAM-availability, SSD and snapshot-store guards remain in force.

Existing requests keep their original bounds. Analytic phases retain their
four-GiB doubled-exposure allocation; unqualified builds, other families and
other phase schedules keep their previous resource path. The argument proof
over all three saved native templates changes only `run-heavy --mem`.
Scientific command arguments, exposure, data, seeds, binary bytes and numerical
gates are unchanged. Future throughput under this policy is not yet measured.

The six waiting outer/variant pipelines now use frozen `13a57031ab88` source
under `streaming-memory-runtime-13a57031-20261109` beside the study. Their exact
commands, new status paths and live PIDs are retained there. Extension training,
terminal collection, bootstrap and finalist-cost owners retain their current
sources. The shared four-GiB snapshot storage budget and 34-GiB admission floor
are unchanged by the separate RAM profile.

Worker preflight verified all nine transported module hashes, standard-library-only
retirement imports and read-only NFS archive access inside a 512 MiB/no-swap scope
on all six workers. The namespace requires the system NFS mount helper. Resolve
the archive export and server address from live `findmnt` output; a coordinator's
server hostname need not resolve on every worker.

Deployment and the first retirement cycle are recorded under
`/mnt/v/output/zensim/shaping-completion-20261107/phase-recovery-pipeline-install-20261109`.
The continuously supervised owner records each independently verified batch under
`phase-snapshot-retention-0dea5f54-20261109` in the same study root.

## Read-only relocation preflight

`just shaping-relocation-preflight STUDY PLACEMENTS OUTPUT` captures evidence for
up to eight still-running, single-phase 480-epoch exposure extensions. It checks
the original phase/parent hashes, complete conditional registration, bound live
native PID/start ticks/command, native binary closure, state payload checksums,
and every completed checkpoint receipt from the selected full state. It retains
one bounded state witness and its immutable input metadata per proposed move,
under an explicit SSD reservation capped at 256 MiB. It never signals a trainer,
starts recovery, changes an attempt or alters its scientific eligibility.

A captured witness is not automatic authority to stop a healthy fit. Actual
relocation needs reconciliation of the original live controller, a then-current
state and a collection/observation path that preserves the complete attempt
history. The fleet currently assumes an attempt executes on one host. Native
state capture alone does not qualify that orchestration or cross-host trajectory
parity. Keep original trainers running while preparing a proposal.

The first actual preflight captured six active extensions without signalling or
replaying a trainer. All 537 completed checkpoint receipts matched their native
files. The full states contained 84,863,433 bytes; the complete evidence tar's
86,361,449 file bytes compressed to 9,671,899 bytes. Independent decompression
verified every state's file hash and native body digest. The source wrapper
reported 117 s, peak RSS 0.26 GiB, minimum available 34,195 MiB and peak load
16.17; this is read-only preflight cost, not a native training measurement.

The closed `relocation-proposal-evidence-20261109` archive passed durable cold
readback as 14 payload files, 9,897,870 bytes, plus receipt. Its proposed resource
placement uses two fits each on r3500, i134 and r7900x, preserving original
scientific conditions and memory caps. That proposal is pending explicit
reconciliation authority for healthy active attempts. It has not moved a fit,
qualified cross-host continuation or changed the recovered-attempt allowlist.
All six original native owners were subsequently verified still live by their
exact PID, start ticks and command.


## Finalizing a transferred complete native phase

shaping_continuation_finalize.py accepts one successful registered phase continued from a checksum-verified full optimizer snapshot at an epoch boundary. It requires unchanged scientific arguments and manifests, a matching snapshot checkpoint prefix, all native checkpoint write receipts, and complete epoch coverage. A completed tail beyond the captured optimizer state is accepted only when its raw checkpoint matches the resumed raw checkpoint bit for bit; it is counted separately. It preserves the original RUN identities, both raw logs, optimizer snapshot and prior best model; CONTINUATION.json records both execution hosts and the observed recomputed tail count. Committed optimizer-snapshot epochs are never replayed. It publishes the phase and controller RESULT receipts only after validating and retaining that evidence. This operation never executes a trainer.
