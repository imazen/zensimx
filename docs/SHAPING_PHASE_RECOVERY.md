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
Actual phase bindings and later snapshot retirement remain execution evidence
to collect after deploying the new owners; these unit tests do not establish
that future scientific fits have run.

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
