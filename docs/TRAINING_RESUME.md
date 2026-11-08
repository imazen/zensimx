# Bounded full-state CPU snapshots

The constrained shared-depth rank/MSE trainer supports opt-in crash snapshots
on Linux. Other operating systems explicitly refuse this option; their normal
training path remains available.
Existing frozen study executables remain unchanged. A weight-only bake is still
a warm start, not an optimizer continuation.

Set `ZENSIM_TRAIN_SNAPSHOT_ROOT` to one shared store on the active SSD. Use the
same store for all seeds on that filesystem. Preserve the immutable executable
with its snapshots; continuation verifies its exact bytes.

For the CLI, preserve the interrupted output directory and process log. Set
`ZENSIM_TRAIN_RESUME=1` and `ZENSIM_TRAIN_RESUME_CHECKPOINT_ORIGIN` to the original
explicit checkpoint directory. Keep the model/data/seed arguments unchanged,
but give `--out` and `--dump-checkpoints-dir` fresh destinations. Full snapshot
identity is checked before any prefix is copied. Only completed checkpoints
named in native write receipts, with matching SHA-256 hashes, may be copied.
The interrupted attempt remains untouched. The ordinary CLI fresh-directory
preflight remains in force. The original checkpoint directory remains the
identity anchor for subsequent continuations too.

A changed or absent identity refuses continuation before reserving a new run.
A drifted checkpoint prefix refuses copying. If optimization already completed
and the CLI finalized those files, reconcile the existing completed artifacts;
this training-resume route expects its retained raw checkpoint prefix.

Snapshots retain full f64 parameters, all Adam gradients/moments/step counters,
shared or analytic heads, learned-shape parameters and optimizer state, sampler
and dropout RNGs, epoch/pair cursor, accumulated loss/counters, best bake and
selection state, and completed checkpoint write receipts for CLI finalization.
Effective hyperparameters, executable bytes, standardized input
rows, targets, reference membership, scalers, masks and research environment bind
the identity. Initialization/scaler reconstruction is checked before restoring.

The interval is 60 seconds, checked between sampled pairs. There is also an
initial snapshot and a snapshot after each completed epoch. Serialization and
filesystem latency can extend that interval. Auxiliary losses, sample-digest
recording, alternate sampler plans, EMA and early stopping are currently refused
with snapshots enabled. This implementation covers the constrained shared-depth
owner; other training owners require separate state coverage.

Storage has fixed limits:

- Two retained snapshots and one temporary write, within 64 MiB per run.
- A 4 GiB shared reservation ceiling, including completed runs. This admits at
  most 63 distinct run identities after shared metadata allowance.
- At least 30 GiB free plus the 4 GiB shared pending allowance before admission
  and each write. Existing study output reservations still apply.
- Bounded serialization before touching a retained state; checksummed contents,
  file and directory fsync, atomic rotation and an exclusive per-run writer.

A corrupt newest state falls back to the previous intact state. An unfinished
temporary write is never a recovery source. A quota or size refusal preserves
the old snapshots. Completed states continue to consume their reservations;
archiving and retiring them requires verified durable readback and no active
users. Changing the store root to bypass a full budget is not admission.

`shaping_snapshot_retention.py bind` records the live native PID, process
identity, held snapshot lock and immutable RUN hash beside the run, outside
the native store. `stage` requires successful terminal evidence and both the
shared admission lock and the per-run writer lock before copying the retained
states. After the ordinary artifact mirror has cold-verified that stage,
`retire` rereads every durable state, checks active users and writes its removal
journal before freeing the entire completed store directory. It preserves the
shared lock, archived states, binding and scientific outputs. Interrupted runs,
active locks, unfinished writes and unknown files refuse cleanup. A partial
removal journal requires explicit reconciliation; it never triggers retraining.

The Python nonblocking lock was checked against an actual active native writer
on November 8, 2026 and correctly refused acquisition. Fixture coverage also
checks cold-readback corruption and complete reservation release. No completed
native store had yet been retired through this new owner at that observation.

Corrected worker admission counts completed store directories and pending CPU
claims against those same 63 native reservations. Actual live snapshot-lock
owners avoid double-counting. Admission waits before creating an attempt when
retirement is needed; it releases the CPU claim while waiting. This transaction
does not serialize training. The corrected capsule also pins command and
snapshot modules loaded lazily during execution.

Corrected-fit Python supervisors run in individual 512-MiB scopes, outside the
wave controller's shared limit; native trainers retain their 2-GiB scopes. RAM
admission keeps an extra startup allowance until epoch zero exists, so a newly
visible PID does not prematurely release its allocation allowance. This
staggers initialization while allowing independent training to overlap. CPU
ceilings, datasets, seeds, pair budgets and snapshot limits are unchanged.

Qualification uses real native training, with process exit during an epoch and
at an epoch boundary. ReLU, anchored Softplus, frozen input shaping and joint
input shaping must reproduce both the returned bake and the complete final
serialized optimizer state bit for bit. A separate 2,345-feature, 128-unit
fixture checks the retained size and space for a simultaneous third write.
These are small implementation controls, not scientific result replays.
