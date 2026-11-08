# Bounded full-state CPU snapshots

The constrained shared-depth rank/MSE trainer supports opt-in crash snapshots
on Linux. Other operating systems explicitly refuse this option; their normal
training path remains available.
Existing frozen study executables remain unchanged. A weight-only bake is still
a warm start, not an optimizer continuation.

Set `ZENSIM_TRAIN_SNAPSHOT_ROOT` to one shared store on the active SSD. Use the
same store for all seeds on that filesystem. Set `ZENSIM_TRAIN_RESUME=1` only
when continuing an interrupted run with identical executable, arguments, inputs,
feature mask and research manifests. A missing or changed identity refuses
continuation before reserving a new run. Keep the original output paths and
checkpoint directory; preserve the interrupted process log separately.

Snapshots retain full f64 parameters, all Adam gradients/moments/step counters,
shared or analytic heads, learned-shape parameters and optimizer state, sampler
and dropout RNGs, epoch/pair cursor, accumulated loss/counters, best bake and
selection state. Effective hyperparameters, executable bytes, standardized input
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

Qualification uses real native training, with process exit during an epoch and
at an epoch boundary. ReLU, anchored Softplus, frozen input shaping and joint
input shaping must reproduce both the returned bake and the complete final
serialized optimizer state bit for bit. A separate 2,345-feature, 128-unit
fixture checks the retained size and space for a simultaneous third write.
These are small implementation controls, not scientific result replays.
