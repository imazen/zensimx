# Threshold-coordinate optimization probe

Before fitting, register two short TRAIN-only conditions with
`shaping_threshold_probe.py`. Use completed inner-0-0 separate/frozen setting-3
seeds 1 and 2: one previously collapsed, one surviving. These are purposively
chosen optimization examples, not a representative quality comparison.

Use the qualified corrected-initializer executable, each seed's original frozen
bank, keep list, five TRAIN groups, sampler/init seeds and optimizer controls.
Run two epochs of 50,000 sampled pairs. Compare native per-epoch loss and the
number of nonzero output weights against the existing original epoch-0/1 bakes
and logs. Reuse those observations without replaying the original fits. No outer
or final-holdout responses are read. Do not select a checkpoint or change the
registered primary study based on this probe.

The executable also includes qualified arithmetic-preserving training
optimizations and full-state snapshots. The registration pins its bytes, all
input/provenance files and the driver. This is an initialization diagnostic,
not a clean quality attribution to a single source-code edit. A surviving short
fit does not establish convergence, transfer benefit or architecture superiority.

The two runs share the existing CPU/RAM admission owner and immutable inputs.
Each reserves one GiB for outputs; snapshots use one shared four-GiB store,
64 MiB per run including a pending write. Preserve 30 GiB free and existing
pending reservations. Enable two concurrent seeds only when aggregate admission
allows them. Preserve both failures and successes; archive and cold-readback
verify the resulting wave.
