# Owner-expanded compute placement — November 8, 2026

The owner authorized all other known machines and excluded dev, i265 and
r5900xt from new compute or probes. The private study records the exact owner
message and its hash in `COMPUTE_POOL.json`; the old reduced registration stays
unchanged. Coordinator metadata and the statistics page remain local.

This amendment changes placement only. Frozen data, seeds, native binary SHA,
feature regimes, budgets, splits, selectors and qualification gates stay intact.
The transport supports the existing Linux/x86 runtime on the original four
workers plus r7900x and i270. Other authorized platforms need a compatible
qualified runtime/executor before fitting; an allowed hostname alone does not
establish artifact compatibility.

The original aggregate ceilings 5/6, 11/12, 14/16 and 14/16 remain unchanged.
Newly admitted compatible workers use ordinary `nproc - 4` headroom, resolved
at admission, and retain per-job thread limits, finite memory caps, shared
RAM/CPU admission and the 30-GiB SSD reserve. Existing other jobs still count
toward practical capacity. No unrelated process may be killed to create room.

The owner also authorized deleting actual metric image maps when space is
needed. Scalar scores, label-calibration tables and learned input-shaping
parameters are required data and are not pixel maps. Current generated artifacts
remain preserved unless covered by that explicit map permission or the study's
verified-completed-output retention policy.
## Resource withdrawal overlay

An owner reservation can remove a worker from new compute through
`study/COMPUTE_WITHDRAWALS.json`. It binds the plan, an exact direction record
and its hash. The original capsule-bound `COMPUTE_POOL.json` remains unchanged,
so existing attempts, imports and archives retain their original provenance.
New placements exclude withdrawn hosts before probing them. Outer/variant and
bootstrap placement use the reduced live host set. Pending evaluations on a
withdrawn host stay explicitly pending without transfer or remote execution.

On November 9 the owner reserved i270 for benchmarking. New work must use the
other eligible workers; the previously admitted scientific fits remain protected.
This is a resource change, not a change to data, seeds, exposure or numerical
gates. Withdrawal, placement and evaluation checks pass on an eligible worker.

