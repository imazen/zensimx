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
