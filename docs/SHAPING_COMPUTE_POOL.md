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

The `85fd619d` source was deployed to the five remaining compatible workers.
All eleven replacement metadata owners were observed live; long first scoring
sweeps are tracked by their actual PID/child even before their first status
file. Fold-1 preparation/scalar work targets r7900x. Future finalist cost work
targets r3800x, retaining the cost owner's existing refusal of r7900x. The first
cost replacement's refusal occurred before a fit, score or benchmark started.
The closed resource-change evidence passed durable cold checksum readback as
59 payload files, 1,074,485 bytes, plus its receipt, under
`i270-benchmark-withdrawal-evidence-20261109` beside the study archives.

Completed ReLU/Softplus exposure extensions can be evaluated on another eligible
worker. The transport keeps the original training host and RUN bytes, binds the
complete extension resolution, and includes the native phase receipt and its
head manifest alongside final/diagnostic checkpoints. The existing 128 MiB
input limit, no-overwrite transfer, native build checks and scoring gates remain.
An explicit `--cell` scope lets a new owner handle only the withdrawn host's
pending fits without duplicating another owner's evaluations. Primary transport
continues to accept only the previously qualified ReLU controls; other stages
and families still refuse cross-host requests.

All admitted i270 native fits drained successfully on November 9; the final fit
finished at 12:15:22 UTC and the following live scan found zero native trainers.
The four remaining control evaluations were imported from r7900x without any
training replay. The closed transport evidence passed durable cold checksum
readback: 56 payload files, 1,382,319 bytes, plus its receipt, under
`extension-transport-evidence-20261109`. Its capture includes the first two
imports; later scientific artifacts use the ordinary archive owners.

Collected-fit rows in the statistics page use RUN.json's recorded training host.
Previously, a completed worker fit could be labeled with the coordinator host
after its copied files were observed locally. This display correction preserves
the actual training origin when evaluation occurs on another worker.

The four relocated evaluations subsequently completed native, converted-bake and
canonical-panel archival: twelve original cold-readback mirror receipts, with
current manifest identities checked in `extension-relocation-complete-evidence-20261109`
(19 payload files, 734,663 bytes, plus receipt). The deployed training-origin
fix and its independent HTTP verification passed cold readback separately in
`dashboard-origin-evidence-20261109` (17 payload files, 483,041 bytes, plus receipt).
The first publisher launch rejected an unregistered optional supervisor label
before publication; the corrected launch and original failed log are preserved.
Scientific fits and scoring were unaffected by that launch refusal.

Scalar-wave preparation now validates each complete immutable extension fold
once and includes every cell, rather than reconstructing that fold separately
for each cell. Selection, policy, matrix and provenance checks use the same
full-fold validator. Duplicate IDs and registration failures still refuse.
Four preparation/launch/watch checks passed. The actual three-fold check retained
all 60 registered extension cells exactly; run-heavy reported 9 s, peak RSS
0.11 GiB and minimum available 44,067 MiB. This is metadata preparation cost.
Scientific models and native scoring arithmetic are unchanged.
