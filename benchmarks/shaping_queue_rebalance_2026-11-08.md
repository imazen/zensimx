# Shaping queue placement and snapshot retirement — November 8, 2026

The scientific study remains incomplete. At 19:11:28 UTC, the published audit
counted 693/816 primary fits trained. Outer finalist, source-curve, shuffle and
human-only evidence still require the registered inner selections and rescues.
No inference about quality follows from the scheduling changes below.

Source: `68dbeb39e34c`; study root on the coordinator:
`/mnt/v/output/zensim/shaping-completion-20261107`.

The original idle-wave boundary moved 12 unopened conditions from r5600g, six
each to i270 and r7900x. A separate mixed-queue boundary then moved 32 unopened
conditions from i134, 16 each to those workers. It stopped 13 queue parents
under the existing short admission locks and signalled no native processes.
Every moved condition had neither an attempt nor a CPU reservation. Existing
attempts stayed on their original hosts. The source, coordinator and destination
placements were reconciled before queue resumption. Original requests, manifests
and assignment backups remain preserved; superseded requests have explicit
annotations pointing to their replacements. No scientific fit was replayed.

At 19:13:43 UTC, the native observer measured 79 simultaneous fits: i134 14,
i270 20, r3800x 14, r5600g 11 and r7900x 20. r3500 had no native trainer and
subsequently started four scalar-panel workers in two separately capped jobs.
These observations establish concurrency, not a throughput ratio or completion ETA.

Earlier two-second process CPU samples at 19:10:16–17 UTC measured:

| Worker | Stable native processes | Aggregate native busy cores |
|---|---:|---:|
| i270 | 16 | 15.912761 |
| r7900x | 9 | 8.928809 |
| i134 | 14 | 13.423540 |
| r5600g | 11 | 9.401208 |
| r3800x | 14 | 13.635307 |

The fast workers were still warming additional fits during those samples.
The tightest study SSD was i270: 62,423,932,928 bytes free, above the retained
30-GiB reserve. No pixel maps were deleted. dev received no new native training,
evaluation or benchmark workload; i265 and r5900xt were not contacted.

Snapshot batches preserved 62 completed stores from i270 and 61 from r7900x.
Their durable archives were cold-read and checksum-verified before local store
retirement. Open-file qualification covered file-descriptor and mapping aliases.
Retirement wrapper measurements were respectively 28 s / peak RSS 0.07 GiB /
minimum available 26,372 MiB, and 37 s / peak RSS 0.06 GiB / minimum available
14,919 MiB. Failed initial checks remain preserved. These wrapper measurements
describe retirement, not model training memory.

Verified metadata archives, under the same relative NAS study root:

| Archive | Artifacts before receipt | Bytes | Receipt SHA-256 |
|---|---:|---:|---|
| idle-transition-r5600g-20261108 | 62 | 850206 | `9fd259aa01ff8839eb368c6e5cc1f5bb57baa4275e136e0569edeb94011b5cfe` |
| queue-transition-i134-20261108 | 138 | 1900992 | `5cd034d02354836411308dfc4ea30110c7df7174af51b9b6f23f9a7ee02ddcc3` |
| completed-snapshot-retirement-proof-i270-20261108 | 65 | 136155 | `2b16408d4a312d2b9c2f89d186cdcccb4412476f01a033d4ae09421d99f42877` |
| completed-snapshot-retirement-proof-r7900x-20261108 | 64 | 134353 | `aeb01bd4ba124db9c9513c54a576b3480947a1e1561652e49aded1bfe0adf4b3` |

`just shaping-queue-placement-check` passed five checks, including continued
progress of a real child while its parent was stopped, and rejection of an old
placement before creating an attempt. Wrapper: 1 s, peak RSS 0.02 GiB, minimum
available 43,997 MiB. Native model identity, seeds, data, exposure and gates were
unchanged. The gallery preparation published 804 scalar scatter sets; 144 missing
native detailed panels were queued on r3500, preserving the previous 660.
