# Adam engineering evidence

Findings and commands: [measurement report](adam_denormal_2026-10-26.md).
These are engineering replays of a frozen control, not new canonical training
data or model qualification evidence. Failed and interrupted experiments retain
their original status.

## Verified Intel archive

Observed on i265, 2026-10-26 08:06Z:

- Working evidence: `/home/lilith/tmp/adam-denormal-20261026/`.
- Archive: `/home/lilith/tmp/adam-denormal-20261026/archive-i265-20261026-0806/`.
- NAS mirror: `/mnt/tower/output/zensim-adam-denormal-20261026/intel-i265/`.
- `SHA256SUMS` SHA-256:
  `a1db40ce93ce19c9c80971e8a876b18afc42196167f8825f6c04dccdbad99478`.

The archive includes frozen reference bakes and argv, preserved executables and
source snapshots, profiler recordings, complete Intel after-phase outputs,
exact byte-offset comparisons and native component inspections. Previously
completed dev comparisons already copied to i265 are included. Every file was
checksum-verified at both the archive and NAS destination using
`just adam-denormal-manifest` and `just adam-denormal-mirror`.

The copy ran directly from i265 to NAS, without using dev or its heavy lock.
Receipt: `ARCHIVE_I265_RECEIPT.json` in the i265 working evidence directory.
The initial attempt could not read two root-owned perf recordings; the retry
read those with elevated privileges and verified the entire archive.
Resource record for the successful copy and verification:
`run-heavy: done rc=0 10s | peak-RSS 0.02GiB | min-avail 24569MiB | peak-load 0.25`.

## Selected arithmetic and incomplete evidence

Selected kernel source SHA-256:
`4dc9e41f303cd021950936484d7e1a81f537f9282e5301394b6eb13edd06b91e`.
Selected trainer executable SHA-256:
`c82bdf978fe05c99f8e0d1a7f8788b15408def271d2145e86c2c3e1325a5e44b`.
Arithmetic tag: `adam-f64-normal-moments-v1`. Main selected this kernel in
`088ef0d363e9`; the later classification experiment is retained as rejected.

The final dev pair is interrupted and scientifically incomplete. Its original
outputs and interruption sidecars remain at
`/mnt/v/output/zensim/adam-denormal-20261026/dev/` on dev. They were not copied
by the canceled archive waiter and are not part of the verified Intel archive.
No dev benchmark rerun is authorized; subsequent benchmarking uses other hosts.

The matched i134 full control remains in progress under
`/home/lilith/tmp/adam-denormal-20261026/` on i134, with owned input and coverage
snapshots. It is not represented as a completed pair in this archive.
No R2 mirror was made for these engineering artifacts.
