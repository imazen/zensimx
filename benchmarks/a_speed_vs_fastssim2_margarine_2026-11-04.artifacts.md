# Public A peer measurement artifacts — 2026-11-04

Measurement host: **i270**. The completed run remains at
`/home/lilith/tmp/zensim-bench` on i270. Persistent copies:

- **dev**: `/mnt/v/output/zensim/a-speed-vs-fastssim2-margarine-2026-11-04-i270/`
- **tower**, verified through its NFS mount on dev:
  `/mnt/tower/output/zensim/a-speed-vs-fastssim2-margarine-2026-11-04-i270/`

Both copies passed **880/880** checks from `SHA256SUMS` on 2026-11-04.
The inventory covers **2,149,834,475 bytes**, excluding the inventory itself,
Cargo targets and Git metadata. `SHA256SUMS` SHA-256:

```text
ccfa8d480dc0e006e553880592316764b2443a62598751bca07646189adb310c
```

The archive preserves all raw paired rounds, full collector and gate logs,
24 isolated RSS logs and GNU-time records, 80 RGB8/JPEG fixture pairs,
source PNG payloads and provenance, source/header/hash checks, original and
final-layout harness sources, frozen binaries, build flags and Cargo artifact
identity, live PID/lock supervision, and the two separate command-startup
runs. `FINAL_VALIDATION.json` records completed coverage and the verified
absence of remaining run processes. `RELEASED.json` records release of the
owner-reserved i270 window.

The interrupted foreign-build attempt, the excluded completed tiny cell,
and the failed LFS-pointer preparation remain in separately named backup
directories with invalidation records. They contribute no reported timings.
Full process logs are retained in these external copies rather than git.

Frozen scoring executable SHA-256:
`489a3ceabf0b85abbf58129309d922129f99fa94309aae73fdcf77b6b2284c3b`.
The committed [summary](a_speed_vs_fastssim2_margarine_2026-11-04.json)
matches archived `SUMMARY.json`, SHA-256:
`592829f0a2a9716b3b143fad10707ff92b3a7ee934b16c0ec1f3baa3d736f0bf`.

Recheck either mounted copy from its root with `sha256sum -c SHA256SUMS`.
Recompute the [report](a_speed_vs_fastssim2_margarine_2026-11-04.md) using
`just a-speed-report <artifact-root> <output.md>`.
