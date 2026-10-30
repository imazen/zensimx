# zensim_mlp_train lazy fx product columns: controls and memory (2026-10-30)

Change: `10d087cd` (`Fx1Declaration::extend_compact_lazy`). The trainer keeps only the
declaration's stored read set resident and computes each product column where a row is read,
with the same single f32 multiply (`derived_inputs::product_f32`). `extend_compact` (materializing)
is unchanged for e36 and the oracle tests.

Binary: release build of `10d087cd` on i265, trainer sha256 `cfac5751…`, deployed with the round-2
tools as `/var/tmp/zensim-bin-10d087cd/` (SHA256SUMS inside) on i265, i270, r3800x and dev.

## Byte-identical controls (ZENSIM_MAX_TIER=v3, RAYON_NUM_THREADS=1)

Each control retrains one stored cell (KADID fold, seed 0) and compares the prediction file and the
bake with `zentrain.repro` stripped (`bake_dial_refit strip --key zentrain.repro`).

| control | host | prediction | stripped bake | result |
|---|---|---|---|---|
| Mlic (fx1) | i265 | `0cae0f6f…` = stored | `0e03eaf0…` = stored | pass |
| Mlicfx2 (fx2) | i265 | `b9deffa0…` = stored | = stored | pass |
| Mlicfx2 (fx2) | i270 | `b9deffa0…` = stored | `9a70905f…` = stored | pass |
| Mlicfx2 (fx2) | r3800x | `b9deffa0…` = stored | `9a70905f…` = stored | pass |

## Memory and time (`/usr/bin/time -v`, maximum resident set size)

| cell | host | wrapper | old trainer | new trainer | wall old / new |
|---|---|---|---|---|---|
| Mlic (fx1) | i265 | heaptrack + time -v | 2,007,204 KiB | 911,808 KiB | 48:29 / 48:03 |
| Mlicfx2 (fx2) | i265 | heaptrack + time -v | 3,875,556 KiB | 1,873,912 KiB | 52:47 / 54:12 |
| Mlicfx2 (fx2) | i270 | time -v only | not run | 1,868,600 KiB | — / 51:40 |
| Mlicfx2 (fx2) | r3800x | time -v only | not run | 1,812,868 KiB | — / 59:48 |

Peak memory falls by 55% (fx1) and 52% (fx2). Wall time is unchanged within the noise of
shared hosts; this change buys trainers per host, not speed per trainer. Old binaries:
`p1fx2/bin` trainer `4347d73e…` (fx1 control) and the stored fx2 cell's trainer. Old/new runs on i265
ran at the same time on the same host. Scripts: Devin lane `~/tmp/devin-lowmem/runctl.sh` (i265) and
coordinator `~/tmp/lowmem-ctl/fx2ctl.sh` (i270, r3800x).
