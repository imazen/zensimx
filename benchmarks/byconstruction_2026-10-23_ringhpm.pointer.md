# Pointer: ringing-correction and descriptor data (2026-10-23)

Record: [byconstruction_2026-10-22.md](byconstruction_2026-10-22.md) ("Ringing corrected" and "Next feature"
sections). Paths checked on **dev** only (`/mnt/v` differs between hosts); mirrored to tower at
`/mnt/tower/output/zensim/byconstruction-2026-10-22/`.

| What | Path (dev, under `/mnt/v/output/zensim/byconstruction-2026-10-22/`) |
|---|---|
| `hpm` bank (8 sets, `rev5_bank.py` layout; per-set `verify_*.json`: exactly the 12 ringing slots differ from `/var/tmp/rev5-featbank`) | `ringhpm/bank/`, `ringhpm/verify_*.json` |
| `hpm` coverage-pool extraction (44,000 pairs) | `ringhpm/pool/` |
| Derived training root (12 ringing columns replaced; frozen/receipt re-pinned) | `ringhpm/root/` |
| Root widened with the byc_globaff descriptor at f1867-f1890 | `ringhpm/root_globaff/` |
| byc_globaff sidecars (per set + pool) and their pair lists | `ringhpm/globaff/`, `ringhpm/globaff_pairs/` |
| Feature probe under `hpm` | `feature-probe-hpm/` |
| Mixture-test table (hpm dumps + descriptor) for table-route scoring | `mixture_table_hpm/` |
| Mixture scores for the new arms (`scores_{Bhpm,FRhpm,Mhpm,Thpm,Tghpm,Bghpm}_*.json`) | `mixture/` |
| Timing rounds | `timing-2026-10-23/` (see `benchmarks/byconstruction_timing_2026-10-23/README.md`) |

Cells (train.log, bakes, held-out predictions, `byc_record.json`) are in `~/tmp/byc/cells/{Bhpm,FRhpm,Mhpm,Thpm,Tghpm,Bghpm}/`
on dev and mirrored to `cells/` beside the earlier arms. Pinned binaries with `SHA256SUMS` and `BUILD_COMMIT`:
`/var/tmp/byconstruction-bin/{980fde3fd887,35e902546c52,ea15b78df4a1,58a4d100ad36}/`.
