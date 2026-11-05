# A parallel gate artifacts — 2026-11-04

Final scope: **49/50 admitted cells**, 980 matching serial/parallel score
comparisons. The gated 12 MP cell is not measured; its interrupted and
zero-sample attempts are retained. The [report](a_parallel_gate_2026-11-04.md)
contains the measured tables and a one-command idle-host retry recipe.

## Verified locations

Checked from dev on 2026-11-05 at 04:20 UTC:

- Local copy: `/mnt/v/output/zensim/a-parallel-gate-2026-11-04-i270/raw/`.
- Canonical mirror: `/mnt/tower/output/zensim/a-parallel-gate-2026-11-04-i270/raw/`.
- Original capture on i270: `$HOME/tmp/zensim-pargate/`; its inventory was
  copied back after finalization. No benchmark remains running there.
- R2: not uploaded. These fixture-bearing archives remain private.

`SHA256SUMS` has SHA-256
`4341279a074c8d49b61ab4db094d3cac5320f6b07407c046c14047c1d21a85a5`.
It inventories **15,912 files / 11,385,804,645 logical bytes**. Hard links
retain 11,977 distinct file inodes / 2,438,020,410 data bytes in the verified
mirror. The inventory excludes VCS metadata, root Cargo output and the live
ownership lock; preserved nested Cargo reference controls are included.

Every inventoried mirror file was hash-verified, with zero mismatches, and
three additional random files were independently read and verified.
`VERIFY.json` beside `raw/` in both copies records the complete check and
sample paths. Source, local and mirror inventory hashes match.

## Contents and provenance

The archive retains all 260 fixture pairs and q40 JPEGs, raw paired rounds,
latency/score/resource logs, source snapshots, manifests, model and compiler
identities, validation logs, split parity TSVs, hardware-counter captures,
failed attempts, and preserved/rebuilt reference executables. Fixture aliases
were materialized as hard links without changing bytes. `FINAL_SCOPE.json`
records the authorized missing cell; `REBOOT_EXISTING_CELLS.json` records the
49 admitted cells and hashes of their unchanged outputs.

- Research gate: `bfabf0d50e312e866a719a36c214c4bdba7adbd1`.
- Public gate on `profile-f`: `398578fc5d513db7ced45062e152c68acdd3edef`.
- Baseline public source: `85bd08d1587161cccd64bda88c00567a0d12d215`.
- Paired instrument: `f3e19a475729ff842f0064166df54f90b25dfeb4`.
- Before timing binary SHA-256:
  `e78d47c13428618da4ee4cf7e5656860bcf349b3564eb40530bf825d3dabc791`.
- After timing binary SHA-256:
  `d03774877de2e095768d5ddf98673246588dfe5fdecbf782611ddb9ff7fa3dc6`.

Both timed binaries are original. An auxiliary public checker was rebuilt
from its unchanged frozen source after the initial cache preservation miss;
its complete raw TSV is byte-identical to the original. The original checker
ELF hash was not recorded. The report documents that limitation explicitly.

The corrected `RETRY_ONLY_12MP.sh` beside `raw/` has SHA-256
`5a125030679a3e3f537ef7d7dfbd72024059c5988db28433b06262affa229226`. Its shell syntax, referenced paths and frozen binary hash
were checked without running timing. `RETRY_RECIPE_NOTE.json` distinguishes
it from the earlier unexecuted draft retained inside the raw archive. The
tracked report contains the corrected command.
