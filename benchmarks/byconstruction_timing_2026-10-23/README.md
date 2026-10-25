# Ringing-correction timing (2026-10-23, dev)

Raw paired rounds behind "Cost, and a 9% speedup found on the way" in
[byconstruction_2026-10-22.md](../byconstruction_2026-10-22.md). The rounds (95 JSONL files, 552 KB) live on dev at
`/mnt/v/output/zensim/byconstruction-2026-10-22/timing-2026-10-23/`, mirrored to
`/mnt/tower/output/zensim/byconstruction-2026-10-22/timing-2026-10-23/`. `SHA256SUMS` there lists every file
(its own sha256: `3b9983934447ad99c0836413e02436bb00bed94228626b7059682aa0daac8cb7`); `BINARIES.sha256` lists the
binaries. Summarize a pair with `scripts/byconstruction/timing_summary.py <file>` and a shuffle set with
`scripts/byconstruction/shuffle_summary.py <dir>`.

Every round: `byc_score_timing` on B's served bake (`zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin`), one
deterministic textured pair, `RAYON_NUM_THREADS=1 taskset -c 2` (8-thread files: `0-7`), 3 warmup calls, arms
alternating per round (`scripts/byconstruction/ring_timing.sh`, `RING_A`/`RING_B` set `ZENSIM_RING_FORM`).

| Files | What |
|---|---|
| `old_vs_hp4_*`, `newraw_vs_hp4_512` | first form (`hp4`, build `08930023`) against the pre-change baseline `bb445dca` |
| `old_vs_hpm_*`, `old_vs_newraw_512`, `newraw_vs_hpm_512` | `hpm` build before the alignment fix |
| `phase_hpm_*`, `phase_inl_2048` | `ZENSIM_FOLD_TIMING` phases (`scripts/byconstruction/phase_timing.py`) |
| `layout_<align>_*`, `layoutinl_*`, `inlboth_*` | `-C llvm-args=-align-all-functions=<align>` builds, without and with `OnlineMoments::block` inlined |
| `align_*` | 64-byte-aligned lane accumulators (old and new trees) |
| `shuffle/shuf<seed>_*` | five lld `--shuffle-sections=.text*=<seed>` builds per tree, before the alignment fix (`scripts/byconstruction/shuffle_study.sh`) |
| `final/shuf<seed>_*`, `final/extra_*` | the final comparison at `ec764cd7` (`scripts/byconstruction/final_shuffle.sh`) |

Source trees for the layout builds were exported from the workspace: "old" is `2ce24d8e` (pre-change), "new" is the
`hpm` code (`980fde3f`/`58a4d100`), "*al" add `#[repr(align(64))]` to the lane accumulators, "newal" in `final/` is
`ec764cd7`. Builds: `cargo build --release -p zensim-validate --bin byc_score_timing` with the RUSTFLAGS named above.
Load average stayed under 3 during the timed sections; no foreign build or training ran.
