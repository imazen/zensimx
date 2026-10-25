# Ringing cost timing (2026-10-22, dev)

Raw paired rounds behind "Cost of the ringing fix, and of dropping ringing" in
[byconstruction_2026-10-22.md](../byconstruction_2026-10-22.md). Each line is one round of one arm:
`byc_score_timing` output (median/p25/p75 ms over the round's calls, score, `hp_ring`).
Summarize with `python3 scripts/byconstruction/timing_summary.py <file> [--arms C,N]`.

Builds come from the source in the commit that adds this directory. Afterwards
`zensim-validate/build.rs` gained a `check-cfg` line for `byc_hp_ring`, which only silences a lint.

| Build | Command | sha256 |
|---|---|---|
| baseline | `CARGO_TARGET_DIR=/var/tmp/byconstruction-target run-heavy -- cargo build --locked --release -p zensim-validate --bin byc_score_timing` | `bb445dca30db399e2cda17d890c6cb915f0125bb82e99086f9cf550933b84d2a` |
| high-pass ringing | same, with `RUSTFLAGS=--cfg byc_hp_ring` and `CARGO_TARGET_DIR=/var/tmp/byconstruction-target-hp` | `482aa4cb5932a6f8a019f81f67e47b5ca7a061aeffd6e505cf1bbcc93aabf972` |

| Files | Arms | Bake(s) |
|---|---|---|
| `ring_timing_{512,2048}.jsonl` | A = baseline, B = high-pass | B's served bake `zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin` (`89cb4861…`) |
| `bnr_timing_{512,2048}.jsonl` | C = `base-c`, N = Bnr (both baseline build) | `base-c` kadid_s0 dense bake `7bd14f569f04e8d8…` (E34 evaluation predictions); Bnr kadid_s0 dense bake `3b7af5906272f3a0…` (cells mirror) |

Commands: `scripts/byconstruction/ring_timing.sh <A_BIN> <B_BIN> <BAKE> 512 40 12 <out>` and
`... 2048 8 12 <out>`. The Bnr pair was run with the same loop inline (arm labels C/N); the
script's optional 8th argument (`B_BAKE`) now reproduces it with labels A/B.
Single thread, `taskset -c 2`, 3 warmup calls per round, 12 rounds with alternating arm order.
Load average 3–8 (decaying from this lane's finished jobs; no foreign build or training).

`perf_2048_symbols.txt`: flat profile (top 20 symbols) of the baseline binary at 2048², single thread;
the raw `perf.data` is on dev at `/mnt/v/output/zensim/byconstruction-2026-10-22/perf/perf_2048.data`.
