#!/bin/bash
# Final perf parity for the ringing correction (exploration record benchmarks/byconstruction_2026-10-22.md):
# shipped old code vs the corrected code (aligned accumulators, ZENSIM_RING_FORM raw and hpm), and aligned old vs
# corrected (the ring form's own cost), each under several lld link-order shuffles.
# Usage: final_shuffle.sh LAYOUT_ROOT OUT_DIR BAKE SEEDS...   (trees: old, oldal, newal under LAYOUT_ROOT)
set -euo pipefail
L=$1; OUT=$2; BAKE=$3; shift 3
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for s in "$@"; do
  for tree in old oldal newal; do
    B=$L/bin/shuf-$tree-$s; [ -x $B/byc_score_timing ] && continue
    T=$L/target-shuf-$tree-$s
    (cd $L/$tree && CARGO_TERM_COLOR=never RUSTFLAGS="-C link-arg=-Wl,--shuffle-sections=.text*=$s" CARGO_TARGET_DIR=$T \
      cargo build --release -p zensim-validate --bin byc_score_timing > $T.log 2>&1)
    mkdir -p $B && cp -p $T/release/byc_score_timing $B/ && rm -rf $T
  done
done
echo "builds done $(date -u +%T)"
for s in "$@"; do
  b() { echo $L/bin/shuf-$1-$s/byc_score_timing; }
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b old) $(b newal) "$BAKE" 2048 8 12 "$OUT/shuf${s}_old_vs_newal-hpm_2048.jsonl"
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b old) $(b newal) "$BAKE" 512 40 12 "$OUT/shuf${s}_old_vs_newal-hpm_512.jsonl"
  RING_A=raw RING_B=raw "$H/ring_timing.sh" $(b old) $(b newal) "$BAKE" 2048 8 12 "$OUT/shuf${s}_old_vs_newal-raw_2048.jsonl"
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b oldal) $(b newal) "$BAKE" 2048 8 12 "$OUT/shuf${s}_oldal_vs_newal-hpm_2048.jsonl"
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b oldal) $(b newal) "$BAKE" 512 40 12 "$OUT/shuf${s}_oldal_vs_newal-hpm_512.jsonl"
  echo "seed $s timed $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)"
done
echo FINAL-DONE
