#!/bin/bash
# Layout-randomized timing: build byc_score_timing for each source tree under several link-time function-order
# shuffles (lld --shuffle-sections), then time paired arms per shuffle seed. Whole-walk timing between two builds
# moves several percent with placement alone; averaging over shuffles separates a change's own cost from it.
# Exploration record: benchmarks/byconstruction_2026-10-22.md.
# Usage: shuffle_study.sh LAYOUT_ROOT OUT_DIR BAKE SEEDS...   (trees: LAYOUT_ROOT/{old,new,oldinl,newinl})
set -euo pipefail
L=$1; OUT=$2; BAKE=$3; shift 3
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for s in "$@"; do
  for tree in old new oldinl newinl; do
    T=$L/target-shuf-$tree-$s; B=$L/bin/shuf-$tree-$s
    [ -x $B/byc_score_timing ] && continue
    (cd $L/$tree && CARGO_TERM_COLOR=never RUSTFLAGS="-C link-arg=-Wl,--shuffle-sections=.text*=$s" CARGO_TARGET_DIR=$T \
      cargo build --release -p zensim-validate --bin byc_score_timing > $T.log 2>&1)
    mkdir -p $B && cp -p $T/release/byc_score_timing $B/ && rm -rf $T
  done
done
echo "builds done $(date -u +%T)"
for s in "$@"; do
  b() { echo $L/bin/shuf-$1-$s/byc_score_timing; }
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b old) $(b new) "$BAKE" 2048 8 12 "$OUT/shuf${s}_old_vs_new_2048.jsonl"
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b old) $(b new) "$BAKE" 512 40 12 "$OUT/shuf${s}_old_vs_new_512.jsonl"
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b oldinl) $(b newinl) "$BAKE" 2048 8 12 "$OUT/shuf${s}_oldinl_vs_newinl_2048.jsonl"
  RING_A=raw RING_B=raw "$H/ring_timing.sh" $(b old) $(b oldinl) "$BAKE" 2048 8 12 "$OUT/shuf${s}_old_vs_oldinl_2048.jsonl"
  RING_A=raw RING_B=hpm "$H/ring_timing.sh" $(b new) $(b new) "$BAKE" 2048 8 12 "$OUT/shuf${s}_newraw_vs_newhpm_2048.jsonl"
  echo "seed $s timed $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)"
done
echo SHUFFLE-DONE
