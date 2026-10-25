#!/usr/bin/env bash
# Before/after bit identity for speed work: the score and every emitted feature's bits for B (revision 5) and Tga
# (revision 6) on synthetic odd geometries and five gb82 photos, at every x86-64 tier with one thread, plus v4x with
# eight. Compare two output dirs with perf_bits_compare.py. Record: benchmarks/byconstruction_2026-10-22.md.
# Usage: perf_bits.sh BIN OUT_DIR   (B_BAKE / TGA_BAKE override the two bakes, e.g. revision-7 stamps)
set -euo pipefail
BIN=$1
OUT=$2
mkdir -p "$OUT"
cd ~/work/zen/zensim--byconstruction
B=${B_BAKE:-$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin}
TGA=${TGA_BAKE:-$HOME/tmp/byc/parity/tga_full_rev6.bin}
G=$HOME/.cache/codec-corpus/v1/gb82
unset ZENSIM_FORMULA_REV ZENSIM_RING_FORM
run() { # name bake tier threads tag args...
  local name=$1 bake=$2 tier=$3 threads=$4 tag=$5
  shift 5
  RAYON_NUM_THREADS=$threads "$BIN" --bake "$bake" --iters 1 --warmup 0 --tier "$tier" \
    --dump "$OUT/${name}_${tag}_${tier}_t${threads}.hex" "$@" > /dev/null
}
for bake in "b:$B" "tga:$TGA"; do
  name=${bake%%:*}
  path=${bake#*:}
  for tier in v4x v4 v3 scalar; do
    for threads in 1 8; do
      [ "$threads" = 8 ] && [ "$tier" != v4x ] && continue
      for geo in 97x63 131x65 255x129 509x311 1031x517 2050x1030; do
        run "$name" "$path" "$tier" "$threads" "$geo" --width "${geo%x*}" --height "${geo#*x}"
      done
      for img in baby bulb city dog flowers; do
        run "$name" "$path" "$tier" "$threads" "$img" --image "$G/$img-lossless.png"
      done
    done
  done
done
echo "$(ls "$OUT" | wc -l) dumps in $OUT"
