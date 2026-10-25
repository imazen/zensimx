#!/usr/bin/env bash
# Before/after bit identity for speed work: the score and every emitted feature's bits for B (revision 5) and Tga
# (revision 6) on synthetic odd geometries and five gb82 photos, at every x86-64 tier with one thread, plus v4x with
# eight. Compare two output dirs with perf_bits_compare.py. Record: benchmarks/byconstruction_2026-10-22.md.
# Usage: perf_bits.sh BIN OUT_DIR   (B_BAKE / TGA_BAKE override the two bakes, e.g. revision-7 stamps)
# PERF_BITS_MODELS: optional TSV (name, bake|profile, path|name). PERF_BITS_HDR=1
# sends the same eleven pairs through the named profile's absolute-nits entry.
set -euo pipefail
BIN=$(realpath "$1")
OUT=$(realpath -m "$2")
mkdir -p "$OUT"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
B=${B_BAKE:-$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin}
TGA=${TGA_BAKE:-$HOME/tmp/byc/parity/tga_full_rev6.bin}
G=$HOME/.cache/codec-corpus/v1/gb82
unset ZENSIM_FORMULA_REV ZENSIM_RING_FORM
run() { # name kind model tier threads tag args...
  local name=$1 kind=$2 model=$3 tier=$4 threads=$5 tag=$6
  shift 6
  local hdr=()
  if [[ ${PERF_BITS_HDR:-0} == 1 ]]; then hdr=(--hdr-nits); fi
  echo "$name pair=$tag tier=$tier threads=$threads"
  RAYON_NUM_THREADS=$threads "$BIN" "--$kind" "$model" "${hdr[@]}" --iters 1 --warmup 0 --tier "$tier" \
    --dump "$OUT/${name}_${tag}_${tier}_t${threads}.hex" "$@" > /dev/null
}
models() {
  if [[ -n ${PERF_BITS_MODELS:-} ]]; then
    cat "$PERF_BITS_MODELS"
  else
    printf 'b\tbake\t%s\ntga\tbake\t%s\n' "$B" "$TGA"
  fi
}
while IFS=$'\t' read -r name kind model; do
  [[ -n $name && ( $kind == bake || $kind == profile ) && -n $model ]] || exit 2
  for tier in v4x v4 v3 scalar; do
    for threads in 1 8; do
      [ "$threads" = 8 ] && [ "$tier" != v4x ] && continue
      for geo in 97x63 131x65 255x129 509x311 1031x517 2050x1030; do
        run "$name" "$kind" "$model" "$tier" "$threads" "$geo" --width "${geo%x*}" --height "${geo#*x}"
      done
      for img in baby bulb city dog flowers; do
        run "$name" "$kind" "$model" "$tier" "$threads" "$img" --image "$G/$img-lossless.png"
      done
    done
  done
done < <(models)
echo "$(find "$OUT" -maxdepth 1 -name '*.hex' | wc -l) dumps in $OUT"
