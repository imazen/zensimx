#!/usr/bin/env bash
# Run bake_compare for the packed median seed against current ships.
# Usage: bake_compare_vs_ships.sh <packed_bake>
set -euo pipefail
PACKED="${1:?packed_bake}"

BC=/home/lilith/work/zen/zensimx/target/release/bake_compare
OUT_DIR=/mnt/v/zen/zensim-eval/exp_multi_codec_2026-05-18/bake_compare
mkdir -p "$OUT_DIR"

# Current ships:
COMPRESSION_SHIP=/home/lilith/work/zen/zensimx/zensim/weights/v_compression_persample_2026-05-18.bin
BALANCED_SHIP=/home/lilith/work/zen/zensimx/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin

for ship in compression:"$COMPRESSION_SHIP" balanced:"$BALANCED_SHIP"; do
    name="${ship%%:*}"
    path="${ship#*:}"
    if [[ ! -f "$path" ]]; then
        echo "MISSING ship bake: $path"
        continue
    fi
    OUT="$OUT_DIR/vs_${name}_ship.md"
    echo "=== bake_compare vs ${name} ship ==="
    "$BC" --a "$PACKED" --b "$path" \
        --corpora cid22,kadid,tid,konjnd,aic3 \
        --bands 10 \
        --bootstrap-resamples 1000 \
        --output "$OUT" \
        --seed 42 2>&1 | tail -20
    echo "wrote $OUT"
    echo
done
