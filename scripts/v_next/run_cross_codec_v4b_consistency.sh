#!/usr/bin/env bash
# EXP-CROSS-CODEC-V4 T=63 cross-codec consistency eval driver.
#
# Runs `cross_codec_consistency.py` on each V4 bake at T=63 with n=20
# images. V4 bakes carry `zentrain.tanh_output_head` metadata so the
# runtime score is natively [0, 100] — no affine calibration step needed
# (unlike V3 which lived under `calibrated/` after affine fitting).
#
# Reports mean butter_max, butter_p3 across the 3-codec pairings — the V4
# ship gate requires butter_max < 2.5 OR butter_p3 < 2.5.
set -euo pipefail

V4_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v4b_2026-05-19"
OUT_DIR="${V4_DIR}/cross_codec_t63"
mkdir -p "${OUT_DIR}"

TOOL=/home/lilith/work/zen/zensimx/target/release/predict_features_with_bake
ZEN_METRICS=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
CONSISTENCY=/home/lilith/work/zen/zensimx/scripts/v_next/cross_codec_consistency.py

for bake in "${V4_DIR}"/cc4v4b_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_tsv="${OUT_DIR}/${name}_t63_n20.tsv"
    if [ -f "$out_tsv" ]; then
        echo "=== ${name} (already exists, skip) ==="
        continue
    fi
    echo "=== ${name} ==="
    python3 "${CONSISTENCY}" \
        --target 63 \
        --bake "${bake}" \
        --bake-post clamp \
        --n-images 20 \
        --predict-tool "${TOOL}" \
        --zen-metrics "${ZEN_METRICS}" \
        --out "${out_tsv}" 2>&1 | tail -8 || echo "  driver failed for ${name}"
done

echo
echo "All cross-codec-T63 results in ${OUT_DIR}/"
