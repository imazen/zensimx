#!/usr/bin/env bash
# EXP-CROSS-CODEC-V3: T=63 cross-codec consistency eval driver.
#
# After V3 bakes are calibrated, runs the cross_codec_consistency.py script
# on each candidate at T=63 with n=20 images. Reports mean butter_max,
# butter_p3 across the 3-codec pairings — the V3 ship gate requires either
# butter_max < 2.5 OR butter_p3 < 2.5.
set -euo pipefail

V3_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19"
CALIB_DIR="${V3_DIR}/calibrated"
OUT_DIR="${V3_DIR}/cross_codec_t63"
mkdir -p "${OUT_DIR}"

TOOL=/home/lilith/work/zen/zensimx/target/release/score_pair_with_bake
ZEN_METRICS=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
CONSISTENCY=/home/lilith/work/zen/zensimx/scripts/v_next/cross_codec_consistency.py

for bake in "${CALIB_DIR}"/*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_tsv="${OUT_DIR}/${name}_t63_n20.tsv"
    echo "=== ${name} ==="
    python3 "${CONSISTENCY}" \
        --target 63 \
        --bake "${bake}" \
        --bake-post clamp \
        --n-images 20 \
        --tool "${TOOL}" \
        --zen-metrics "${ZEN_METRICS}" \
        --out "${out_tsv}" 2>&1 | tail -5
done

echo
echo "All eval results in ${OUT_DIR}/"
