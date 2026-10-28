#!/usr/bin/env bash
# Cross-codec metric eval driver (EXP-CROSS-CODEC-METRIC, 2026-05-19).
#
# Args: <bake_path> <label>
# Output: writes per-T TSV + summary to OUT_DIR/<label>/
#
# Targets: T ∈ {30, 50, 63, 70, 80, 90}. Each runs binary-search-q per
# codec (jpeg/webp/avif via PIL) for each of 20 sources, then computes
# pairwise butter on the 3 decodes.

set -euo pipefail

BAKE="${1:?usage: $0 <bake_path> <label>}"
LABEL="${2:?usage: $0 <bake_path> <label>}"

ROOT="/home/lilith/work/zen/zensimx"
SCRIPT="${ROOT}/scripts/v_next/cross_codec_consistency.py"
TOOL="/home/lilith/work/zen/zensimx/target/release/score_pair_with_bake"
ZEN_METRICS="/home/lilith/work/zen/zenmetrics/target/release/zenmetrics"

OUT_DIR="/mnt/v/output/zensim/cross_codec_metric_2026-05-19/${LABEL}"
mkdir -p "${OUT_DIR}"

if [ ! -x "${TOOL}" ]; then
    echo "ERROR: missing ${TOOL}; build with cargo build --release -p zensim-validate --bin score_pair_with_bake" >&2
    exit 2
fi
if [ ! -x "${ZEN_METRICS}" ]; then
    echo "ERROR: missing ${ZEN_METRICS}" >&2
    exit 2
fi
if [ ! -f "${BAKE}" ]; then
    echo "ERROR: missing bake ${BAKE}" >&2
    exit 2
fi

echo "EVAL ${LABEL}: bake=${BAKE}"
echo "  out_dir=${OUT_DIR}"

for T in 30 50 63 70 80 90; do
    OUT_TSV="${OUT_DIR}/t${T}.tsv"
    LOG="${OUT_DIR}/t${T}.log"
    echo "  T=${T} ..."
    python3 "${SCRIPT}" \
        --bake "${BAKE}" \
        --bake-post clamp \
        --tool "${TOOL}" \
        --zen-metrics "${ZEN_METRICS}" \
        --target "${T}" \
        --n-images 20 \
        --out "${OUT_TSV}" 2> "${LOG}" \
        | tee -a "${OUT_DIR}/all.stdout"
    BMAX=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="pairwise_butter_max_mean"){idx=i}}; next} {sum+=$idx; n+=1} END{if(n>0) print sum/n; else print "NaN"}' "${OUT_TSV}")
    BP3=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="pairwise_butter_p3_mean"){idx=i}}; next} {sum+=$idx; n+=1} END{if(n>0) print sum/n; else print "NaN"}' "${OUT_TSV}")
    echo "    T=${T}: mean_butter_max=${BMAX} mean_butter_pnorm3=${BP3}"
done

echo
echo "DONE label=${LABEL} bake=${BAKE}"
