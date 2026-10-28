#!/usr/bin/env bash
# Tuner v11 hyperparam sweep — task #6 iteration tool (2026-05-24).
#
# Given a list of (konjnd_aggregation_weight, step_p) values, runs
# 1-seed training for each, scores via bake_verdict + qsweep_eval,
# and emits a comparison matrix vs V_tuner_v10 baseline.
#
# Faster than the full 5-seed pipeline: 1 seed per config × ~15 min
# = ~15 min per cell. 6 cells = ~90 min to map the weight surface.
#
# Usage: bash scripts/v_next/tuner_v11_hparam_sweep.sh
# Env:
#   SWEEP_CONFIGS — space-separated list of "WEIGHT:STEP_P" pairs.
#     Default: "0.01:0.10 0.05:0.10 0.10:0.10 0.10:0.30 0.15:0.20 0.30:0.30"

set -euo pipefail

REPO="/home/lilith/work/zen/zensimx"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v11_sweep_2026-05-24"
SUMMARY="${OUT_DIR}/sweep_summary_$(date -u +%Y-%m-%d).md"

SWEEP_CONFIGS="${SWEEP_CONFIGS:-0.01:0.10 0.05:0.10 0.10:0.10 0.10:0.30 0.15:0.20 0.30:0.30}"

mkdir -p "${OUT_DIR}"

VERDICT="${REPO}/target/release/bake_verdict"
QSWEEP="${REPO}/target/release/qsweep_eval"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
V10_BAKE="${REPO}/zensim/weights/v_tuner_v10_2026-05-20.bin"

extract_srocc() {
    awk -v corpus="$2" '
        /^## Summary/ { in_summary = 1; next }
        in_summary && $0 ~ "\\| " corpus " \\|" {
            n = split($0, fields, "|")
            gsub(/ /, "", fields[4])
            print fields[4]
            exit
        }
    ' "$1"
}

# Train each cell + score.
for CFG in ${SWEEP_CONFIGS}; do
    W="${CFG%:*}"
    STEP="${CFG#*:}"
    TAG="w${W//.}step${STEP//.}"
    BAKE="${OUT_DIR}/tuner_v11_${TAG}.bin"
    LOG="${OUT_DIR}/tuner_v11_${TAG}.log"
    if [ -f "${BAKE}" ]; then
        echo "  ${TAG}: bake exists, skipping"
        continue
    fi
    echo
    echo "===== Training cell w=${W} step_p=${STEP} (tag=${TAG}) ====="

    "${REPO}/target/release/zensim_mlp_train" \
        --group "safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet:1.0:0.0" \
        --group "cid22_train:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/cid22_train.parquet:0.5:0.0" \
        --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5 \
        --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
        --max-features 372 --minibatch-size 32 \
        --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
        --per-sample-alpha-head \
        --tanh-output-head-scale 20.0 \
        --ranknet-weight 0.0 --mse-weight 1.0 \
        --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
        --anchor-parquet /mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet \
        --anchor-loss-weight 0.5 --anchor-target-score 60.0 --anchor-step-p 0.30 \
        --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
        --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 \
        --cross-codec-rank-preserve-weight 0.2 \
        --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 \
        --dynamic-range-step-p 0.05 --dynamic-range-probe-n 40 \
        --konjnd-aggregation-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/konjnd-dense.parquet \
        --konjnd-aggregation-weight "${W}" \
        --konjnd-aggregation-step-p "${STEP}" \
        --konjnd-aggregation-samples-per-ref 5 \
        --konjnd-aggregation-refs-per-step 8 \
        --seed 1 --out "${BAKE}" --log-path "${LOG}" 2>&1 | tail -5
done

# Eval all bakes (+ v10 baseline).
echo
echo "===== Eval matrix ====="
"${VERDICT}" --bake "${V10_BAKE}" --output "${OUT_DIR}/v10_verdict.md" > /dev/null 2>&1
QSWEEP_ARGS=( --features "${QSWEEP_FEATURES}" --manifest "${QSWEEP_MANIFEST}" \
              --bake "tuner_v10=${V10_BAKE}" )
for CFG in ${SWEEP_CONFIGS}; do
    W="${CFG%:*}"
    STEP="${CFG#*:}"
    TAG="w${W//.}step${STEP//.}"
    BAKE="${OUT_DIR}/tuner_v11_${TAG}.bin"
    [ -f "${BAKE}" ] || continue
    "${VERDICT}" --bake "${BAKE}" --output "${OUT_DIR}/${TAG}_verdict.md" > /dev/null 2>&1
    QSWEEP_ARGS+=( --bake "${TAG}=${BAKE}" )
done
QSWEEP_ARGS+=( --out "${OUT_DIR}/qsweep_matrix.md" )
"${QSWEEP}" "${QSWEEP_ARGS[@]}" 2>&1 | grep -E "monotonicity="

# Build summary.
{
    echo "# Tuner v11 hparam sweep summary"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Out: ${OUT_DIR}"
    echo
    echo "## SROCC matrix"
    echo
    echo "| config | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |"
    echo "|---|---:|---:|---:|---:|---:|---:|"
    row="| v10 baseline |"
    for c in "CID22" "KADIK10k" "TID2013" "KonJND-1k (full)" "AIC-3 CTC" "AIC-4 sample"; do
        s=$(extract_srocc "${OUT_DIR}/v10_verdict.md" "${c}")
        row="${row} ${s:-?} |"
    done
    echo "${row}"
    for CFG in ${SWEEP_CONFIGS}; do
        W="${CFG%:*}"
        STEP="${CFG#*:}"
        TAG="w${W//.}step${STEP//.}"
        V="${OUT_DIR}/${TAG}_verdict.md"
        [ -f "${V}" ] || continue
        row="| w=${W} step=${STEP} |"
        for c in "CID22" "KADIK10k" "TID2013" "KonJND-1k (full)" "AIC-3 CTC" "AIC-4 sample"; do
            s=$(extract_srocc "${V}" "${c}")
            row="${row} ${s:-?} |"
        done
        echo "${row}"
    done
    echo
    echo "Monotonicity: see [\`qsweep_matrix.md\`](qsweep_matrix.md)."
} > "${SUMMARY}"
echo "Summary: ${SUMMARY}"
