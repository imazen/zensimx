#!/usr/bin/env bash
# V11-SUBSTRATE-V2 (task #190, 2026-05-20).
#
# V11-A' v2 trainer driver — REPLACES previous V11-A' attempt by
# CORRECTLY setting --per-sample-alpha-head, so the anchor + cross-codec-eq
# aux losses actually fire (V11-A' v1 used plain MLP, the trainer warned
# at startup that anchor data would be IGNORED).
#
# Substrate input: V11-SUBSTRATE-V2 from R2 omni multi-codec data
# (/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/), which includes:
# - 8527 anchor rows × 4 codecs × 10 bands × direct ssim2 anchoring
# - 1739 cross-codec equivalence pairs across 6 codec-pair combinations
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-1}"  # per-sample-α head requires --minibatch-size 1
USE_GPU="${USE_GPU:-cuda}"  # cuda | cpu

OUT_DIR="/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_2026-05-20"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_300col_v2.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_v2.parquet"

BAKE="${OUT_DIR}/cc4v11a_v2_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_v2_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_v2_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

if [ ! -f "${ANCHOR}" ]; then
    echo "ERROR: anchor parquet missing: ${ANCHOR}"; exit 1
fi
if [ ! -f "${EQUIV}" ]; then
    echo "ERROR: equiv parquet missing: ${EQUIV}"; exit 1
fi

echo "V11-A' v2: seed=${SEED}"
echo "  trainer:  ${TRAINER}"
echo "  anchor:   ${ANCHOR}"
echo "  equiv:    ${EQUIV}"
echo "  out_bake: ${BAKE}"
echo "  kbatch:   ${KBATCH}"
echo "  gpu:      ${USE_GPU}"

GPU_FLAG=""
if [ "${USE_GPU}" = "cuda" ]; then
    GPU_FLAG="--gpu-runtime cuda"
fi

# Per task brief — full recipe with per-sample-α head so anchor data
# actually fires. JND is auto-pulled from anchor parquet's target_score
# column (V5 multi-band) so --anchor-target-score is unused.
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.6:0.0" \
    --group "large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:1.0:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "pipal:${PARQ_DIR}/pipal.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size "${KBATCH}" \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 300 --target-column mix_cv35_iw65 \
    --per-sample-alpha-head --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.0 --mse-weight 1.0 --monotonicity-reg 1.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 \
    --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 \
    ${GPU_FLAG} \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
