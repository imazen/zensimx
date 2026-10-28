#!/usr/bin/env bash
# V14-CVVDP-LOGNORM ablation — uses cvvdp_log_norm (precomputed 0..100) instead
# of cvvdp_score (JOD 0..10 × 10). Same recipe as V13 otherwise.
#
# V13 saturated because cvvdp_score is right-skewed (mean 9.59 JOD → 95.9 score,
# 73% safesyn samples ≥ 9.5 JOD). cvvdp_log_norm compresses the high-quality
# tail: mean 27.8, median 23.5, p10 unknown but distribution is much flatter.
# If this ablation recovers KonJND while preserving CID22 lift, the V13 root
# cause was target distribution, not the cvvdp distillation idea.
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
USE_GPU="${USE_GPU:-cuda}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v14_cvvdp_lognorm_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

BAKE="${OUT_DIR}/cc4v14_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v14_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v14_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "V14-CVVDP-LOGNORM ablation: seed=${SEED}"
echo "  target:   cvvdp_log_norm (already 0..100, scale 1.0)"

GPU_FLAG=""
if [ "${USE_GPU}" = "cuda" ]; then
    GPU_FLAG="--gpu-runtime cuda"
fi

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size "${KBATCH}" \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 300 --target-column cvvdp_log_norm --target-scale 1.0 \
    --per-sample-alpha-head --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.5 --mse-weight 0.5 --monotonicity-reg 0.5 \
    ${GPU_FLAG} \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
