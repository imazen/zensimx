#!/usr/bin/env bash
# V11-A' v2 — clean GPU recipe (test if dropping MSE/mono/tanh helps).
# Keeps anchor + cross-codec-eq + dynamic-range from brief, but uses
# RankNet (default) + no MSE + no mono + no tanh-output-head-scale.
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_clean_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_300col_v2.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_v2.parquet"

BAKE="${OUT_DIR}/cc4v11a_v2clean_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_v2clean_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_v2clean_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "V11-A' v2 CLEAN recipe: seed=${SEED}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.6:0.0" \
    --group "large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:1.0:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "pipal:${PARQ_DIR}/pipal.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 1 \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 300 --target-column mix_cv35_iw65 \
    --per-sample-alpha-head \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 \
    --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 \
    --gpu-runtime cuda \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED}"
