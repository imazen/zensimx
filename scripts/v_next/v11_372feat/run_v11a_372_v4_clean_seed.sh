#!/usr/bin/env bash
# V11-DECODER-FIX V11-A'-372 v4 CLEAN-recipe retrain (task #195, 2026-05-20).
#
# Mirror of run_v11av2_gpu_clean_seed.sh (300-feat clean recipe that
# achieved CID22 0.8754) but at --max-features 372 to include the
# IW-pool block, and uses the new 4-codec × 372-feat v4 substrate.
#
# Differences vs run_v11a_372_v4_seed.sh (brief recipe):
# - DROPS --mse-weight 1.0 (defaults to 0.0, RankNet drives ranking)
# - DROPS --ranknet-weight 0.0 (defaults to 1.0)
# - DROPS --monotonicity-reg 1.0 (defaults to 0.0)
# - DROPS --tanh-output-head-scale 20.0 (defaults to 0.0)
# - --minibatch-size 1 (per-sample-α head default; GPU auto-bumps to 512)
#
# Skips the `large` group present in v2 clean (cvvdp_iwssim_LARGE is
# 300-feat only). The 6-group training is the brief's group set.
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
USE_GPU="${USE_GPU:-cuda}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11a_372_v4_clean_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_372col_v4.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet"

BAKE="${OUT_DIR}/cc4v11a_372_v4_clean_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_372_v4_clean_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_372_v4_clean_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

[ -f "${ANCHOR}" ] || { echo "ERROR: anchor parquet missing: ${ANCHOR}"; exit 1; }
[ -f "${EQUIV}" ] || { echo "ERROR: equiv parquet missing: ${EQUIV}"; exit 1; }

echo "V11-A'-372 v4 CLEAN recipe (RankNet default + per-sample-α): seed=${SEED}"

GPU_FLAG=""
[ "${USE_GPU}" = "cuda" ] && GPU_FLAG="--gpu-runtime cuda"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.6:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "pipal:${PARQ_DIR}/pipal.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 1 \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 372 --target-column mix_cv35_iw65 \
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
    ${GPU_FLAG} \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED}"
