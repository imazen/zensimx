#!/usr/bin/env bash
# V11-DECODER-FIX V11-A'-372 v4 retrain (task #195, 2026-05-20).
#
# Builds the V11-A' Balanced 372-feature retrain on the V11-SUBSTRATE-V2
# anchor + cross-codec-equivalence parquets that were rebuilt at
# 4-codec coverage after the V11-DECODER-FIX. Matches the V11-A' v2
# 300-feat recipe (run_v11av2_seed.sh) but with --max-features 372 and
# v4-suffixed substrate filenames (full 117,800-cell coverage rather
# than the v3 partial 62,600-cell zenjpeg+zenwebp-only set).
#
# Args: <seed>
# Env:
#   USE_GPU=cuda|cpu      default cuda
#   KBATCH=N              default 1 (per-sample-α head + aux losses)
#   OUT_DIR=path          override output directory
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
# Brief recipe specifies --minibatch-size 32. On GPU the trainer
# auto-bumps to 512 (zensim-validate/src/bin/zensim_mlp_train.rs:2068).
KBATCH="${KBATCH:-32}"
USE_GPU="${USE_GPU:-cuda}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11a_372_v4_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_372col_v4.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet"

BAKE="${OUT_DIR}/cc4v11a_372_v4_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_372_v4_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_372_v4_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

if [ ! -f "${ANCHOR}" ]; then
    echo "ERROR: anchor parquet missing: ${ANCHOR}"; exit 1
fi
if [ ! -f "${EQUIV}" ]; then
    echo "ERROR: equiv parquet missing: ${EQUIV}"; exit 1
fi

echo "V11-A'-372 v4 (V11-DECODER-FIX retrain): seed=${SEED}"
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

# Same recipe as run_v11av2_seed.sh (V_24-per-sample-α adapted to
# anchor + cross-codec-eq) but --max-features 372 instead of 300 so
# the IW-pool block (last 72 features) is included.
#
# CID22 is VALIDATION-ONLY per CLAUDE.md — `cid22_train` is the
# training-only subset that excludes the 49 held-out references.
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.6:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "pipal:${PARQ_DIR}/pipal.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size "${KBATCH}" \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 372 --target-column mix_cv35_iw65 \
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
