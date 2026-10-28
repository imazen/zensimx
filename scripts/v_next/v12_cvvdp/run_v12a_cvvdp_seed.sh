#!/usr/bin/env bash
# V12-A cvvdp-anchored substrate retrain (task #199, 2026-05-20).
#
# Mirrors run_v11a_372_v4_seed.sh (V11-A'-372 v4 brief recipe — per-sample-α +
# MSE + monotonicity + tanh-output-head-scale 20.0) but pivots the cross-codec
# anchor + equivalence substrate from ssim2 → cvvdp. Per task brief
# --cross-codec-eq-weight is 0.5 (middle of the V11 sweep at which KonJND
# collapsed identically at every weight in {0.05, 0.10, 0.20, 0.50, 1.00}).
#
# The hypothesis under test: does the basin-B KonJND collapse trap collapse
# also when the pivot metric is cvvdp instead of ssim2? Per Mohammadi 2025,
# cvvdp Z-RMSE 9.45 vs ssim2 47.63 — 5x better absolute calibration. If
# KonJND survives at the cvvdp pivot, ship as V0_5BalancedV4. If KonJND
# still collapses, the basin-B mechanism is anchor-metric-independent and
# the V11/V12 cross-codec-eq frontier is closed structurally.
#
# Args: <seed>
# Env:
#   USE_GPU=cuda|cpu      default cuda
#   KBATCH=N              default 32 (per task brief)
#   OUT_DIR=path          override output directory
#   CC_EQ_WEIGHT=N        default 0.5 (per task brief)
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
USE_GPU="${USE_GPU:-cuda}"
CC_EQ_WEIGHT="${CC_EQ_WEIGHT:-0.5}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v12_cvvdp_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/anchors_cvvdp_372col.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/cross_codec_equivalence_cvvdp_372col.parquet"

BAKE="${OUT_DIR}/cc4v12a_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v12a_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v12a_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

if [ ! -f "${ANCHOR}" ]; then
    echo "ERROR: anchor parquet missing: ${ANCHOR}"; exit 1
fi
if [ ! -f "${EQUIV}" ]; then
    echo "ERROR: equiv parquet missing: ${EQUIV}"; exit 1
fi

echo "V12-A cvvdp-anchored (task #199): seed=${SEED} cc_eq=${CC_EQ_WEIGHT}"
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

# CID22 is VALIDATION-ONLY per CLAUDE.md — cid22_train is the training-only
# subset that excludes the 49 held-out references.
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
    --cross-codec-eq-weight "${CC_EQ_WEIGHT}" \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 \
    --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 \
    ${GPU_FLAG} \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
