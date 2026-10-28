#!/usr/bin/env bash
# V12-B continuous-mapping cvvdp anchor variant (task #199, 2026-05-20).
#
# Same recipe as run_v12a_cvvdp_seed.sh but uses the continuous-mapping
# anchor parquet that emits one anchor row per (image, codec, q) with
# target_score derived continuously from cvvdp. Tests whether the
# V12-A's CID22 collapse comes from the band-snap coverage gap (only
# 1 anchor at target_score=20, 0 anchors below) vs the cvvdp metric
# choice itself.
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
USE_GPU="${USE_GPU:-cuda}"
CC_EQ_WEIGHT="${CC_EQ_WEIGHT:-0.5}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v12b_cvvdp_continuous_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/anchors_cvvdp_372col_continuous.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/cross_codec_equivalence_cvvdp_372col.parquet"

BAKE="${OUT_DIR}/cc4v12b_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v12b_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v12b_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

[ -f "${ANCHOR}" ] || { echo "ERROR: anchor parquet missing: ${ANCHOR}"; exit 1; }
[ -f "${EQUIV}" ] || { echo "ERROR: equiv parquet missing: ${EQUIV}"; exit 1; }

echo "V12-B cvvdp-continuous (task #199): seed=${SEED} cc_eq=${CC_EQ_WEIGHT}"

GPU_FLAG=""
[ "${USE_GPU}" = "cuda" ] && GPU_FLAG="--gpu-runtime cuda"

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
