#!/usr/bin/env bash
# V11-A-CC-EQ-WEIGHT-SWEEP (task #197, 2026-05-20).
#
# Sibling of run_v11a_372_v4_clean_seed.sh — same recipe, but
# parameterizes --cross-codec-eq-weight via $CC_EQ_WEIGHT env var.
#
# Hypothesis: at w << 1.0 the rank-preserve term dominates so
# KonJND survives while still extracting partial cross-codec
# benefit. V11-A'-372 v4 clean s2 (w=1.0) hit CID22 0.8944 but
# KonJND collapsed 0.8927 → 0.3942. Is the collapse a smooth
# function of w?
#
# Args: <seed>
# Env: CC_EQ_WEIGHT (required, e.g. 0.05, 0.10, 0.20, 0.50)
#      USE_GPU (default cuda)
set -euo pipefail

SEED="${1:?usage: $0 <seed>; env CC_EQ_WEIGHT=<float>}"
CC_EQ_WEIGHT="${CC_EQ_WEIGHT:?missing env CC_EQ_WEIGHT}"
USE_GPU="${USE_GPU:-cuda}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_372col_v4.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_372col_v4.parquet"

# Embed weight in filename. Strip the leading "0." for compactness.
W_TAG="$(echo "$CC_EQ_WEIGHT" | sed 's/^0*\.//; s/\.//g')"
BAKE="${OUT_DIR}/cc4v11_w${W_TAG}_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11_w${W_TAG}_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11_w${W_TAG}_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

[ -f "${ANCHOR}" ] || { echo "ERROR: anchor parquet missing: ${ANCHOR}"; exit 1; }
[ -f "${EQUIV}" ] || { echo "ERROR: equiv parquet missing: ${EQUIV}"; exit 1; }

echo "V11-A-CC-EQ-WEIGHT-SWEEP w=${CC_EQ_WEIGHT} seed=${SEED}"

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
    --cross-codec-eq-weight "${CC_EQ_WEIGHT}" \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 \
    --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 \
    ${GPU_FLAG} \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE w=${CC_EQ_WEIGHT} seed=${SEED} → ${BAKE}"
