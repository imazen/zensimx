#!/usr/bin/env bash
# PreviewV0_5Tuner v3 (EXP-TUNER, 2026-05-19): aggressive MSE-only with
# much higher weight + no L2 reg + no monotonicity hinge. v2 collapsed
# the bake output to a narrow 30..32 range despite SROCC=0.99 — MSE was
# too gentle to push the per-image variation into use of the full
# target range. v3 attempts to fix by increasing MSE-weight 100×.
#
# Recipe:
#   - --mse-weight 100 --ranknet-weight 0 --monotonicity-reg 0 (pure
#     MSE; monotonicity should emerge from a well-fit MSE, no need
#     for explicit hinge regularizer).
#   - --l2 0  (remove weight regularization; bake can use full range).
#   - rest same as v2.
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"

BAKE="${OUT_DIR}/tuner_v3_s${SEED}_h128.bin"
LOG="${OUT_DIR}/tuner_v3_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/tuner_v3_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 0.0 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.0 \
    --mse-weight 100.0 \
    --monotonicity-reg 0.0 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
