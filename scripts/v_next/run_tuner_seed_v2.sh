#!/usr/bin/env bash
# PreviewV0_5Tuner v2 (EXP-TUNER, 2026-05-19): pure-MSE training with
# monotonicity reg. Previous v1 mixed RankNet 0.3 + MSE 1.0 — RankNet's
# distance-shape convention CANCELLED the MSE score-shape gradient, so
# the bake collapsed to predicting the global mean (~32) for everything.
#
# Discovery: the production trainer's RankNet loss treats bake output as
# DISTANCE-shaped (low = good MOS). With ranknet_weight=0 we get pure MSE
# on score-shaped target (high = good MOS). Bake outputs will then be
# directly score-shaped, no inversion needed at runtime.
#
# Recipe vs V_24-per-sample-α:
#   - NO NiN composition (--norm-in-norm-weight 0).
#   - --mse-weight 1.0, --ranknet-weight 0.0  (pure MSE).
#   - --monotonicity-reg 1.0  (quadratic hinge on per-pair order).
#   - safesyn-only training on mix_cv40_iw60 (canonical val parquets
#     have mix_cv40_iw60 null, so val-only groups would broken).
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"

BAKE="${OUT_DIR}/tuner_v2_s${SEED}_h128.bin"
LOG="${OUT_DIR}/tuner_v2_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/tuner_v2_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
