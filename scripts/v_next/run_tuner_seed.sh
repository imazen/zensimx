#!/usr/bin/env bash
# PreviewV0_5Tuner (EXP-TUNER, 2026-05-18): per-sample-α head trained
# with MSE + monotonicity reg on mix_cv40_iw60 — a codec-tuner-shaped
# zensim variant for user-typed quality targets.
#
# Hypothesis: trading some rank-honesty (lower SROCC) for direct
# calibration honesty (low RMSE per-band on the target axis) plus
# monotonicity-reg (no q-step inversions) yields a metric that codecs
# can binary-search against, with stable cross-codec consistency.
#
# Recipe vs V_24-per-sample-α:
#   - NO NiN composition (--norm-in-norm-weight 0).
#   - Add --mse-weight 1.0 (regression on mix_cv40_iw60).
#   - Add --monotonicity-reg 0.5 (quadratic hinge on per-pair order).
#   - Lower RankNet weight (--ranknet-weight 0.3) — keep some
#     rank-pressure for hard-to-fit features without dominating.
#   - safesyn-only training (mix_cv40_iw60 column is well-defined
#     and score-shaped 0..100 there); KADID + TID + KonJND moved
#     to val-only so SROCC still tracked.
#   - Drop KonJND-dense entirely (its mix_cv40_iw60 ranges to -65,
#     wrong-shape for MSE).
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_2026-05-18"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
VAL_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/val"

BAKE="${OUT_DIR}/tuner_s${SEED}_h128.bin"
LOG="${OUT_DIR}/tuner_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/tuner_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

# Notes on group layout (2026-05-19):
#  - Only safesyn is loaded; the canonical val parquets carry mix_cv40_iw60
#    as null, so val-SROCC would always be 0 and gate-on-val would pick the
#    epoch-0 bake. With safesyn as the sole group, val_indices is empty and
#    the trainer falls back to the safesyn-SROCC mean for best-checkpoint
#    selection — perfectly safe for the tuner experiment (final eval is
#    held-out, via bake_verdict on canonical val parquets after training).
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.3 \
    --mse-weight 1.0 \
    --monotonicity-reg 0.5 \
    --monotonicity-margin 0.0 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
