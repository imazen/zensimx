#!/usr/bin/env bash
# Tuner v11 attempt 3 — konjnd-dense as BOTH training group AND aggregation pool (2026-05-24).
#
# Attempts 1 (w=0.3) + 2 (w=0.05) both fall into α-collapse: either
# α=0 (pool only, rank corpora collapse) or α=1 (rank only, KonJND
# stays at v10 baseline). The structural fix is to use konjnd-dense
# in BOTH roles:
#   - Regular training group with --target-column mix_cv40_iw60
#     (per-pair feature-driven MSE; no zero-gradient pathology
#     because per-row mix_cv40_iw60 varies)
#   - Aggregation pool with --konjnd-aggregation-* (per-ref mean
#     MSE against pjnd_target; soft PJND calibration bias)
#
# Why this works in theory:
#   - The per-pair MSE on konjnd-dense gives feature-driven gradient
#     on the konjnd-dense distribution. Network learns per-pair
#     scores. No per-source-constant zero-gradient.
#   - The aggregation MSE adds the per-ref PJND threshold as a soft
#     constraint on top.
#   - The α gate has room to find a stable middle blend.
#
# Args: <seed>
# Env: KBATCH=32, LR=5.66e-3, KONJND_TRAIN_WEIGHT=0.3, KONJND_AGG_WEIGHT=0.1, KONJND_AGG_STEP_P=0.10

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
LR="${LR_OVERRIDE:-5.66e-3}"
KONJND_TRAIN_WEIGHT="${KONJND_TRAIN_WEIGHT:-0.3}"
KONJND_AGG_WEIGHT="${KONJND_AGG_WEIGHT:-0.1}"
KONJND_AGG_STEP_P="${KONJND_AGG_STEP_P:-0.10}"

REPO="/home/lilith/work/zen/zensimx"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v11_attempt3_2026-05-24"
TRAINER="${REPO}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/tuner_v11_a3_s${SEED}.bin"
LOG="${OUT_DIR}/tuner_v11_a3_s${SEED}.log"
STDOUT="${OUT_DIR}/tuner_v11_a3_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

[ -x "${TRAINER}" ] || { echo "trainer missing" >&2; exit 2; }

echo "Tuner v11 attempt 3: seed=${SEED}"
echo "  konjnd-dense train_weight=${KONJND_TRAIN_WEIGHT} (mix_cv40_iw60 target)"
echo "  konjnd-aggregation weight=${KONJND_AGG_WEIGHT} step_p=${KONJND_AGG_STEP_P} (pjnd_target)"
echo "  cid22-train weight=0.5 (mix_cv40_iw60 target)"
echo "  safesyn weight=1.0"
echo

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "konjnd_dense:${PARQ_DIR}/konjnd-dense.parquet:${KONJND_TRAIN_WEIGHT}:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr "${LR}" --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size "${KBATCH}" \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.0 --mse-weight 1.0 \
    --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 0.5 --anchor-target-score 60.0 --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 --dynamic-range-probe-n 40 \
    --konjnd-aggregation-parquet "${PARQ_DIR}/konjnd-dense.parquet" \
    --konjnd-aggregation-weight "${KONJND_AGG_WEIGHT}" \
    --konjnd-aggregation-step-p "${KONJND_AGG_STEP_P}" \
    --konjnd-aggregation-samples-per-ref 5 \
    --konjnd-aggregation-refs-per-step 8 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
