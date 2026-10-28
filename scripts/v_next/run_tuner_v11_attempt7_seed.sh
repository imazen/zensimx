#!/usr/bin/env bash
# Tuner v11 attempt 7 — recovery phase 4 first cut (2026-05-24).
#
# Per user directive 2026-05-24: "you can train on multiple data sets
# incl synth". Adds kadid + tid as training groups (they had been val
# only). Combined with a4's proven structural fix (konjnd_dense as
# training group + aggregation pool at w=0.3) AND a wider tanh pin
# (scale 30 vs a4's 20) to give the 0-55 dial more room.
#
# Recipe components:
#  - safesyn:1.0 (synth, 196k pairs)
#  - cid22_train:0.5 (real-codec ssim2-anchored, 17.6k)
#  - kadid:0.5 (synth distortions, 10.1k) — was val only
#  - tid:0.5 (synth distortions, 3.0k) — was val only
#  - konjnd_dense:0.3 (real PJND, 20.2k)
#  - konjnd-aggregation pool: w=0.3 step_p=0.30 (a4's sweet spot)
#  - tanh_output_head_scale=30 (was 20 in a4; wider dial range)
#  - All other hyperparams identical to a4
#
# Held-out vals (CLAUDE.md "CID22 is VALIDATION-ONLY" still holds):
#  - CID22 (sacred 49-ref holdout)
#  - KonJND-1k val (1008 rows, PJND threshold per ref)
#  - AIC-3 CTC (600 rows, JND compression)
#  - AIC-4 sample (300 rows, JND compression)
#
# KADID + TID become training-set-like and no longer rigorous val
# anchors. They're "integrity guards" in v11 methodology — losing
# them as integrity is the cost of using their MOS for training.
#
# Args: <seed>
# Env: KBATCH=32, LR=5.66e-3, KONJND_AGG_WEIGHT=0.3, KONJND_AGG_STEP_P=0.30, TANH_SCALE=30

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
LR="${LR_OVERRIDE:-5.66e-3}"
KONJND_AGG_WEIGHT="${KONJND_AGG_WEIGHT:-0.3}"
KONJND_AGG_STEP_P="${KONJND_AGG_STEP_P:-0.30}"
TANH_SCALE="${TANH_SCALE:-30.0}"

REPO="/home/lilith/work/zen/zensimx"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v11_attempt7_2026-05-24"
TRAINER="${REPO}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/tuner_v11_a7_s${SEED}.bin"
LOG="${OUT_DIR}/tuner_v11_a7_s${SEED}.log"
STDOUT="${OUT_DIR}/tuner_v11_a7_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

[ -x "${TRAINER}" ] || { echo "trainer missing" >&2; exit 2; }

echo "Tuner v11 attempt 7: seed=${SEED}"
echo "  Training groups: safesyn:1.0 + cid22_train:0.5 + kadid:0.5 + tid:0.5 + konjnd_dense:0.3"
echo "  konjnd-aggregation: weight=${KONJND_AGG_WEIGHT} step_p=${KONJND_AGG_STEP_P}"
echo "  tanh_output_head_scale: ${TANH_SCALE}"
echo

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.5:0.0" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.5:0.0" \
    --group "konjnd_dense:${PARQ_DIR}/konjnd-dense.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr "${LR}" --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size "${KBATCH}" \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale "${TANH_SCALE}" \
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
