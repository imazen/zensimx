#!/usr/bin/env bash
# PreviewV0_5TunerV2 LIGHT (EXP-TUNER-V2, 2026-05-19, attempt 2):
# Same recipe but --anchor-loss-weight reduced 20× to 0.05 to test
# whether a lighter anchor preserves rank fidelity while still pulling
# cross-codec consistency at T=63.
#
# Hypothesis (revised): anchor weight 1.0 was too aggressive
# (falsification doc benchmarks/v_tuner_v2_falsification_2026-05-19.md).
# Try 0.05 — 1/20th the gradient force.
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v2_light_2026-05-19"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet"

BAKE="${OUT_DIR}/tuner_v2_light_s${SEED}_h128.bin"
LOG="${OUT_DIR}/tuner_v2_light_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/tuner_v2_light_s${SEED}.stdout"

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
    --monotonicity-reg 0.5 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 0.05 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.10 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
