#!/usr/bin/env bash
# PreviewV0_5TunerV2 (EXP-TUNER-V2, 2026-05-19): today's Tuner recipe +
# cross-codec JND anchor loss. Today's Tuner (PreviewV0_5Tuner) has
# 6.68 butter mean pairwise at T=63 across zenjpeg/zenwebp/zenavif —
# meaningfully different perceptual quality at the same "score 63".
# This recipe forces score=63 to land at PJND across codecs by adding
# an MSE anchor loss against 63 on a 9373-row anchor pool that mixes:
#   - 504 KonJND-1k JPEG PJND anchors (real human data, anchor_weight=1.5)
#   - 504 KonJND-1k BPG PJND anchors (real human data, anchor_weight=1.5)
#   - 8365 safesyn synth PJND anchors (ssim2 ≈ 63, anchor_weight=1.0)
#
# Recipe vs today's Tuner (run_tuner_seed_v2.sh, commit 13f0fd0b):
#   - SAME --hidden 128 --epochs 300 --mse-weight 1.0 --ranknet-weight 0
#     --monotonicity-reg 0.5 --target-column mix_cv40_iw60 --minibatch-size 1
#   - ADD --anchor-parquet ... --anchor-loss-weight 1.0
#     --anchor-target-score 63 --anchor-step-p 0.10
#
# Args: <seed>
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet"

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
    --monotonicity-reg 0.5 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.10 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
