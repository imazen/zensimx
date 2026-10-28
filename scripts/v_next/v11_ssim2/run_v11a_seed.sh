#!/usr/bin/env bash
# EXP-CROSS-CODEC-V11-A' trainer driver (2026-05-20) — ssim2-anchored substrate
# rebuild per task #189.
#
# V11-A' recipe (from task brief):
#   - Same as V11-A (V_22-mix-LARGE+iwssim, 300-feat) BUT with new substrate:
#     - anchor: 2026-05-20-ssim2-anchors/anchors_ssim2_300col.parquet
#     - equiv: 2026-05-20-ssim2-anchors/cross_codec_equivalence_ssim2.parquet
#   - Trainer is plain MLP (no per_sample_alpha_head, no hybrid) for Balanced
#     trail. Anchor + cross-codec-eq aux losses ride on the standard
#     RankNet+MSE training loop.
#
# Note: The trainer's anchor/cross-codec args historically required the
# per-sample-α head. For V11-A' we run the standard Balanced MLP recipe;
# any anchor-loss / cross-codec-eq pass-through that requires the
# per-sample-α head is structurally NOT available in this configuration.
# We rely on:
#   * anchor-loss-weight=1.0 + anchor-target-score=80.0 (JND) for the JND
#     calibration ONLY IF the trainer supports anchor with vanilla MLP
#   * pure RankNet+MSE on the canonical mix groups otherwise.
# This script tests both: the conditional flag set is in the brief.
#
# Args: <seed>
# Outputs in OUT_DIR: cc4v11a_ssim2_s<seed>.bin/.log/.stdout
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"  # SPEED-B default

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v11a_ssim2_2026-05-20"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

# Canonical-2026-05-21 training parquets (per task brief)
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

# V11 ssim2-anchored substrate (Phase 1+2 output)
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/anchors_ssim2_300col.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/cross_codec_equivalence_ssim2.parquet"

BAKE="${OUT_DIR}/cc4v11a_ssim2_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_ssim2_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_ssim2_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V11-A': seed=${SEED}"
echo "  trainer:  ${TRAINER}"
echo "  anchor:   ${ANCHOR}"
echo "  equiv:    ${EQUIV}"
echo "  out_bake: ${BAKE}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.6:0.0" \
    --group "large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:1.0:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "pipal:${PARQ_DIR}/pipal.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size "${KBATCH}" \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 300 --target-column mix_cv35_iw65 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 80.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
