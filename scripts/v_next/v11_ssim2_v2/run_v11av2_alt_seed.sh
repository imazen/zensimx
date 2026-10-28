#!/usr/bin/env bash
# V11-SUBSTRATE-V2 ALT recipe (task #190, 2026-05-20).
#
# Alternative recipe that's closer to the proven V_24-per-sample-α
# (PreviewV0_5Compression) training setup, but adds the V11-substrate-v2
# anchor + cross-codec-eq inputs at lower weight. The brief's recipe
# with --mse-weight 1.0, --ranknet-weight 0.0, --monotonicity-reg 1.0,
# --minibatch-size 1 produced CID22 ≈ 0.71 (down from V10 0.83) on
# 2 seeds; this is a recipe issue not a substrate issue.
#
# Recipe deltas from brief:
# - minibatch-size 256 (was 1)
# - ranknet-weight default (RankNet on, was off via 0.0)
# - mse-weight 0.0 (default, was 1.0)
# - monotonicity-reg 0.0 (default, was 1.0)
# - drop --tanh-output-head-scale (default 1.0)
# - keep anchor + cross-codec-eq at full weight (substrate is good)
# - drop dynamic-range floor (low-leverage, simplify)
# - keep target mix_cv35_iw65 per brief
# - add --pwrc-pair-weight (proven on V_24)
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_alt_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/anchors_ssim2_300col_v2.parquet"
EQUIV="/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/cross_codec_equivalence_ssim2_v2.parquet"

BAKE="${OUT_DIR}/cc4v11a_v2alt_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_v2alt_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_v2alt_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "V11-A' v2 ALT recipe: seed=${SEED}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.10:1.0" \
    --group "large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:0.5:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 256 \
    --val-policy min --early-stop-patience 0 --log-every 10 \
    --max-features 300 --target-column mix_cv35_iw65 \
    --per-sample-alpha-head \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --gpu-runtime cuda \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE}"
