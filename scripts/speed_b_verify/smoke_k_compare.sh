#!/usr/bin/env bash
# SPEED-B smoke test: 10-epoch run at K=1 vs K=32 to confirm no panic + early wall time.
set -euo pipefail

KBATCH="${1:?usage: $0 <minibatch_size>}"

OUT_DIR="/mnt/v/zen/zensim-eval/speed_b_verify_2026-05-19/smoke"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/smoke_k${KBATCH}.bin"
LOG="${OUT_DIR}/smoke_k${KBATCH}.log"
STDOUT="${OUT_DIR}/smoke_k${KBATCH}.stdout"

mkdir -p "${OUT_DIR}"

START_TS=$(date +%s)

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 10 --pairs-per-epoch 5000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size "${KBATCH}" \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed 1 --out "${BAKE}" --log-path "${LOG}" \
    >"${STDOUT}" 2>&1

END_TS=$(date +%s)
WALL=$((END_TS - START_TS))
echo "SMOKE kbatch=${KBATCH} wall=${WALL}s bake=${BAKE}"
