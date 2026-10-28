#!/usr/bin/env bash
# EXP-DISTILL-ENSEMBLE: per-sample-α head trainer distilled from
# PreviewV0_5Ensemble (Balanced + Compression + classifier routing).
#
# Args: <seed>
#
# Recipe mirrors V_24-per-sample-α s4 (Compression ship) exactly,
# except `--target-column ensemble_teacher` (teacher = Ensemble's
# routed score, in [0,1] for `--target-scale 100.0`).
#
# Single-bake student aims to match Ensemble's panel performance
# without classifier-routing runtime overhead.

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_distill_ensemble_2026-05-18"
WORKSPACE="/home/lilith/work/zen/zensimx"
TRAINER="${WORKSPACE}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/2026-05-18-distill-ensemble"

BAKE="${OUT_DIR}/distill_s${SEED}_h128.bin"
LOG="${OUT_DIR}/distill_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/distill_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.02:1.0" \
    --group "cvvdp_iwssim_large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:0.5:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
    --target-column ensemble_teacher --target-scale 100.0 \
    --val-policy min --minibatch-size 256 \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
    --per-sample-alpha-head --seed "${SEED}" --log-every 10 --early-stop-patience 0 \
    --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
