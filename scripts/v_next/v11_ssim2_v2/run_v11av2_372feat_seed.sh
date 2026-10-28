#!/usr/bin/env bash
# V11-A' v2 — 372-feat variant (addresses user question on whether
# 372-feat was systematically explored).
#
# Drops the V11 substrate (anchor + cross-codec-eq are 300-feat
# only — can't be padded to 372 without re-extracting from R2
# encoded variants). Tests V_24-per-sample-α recipe + new training
# groups (cid22_train + pipal) at 372 features.
#
# If this beats V10 BalancedV3, the IW-pool block (last 72 features)
# is the lever; the V11 substrate is a separate question.
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_372feat_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

BAKE="${OUT_DIR}/cc4v11a_v2_372_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v11a_v2_372_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v11a_v2_372_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "V11-A' v2 372feat: seed=${SEED}"

# Skip large (it's 300-feat only) — use all-372 groups.
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd-dense.parquet:0.10:1.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --group "pipal:${PARQ_DIR}/pipal.parquet:0.3:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 256 \
    --val-policy min --early-stop-patience 0 --log-every 10 \
    --max-features 372 --target-column mix_cv35_iw65 \
    --per-sample-alpha-head \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE}"
