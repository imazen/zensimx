#!/usr/bin/env bash
# EXP-PERSAMPLE-MIX3: per-sample-α head trainer with the EX-MIX3 cv30_iw40_sm30 target.
#
# Hypothesis: combine the two strongest compression-trail directions
# (per-sample-α architecture from EX-2 + 3-way mix target from EX-MIX3)
# and beat per-sample-α s4 alone on the compression trail.
#
# Args: <seed>
#
# Recipe:
#   - 4 groups: safesyn (1.0:0.0), kadid (0.3:1.0), tid (0.3:1.0), konjnd (0.02:1.0)
#   - LARGE group DROPPED (0% ssim2 coverage, can't supply mix_cv30_iw40_sm30)
#   - --target-column mix_cv30_iw40_sm30 (= 0.3·cvvdp + 0.4·iwssim + 0.3·ssim2,
#     all in log-norm score units 0..100)
#   - konjnd's mix_cv30_iw40_sm30 column is PJND-passthrough (per add_mix3_target.py
#     coverage gate it should be DROPPED, but the file in 2026-05-18-mix3/
#     copies human_score into all mix columns for konjnd so the trainer can use it
#     as PJND-anchor — this is the EX-MIX3 round-2 learning).
#   - --per-sample-alpha-head (V_24-per-sample-α architecture)
#   - h=128, epochs=300, mb=256, max-features=372, target-scale=100.0
#   - PWRC pair weighting + NiN composition (mandatory for V_22-LARGE recipe)

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/2026-05-18-mix3"

BAKE="${OUT_DIR}/persample_mix3_s${SEED}_h128.bin"
LOG="${OUT_DIR}/persample_mix3_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/persample_mix3_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd.parquet:0.02:1.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 256 \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
    --target-column mix_cv30_iw40_sm30 --target-scale 100.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
