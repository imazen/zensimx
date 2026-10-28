#!/usr/bin/env bash
# V13-CVVDP-DISTILL — cvvdp-as-teacher distillation (task #200, 2026-05-20).
#
# Tests whether the structural Basin B trap is specific to equivalence-pair
# loss formulations, or general to any cvvdp-using mechanism. Every V11/V12
# variant fires cross-codec-eq pair loss `(y_a − y_b)²` between codec pairs
# at matched anchor levels and hits KonJND collapse. V13 removes that loss
# entirely — pure distillation `(predicted − cvvdp_score)²` per (ref, dist)
# pair.
#
# Recipe deviations from task brief (documented):
#
# - Brief says "V_22-mix architecture (plain MLP h=128)" AND "mse_weight 0.5
#   + monotonicity_reg 0.5". The MSE / monotonicity aux losses only fire on
#   the per-sample-α head path (zensim-validate/src/mlp_train.rs:419 panics
#   otherwise). Distillation requires MSE on the target, so we use
#   per-sample-α head with cvvdp_score target. Architecture diverges from
#   V_22-mix-LARGE; the *target shape* (cvvdp instead of ssim2-derived
#   human_score) is what's being tested.
#
# - Brief lists 7 groups; only safesyn, kadid, tid, cvvdp_iwssim_LARGE have
#   100 % cvvdp_score coverage. konjnd-dense, cid22_train, pipal are 0 %
#   cvvdp-populated (subjective IQA sets, never cvvdp-scored). Dropped from
#   training per brief's fallback option (computing cvvdp on 60k images
#   blows the 3 hr budget).
#
# - Brief specifies `--target-column cvvdp_target_score`. We use
#   `cvvdp_score --target-scale 10.0` per the trainer's documented pattern
#   (CVVDP JOD ∈ [0, 10]; ×10 brings to 0..100 band-cutoff space).
#
# Args: <seed>
# Env:
#   USE_GPU=cuda|cpu      default cuda
#   KBATCH=N              default 32 (per task brief)
#   OUT_DIR=path          override output directory
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
USE_GPU="${USE_GPU:-cuda}"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v13_cvvdp_distill_2026-05-20}"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"

PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

BAKE="${OUT_DIR}/cc4v13_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v13_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v13_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "V13-CVVDP-DISTILL (task #200): seed=${SEED}"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"
echo "  kbatch:   ${KBATCH}"
echo "  gpu:      ${USE_GPU}"
echo "  target:   cvvdp_score (scaled x10 → 0..100 score space)"
echo "  groups:   safesyn + kadid + tid + cvvdp_iwssim_LARGE (cvvdp 100% coverage)"
echo "  dropped:  konjnd-dense + cid22_train + pipal (cvvdp 0% coverage)"

GPU_FLAG=""
if [ "${USE_GPU}" = "cuda" ]; then
    GPU_FLAG="--gpu-runtime cuda"
fi

# No --anchor-parquet, no --cross-codec-eq-parquet — V13 is pure distillation.
# Architecture: per-sample-α head (only path with MSE + monotonicity aux losses).
# Target: cvvdp_score, JOD ∈ [0, 10], scaled ×10 to 0..100 score space.
# max-features 300 — cvvdp_iwssim_LARGE has only 300 features.
"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.6:0.4" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.6:0.4" \
    --group "large:${PARQ_DIR}/cvvdp_iwssim_LARGE.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size "${KBATCH}" \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 300 --target-column cvvdp_score --target-scale 10.0 \
    --per-sample-alpha-head --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.5 --mse-weight 0.5 --monotonicity-reg 0.5 \
    ${GPU_FLAG} \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
