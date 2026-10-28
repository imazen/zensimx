#!/usr/bin/env bash
# Tuner v11 retrain with --auto-transforms from the widest YJ screen.
#
# Task #214 Phase 2 (2026-05-25). Identical to run_tuner_v11_attempt7_seed.sh
# but layers --auto-transforms onto the v11 recipe to test whether the
# new feature-shaping (53 outright YJ wins + 262 features with ≥0.05
# Pearson lift) improves the ship-grade bake.
#
# Single-seed experiment — gates per CLAUDE.md 2026-05-14 are ADVISORY.
#
# Args: <seed>
# Defaults match the v11 ship recipe (KBATCH=32, LR=5.66e-3, ...).

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
LR="${LR_OVERRIDE:-5.66e-3}"
KONJND_AGG_WEIGHT="${KONJND_AGG_WEIGHT:-0.3}"
KONJND_AGG_STEP_P="${KONJND_AGG_STEP_P:-0.30}"
TANH_SCALE="${TANH_SCALE:-30.0}"

REPO="/home/lilith/work/zen/zensimx"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v11_yj_autotransforms_2026-05-25"
TRAINER="${REPO}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"
AUTO_TRANSFORMS="${REPO}/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
AUTO_TRANSFORMS_MIN_LIFT="${AUTO_TRANSFORMS_MIN_LIFT:-0.05}"

BAKE="${OUT_DIR}/tuner_v11_yj_at_s${SEED}.bin"
LOG="${OUT_DIR}/tuner_v11_yj_at_s${SEED}.log"
STDOUT="${OUT_DIR}/tuner_v11_yj_at_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

[ -x "${TRAINER}" ] || { echo "trainer missing" >&2; exit 2; }
[ -f "${AUTO_TRANSFORMS}" ] || { echo "auto-transforms TSV missing at ${AUTO_TRANSFORMS}" >&2; exit 2; }

echo "Tuner v11 YJ-autotransforms retrain: seed=${SEED}"
echo "  Training groups: safesyn:1.0 + cid22_train:0.5 + kadid:0.5 + tid:0.5 + konjnd_dense:0.3"
echo "  konjnd-aggregation: weight=${KONJND_AGG_WEIGHT} step_p=${KONJND_AGG_STEP_P}"
echo "  tanh_output_head_scale: ${TANH_SCALE}"
echo "  --auto-transforms: ${AUTO_TRANSFORMS} (min-lift ${AUTO_TRANSFORMS_MIN_LIFT})"
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
    --auto-transforms "${AUTO_TRANSFORMS}" \
    --auto-transforms-min-lift "${AUTO_TRANSFORMS_MIN_LIFT}" \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
