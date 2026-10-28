#!/usr/bin/env bash
# SPEED-B lr-retune sweep — single bake driver (2026-05-19, task #168).
#
# Trains a V6 recipe bake at K=32 (--minibatch-size 32) with a swept --lr.
# All other hyperparams identical to run_cross_codec_v6_seed.sh @ K=1.
#
# Hypothesis: lr=1e-3 was tuned for K=1 Adam dynamics. At K=32 the effective
# per-step gradient magnitude is ~K× larger, pushing optimization into a
# worse-generalizing basin. Test lr × √K ≈ 5.66 and lr × K = 32 directions.
#
# Args: <lr> <seed>
# Outputs: cc4v6_lr<lr>_s<SEED>.bin + .log + .stdout in OUT_DIR
set -euo pipefail

LR="${1:?usage: $0 <lr> <seed>}"
SEED="${2:?usage: $0 <lr> <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/speed_b_lr_retune_2026-05-19"
TRAINER="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

# Encode lr in the bake name (filesystem-friendly: 0.00283 -> 0p00283).
LR_TAG=$(printf '%s' "${LR}" | tr '.' 'p')
BAKE="${OUT_DIR}/cc4v6_lr${LR_TAG}_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v6_lr${LR_TAG}_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v6_lr${LR_TAG}_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

# Claim the cell: if a .bin already exists OR a stdout file shows an in-flight run,
# skip. Skips both completed and concurrent-in-flight cells, making the driver safe
# to dispatch under xargs even when a smoke test is already running.
if [ -s "${BAKE}" ]; then
    echo "SKIP (bake exists): lr=${LR} seed=${SEED}"
    exit 0
fi
if [ -f "${STDOUT}" ]; then
    # Check if a trainer process is still writing it (mtime within 5 min)
    AGE=$(( $(date +%s) - $(stat -c %Y "${STDOUT}") ))
    if [ "${AGE}" -lt 300 ]; then
        echo "SKIP (in-flight, stdout age ${AGE}s): lr=${LR} seed=${SEED}"
        exit 0
    fi
fi

echo "SPEED-B lr-retune: lr=${LR} seed=${SEED} K=32"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"

START_TS=$(date +%s)

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr "${LR}" --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 32 \
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
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    >"${STDOUT}" 2>&1

END_TS=$(date +%s)
WALL=$((END_TS - START_TS))
echo "DONE lr=${LR} seed=${SEED} K=32 wall=${WALL}s bake=${BAKE}"
printf '%s\t%s\t%s\n' "${LR}" "${SEED}" "${WALL}" >> "${OUT_DIR}/wall_times.tsv"
