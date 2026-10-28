#!/usr/bin/env bash
# SPEED-B lr-retune sweep — dispatch 5 lrs × 3 seeds = 15 bakes, max 5 parallel.
#
# Each bake uses RAYON_NUM_THREADS=6 (5 procs × 6 threads = 30 threads on 32-core box).
set -euo pipefail

DRIVER="/home/lilith/work/zen/zensimx/scripts/v_next/run_speed_b_lr_retune_seed.sh"
OUT_DIR="/mnt/v/zen/zensim-eval/speed_b_lr_retune_2026-05-19"
PAIRS_FILE="${OUT_DIR}/sweep_pairs.txt"

mkdir -p "${OUT_DIR}"

# Emit (lr, seed) pairs. Skip cells already complete (.bin present).
> "${PAIRS_FILE}"
for LR in 1e-3 1.5e-3 2.83e-3 5.66e-3 8e-3; do
    LR_TAG=$(printf '%s' "${LR}" | tr '.' 'p')
    for SEED in 1 2 3; do
        BAKE="${OUT_DIR}/cc4v6_lr${LR_TAG}_s${SEED}.bin"
        if [ -s "${BAKE}" ]; then
            echo "skip-complete: ${LR} ${SEED}"
            continue
        fi
        echo "${LR} ${SEED}" >> "${PAIRS_FILE}"
    done
done

echo
echo "=== sweep pairs to run ==="
cat "${PAIRS_FILE}"
echo "=== launching with 5-way parallelism, RAYON_NUM_THREADS=6 ==="
echo

# xargs to run 5 in parallel. Each bake takes ~12-15 min at K=32.
export RAYON_NUM_THREADS=6
export DRIVER
xargs -a "${PAIRS_FILE}" -n 2 -P 5 bash -c \
    'set -euo pipefail; "${DRIVER}" "$1" "$2"' _

echo
echo "=== all bakes complete; wall times: ==="
cat "${OUT_DIR}/wall_times.tsv" 2>&1 || true
