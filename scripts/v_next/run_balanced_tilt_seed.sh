#!/usr/bin/env bash
# Per-sample α head + boosted synthetic group weights — single seed.
# Args: <cell_name> <seed> <safesyn_w> <kadid_w> <tid_w> <konjnd_w> <large_w> <out_dir>
set -euo pipefail

CELL_NAME="${1:?cell_name}"
SEED="${2:?seed}"
SAFESYN_W="${3:?safesyn_w}"
KADID_W="${4:?kadid_w}"
TID_W="${5:?tid_w}"
KONJND_W="${6:?konjnd_w}"
LARGE_W="${7:?large_w}"
OUT_DIR="${8:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/${CELL_NAME}_seed${SEED}.bin"
LOG="$OUT_DIR/${CELL_NAME}_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer

"$TRAINER" \
  --group "safesyn:$DATA/safesyn_mix_300col.parquet:${SAFESYN_W}:0.0" \
  --group "kadid:$DATA/kadid_mix_300col.parquet:${KADID_W}:1.0" \
  --group "tid:$DATA/tid_mix_300col.parquet:${TID_W}:1.0" \
  --group "konjnd:$DATA/konjnd_mix_300col.parquet:${KONJND_W}:1.0" \
  --group "cvvdp_iwssim_large:$DATA/cvvdp_iwssim_large_300col_v2.parquet:${LARGE_W}:0.0" \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE cell=$CELL_NAME seed=$SEED bake=$BAKE log=$LOG"
