#!/usr/bin/env bash
# EXP-LARGER-LARGE: V_24 per-sample-α recipe retrained on the expanded LARGE corpus.
# Identical to scripts/v_next/run_persample_capacity_seed.sh for the V_24 ship,
# EXCEPT the cvvdp_iwssim_large parquet path points to the new expanded parquet.
#
# Args: <seed> <hidden> <out_dir> <large_parquet>
set -euo pipefail
SEED="${1:?seed}"
HIDDEN="${2:?hidden}"
OUT_DIR="${3:?out_dir}"
LARGE_PARQUET="${4:?large_parquet}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/larger_large_s${SEED}_h${HIDDEN}.bin"
LOG="$OUT_DIR/larger_large_s${SEED}_h${HIDDEN}.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train

CANONICAL=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer

"$TRAINER" \
  --group safesyn:"$CANONICAL"/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:"$CANONICAL"/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:"$CANONICAL"/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:"$CANONICAL"/konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:"$LARGE_PARQUET":0.5:0.0 \
  --hidden "$HIDDEN" --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED hidden=$HIDDEN bake=$BAKE log=$LOG"
