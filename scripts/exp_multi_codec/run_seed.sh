#!/usr/bin/env bash
# EXP-MULTI-CODEC: V_24-per-sample-α s4 recipe.
# Args: <seed> <out_dir> <large_parquet_path>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
LARGE="${3:?large_parquet_path}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_s${SEED}_h128.bin"
LOG="$OUT_DIR/persample_s${SEED}_h128.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train

CORPUS_ROOT=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer

"$TRAINER" \
  --group safesyn:${CORPUS_ROOT}/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:${CORPUS_ROOT}/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:${CORPUS_ROOT}/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:${CORPUS_ROOT}/konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:"$LARGE":0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
