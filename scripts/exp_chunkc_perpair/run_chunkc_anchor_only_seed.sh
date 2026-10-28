#!/usr/bin/env bash
# EXP-CHUNKC-PERPAIR ANCHOR-ONLY (re-attempt 2) — drops safesyn + cvvdp_iwssim_large
# from training (their CVVDP features are zero-filled). Trains on KADID + TID + KonJND
# only. ~14k pairs. h=128, per-sample-α.
#
# Args: <seed>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR=/mnt/v/zen/zensim-eval/exp_chunkc_perpair_2026-05-18
LOG_DIR=/tmp/exp_chunkc_perpair_logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

BAKE="$OUT_DIR/chunkc_anchor_s${SEED}_h128.bin"
LOG="$LOG_DIR/chunkc_anchor_s${SEED}.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-extfeat

"$TRAINER" \
  --group kadid:"$DATA/kadid_extfeat_343.parquet":1.0:1.0 \
  --group tid:"$DATA/tid_extfeat_343.parquet":1.0:1.0 \
  --group konjnd:"$DATA/konjnd_extfeat_343.parquet":0.1:1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 343 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
