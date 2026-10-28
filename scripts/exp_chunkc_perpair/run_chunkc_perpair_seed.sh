#!/usr/bin/env bash
# EXP-CHUNKC-PERPAIR (re-attempt 2026-05-18) — V_24 per-sample-α + 19 CVVDP-shape
# per-pair features (EX-4 Chunk C). Same recipe as V_24 Compression ship
# (run_persample_capacity_seed.sh) EXCEPT for --max-features 343 and the
# 343-col extfeat parquets.
#
# IMPORTANT: safesyn + cvvdp_iwssim_large have CVVDP features 0-filled
# (training files for those two corpora were never re-extracted). The
# 300 base features still get gradient from those large corpora; the
# 19 per-pair features get gradient ONLY from KADID + TID + KonJND
# (~14k pairs/epoch). This is the corpus-coverage gap the prior agent
# attempted to work around but mis-recipe'd (omitted --per-sample-alpha-head).
#
# Args: <seed>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR=/mnt/v/zen/zensim-eval/exp_chunkc_perpair_2026-05-18
LOG_DIR=/tmp/exp_chunkc_perpair_logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

BAKE="$OUT_DIR/chunkc_s${SEED}_h128.bin"
LOG="$LOG_DIR/chunkc_s${SEED}.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-extfeat

"$TRAINER" \
  --group safesyn:"$DATA/safesyn_extfeat_343.parquet":1.0:0.0 \
  --group kadid:"$DATA/kadid_extfeat_343.parquet":0.3:1.0 \
  --group tid:"$DATA/tid_extfeat_343.parquet":0.3:1.0 \
  --group konjnd:"$DATA/konjnd_extfeat_343.parquet":0.02:1.0 \
  --group cvvdp_iwssim_large:"$DATA/cvvdp_iwssim_large_extfeat_343.parquet":0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 343 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
