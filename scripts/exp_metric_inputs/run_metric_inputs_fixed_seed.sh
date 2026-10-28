#!/usr/bin/env bash
# EXP-METRIC-INPUTS-FIX — same recipe as exp-metric-inputs s4 but with the FIXED parquets
# from /mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed/ and no konjnd-dense.
#
# Defects addressed in this corpus (vs the original exp-metric-inputs run):
#   1. train/kadid + train/tid ssim2_gpu now per-pair (was constant per ref)
#   2. konjnd-dense dropped (was all-NaN); KonJND signal at inference via val/konjnd
#   3. val/konjnd has dist_path → metric scores now joined (was constant safesyn-mean)
#   4. All val ssim2 columns now GPU SSIMULACRA2 (was fast-ssim2 CPU)
#
# Args: <seed>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR=/mnt/v/zen/zensim-eval/exp_metric_inputs_fixed_2026-05-18
LOG_DIR=/tmp/exp_metric_inputs_fix_logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

BAKE="$OUT_DIR/metric_inputs_fixed_s${SEED}_h128.bin"
LOG="$LOG_DIR/metric_inputs_fixed_s${SEED}.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-metric-inputs-fixed

"$TRAINER" \
  --group safesyn:"$DATA/train/safesyn.parquet":1.0:0.0 \
  --group kadid:"$DATA/train/kadid.parquet":0.3:1.0 \
  --group tid:"$DATA/train/tid.parquet":0.3:1.0 \
  --group cvvdp_iwssim_large:"$DATA/train/cvvdp_iwssim_LARGE.parquet":0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 303 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
