#!/usr/bin/env bash
# EXP-METRIC-INPUTS — V_24 per-sample-α recipe with 3 metric inputs as additional features.
# Same recipe as V_24 Compression ship (run_persample_capacity_seed.sh) EXCEPT for
# --max-features 303 and the 303-col metric-inputs parquets.
#
# Training corpora (from canonical-2026-05-18 with f300/f301/f302 appended):
#   safesyn (196k rows):           cvvdp + iwssim + ssim2 all 100% populated
#   kadid (10k):                   constant ssim2 per image (data limitation), cvvdp+iwssim per-pair OK
#   tid (3k):                      same constant-ssim2 limitation
#   konjnd-dense: SKIPPED (all 3 metric scores NaN per the build script)
#   cvvdp_iwssim_LARGE (73k):      ssim2 imputed (constant), cvvdp+iwssim per-pair OK
#
# Args: <seed>
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR=/mnt/v/zen/zensim-eval/exp_metric_inputs_2026-05-18
LOG_DIR=/tmp/exp_metric_inputs_logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

BAKE="$OUT_DIR/metric_inputs_s${SEED}_h128.bin"
LOG="$LOG_DIR/metric_inputs_s${SEED}.log"
TRAINER=/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-metric-inputs

"$TRAINER" \
  --group safesyn:"$DATA/train/safesyn.parquet":1.0:0.0 \
  --group kadid:"$DATA/train/kadid.parquet":0.3:1.0 \
  --group tid:"$DATA/train/tid.parquet":0.3:1.0 \
  --group konjnd:"$DATA/train/konjnd-dense.parquet":0.02:1.0 \
  --group cvvdp_iwssim_large:"$DATA/train/cvvdp_iwssim_LARGE.parquet":0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 303 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
