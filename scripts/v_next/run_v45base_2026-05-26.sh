#!/usr/bin/env bash
# V45 — correct-by-construction monotone retrain of the V39 recipe.
# Identical to the V39-faithful base (MSE 0.6 + RankNet 0.6, anchor 0.01,
# seed 17, target human_score, 2-layer 128, tanh pin 30, YJ auto-transforms,
# by construction). NO konjnd-aggregation (the G5 lever craters the panel).
# Goal: retain V39's Mohammadi panel while gaining the G1+G3 guarantees.
set -euo pipefail
REPO="/home/lilith/work/zen/zensimx"
TRAIN="$REPO/target/release/zensim_mlp_train"
CANON="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
OUTDIR="/mnt/v/output/zensim/bakes"
TXFORM="$REPO/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
OUT="$OUTDIR/v45base_nomono_seed17_2026-05-26.bin"
LOG="$REPO/benchmarks/v45base_nomono_seed17_2026-05-26.log"
mkdir -p "$OUTDIR"
[ -x "$TRAIN" ] || { echo "trainer missing: $TRAIN" >&2; exit 2; }
"$TRAIN" \
  --group "safesyn:$CANON/safesyn.parquet:1.0:0.5" \
  --group "cid22_train:$CANON/cid22_train_norm.parquet:1.5:2.0" \
  --group "kadid:$CANON/kadid.parquet:0.5:1.0" \
  --group "tid:$CANON/tid.parquet:0.5:1.0" \
  --group "konjnd_dense:$CANON/konjnd-dense-norm.parquet:1.2:1.5" \
  --hidden 128 --n-hidden-layers 2 --per-sample-alpha-head --epochs 200 --lr 0.001 --l2 0.0001 --seed 17 \
  --target-column human_score --max-features 372 \
  --auto-transforms "$TXFORM" \
  --val-aggregate geomean3 --out-dtype f32 --mse-weight 0.6 --ranknet-weight 0.6 --monotonicity-reg 1.0 --tanh-output-head-scale 30.0 \
  --anchor-parquet "$CANON/multiband_anchor_dial100.parquet" --anchor-loss-weight 0.01 --anchor-step-p 0.05 \
  --minibatch-size 32 --log-every 25 --out "$OUT" 2>&1 | tee "$LOG"
echo "DONE: $OUT"
