#!/usr/bin/env bash
# G5 (KonJND HF-rank) konjnd-aggregation-head sweep on the V39 2-layer
# recipe. Wires the now-unblocked 2-layer aggregation head; tries 3
# aggregation weights. Each run auto-runs bake_verdict (full Mohammadi
# panel + CODEC_TARGET_GOALS scorecard incl. G1 dial) and writes a
# <bake>.verdict.md sidecar.
#
# V39 baseline (features-root 2026-05-15-full-features):
#   KonJND 0.4197 (G5 floor 0.70) | CID22 0.879 | KADIK 0.925 |
#   TID 0.932 | AIC-3 0.802 | AIC-4 0.905 | G1 dial 1.00
set -euo pipefail

REPO="/home/lilith/work/zen/zensimx/.claude/worktrees/agent-a9484c7118fac8ded"
TRAIN="$REPO/target/release/zensim_mlp_train"
CANON="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
OUTDIR="/mnt/v/output/zensim/bakes"
TXFORM="$REPO/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
mkdir -p "$OUTDIR"

run_one() {
  local AGGW="$1"
  local TAG
  TAG=$(printf 'w%s' "$AGGW" | tr -d '.')
  local OUT="$OUTDIR/v42_konjnd_agg_${TAG}_2026-05-26.bin"
  local LOG="$REPO/benchmarks/v42_konjnd_agg_${TAG}_2026-05-26.log"
  echo "=== konjnd-aggregation weight=$AGGW -> $OUT ==="
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
    --konjnd-aggregation-parquet "$CANON/konjnd-dense.parquet" --konjnd-aggregation-weight "$AGGW" --konjnd-aggregation-step-p 0.15 \
    --minibatch-size 32 --log-every 50 --out "$OUT" 2>&1 | tee "$LOG"
}

for W in "${@:-0.3}"; do
  run_one "$W"
done
