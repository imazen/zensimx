#!/usr/bin/env bash
# V46b — V32-faithful recipe (exact V39 manifest hyperparams) + --monotone-cbc.
#
# Isolates the pure cost of --monotone-cbc by matching V32's recipe
# point-by-point (2-layer 128→64, auto-transforms, target=human_score,
# tanh=30, val_aggregate=geomean3, 5 groups w/ V32's train_w:val_w).
# Compare V46b vs V39 (same recipe, no --monotone-cbc) → that delta
# is the constraint's intrinsic cost. Compare V46b vs V46 (1-layer,
# mix_cv40_iw60, tanh=20) → that delta is the recipe-choice cost.
set -euo pipefail
TRAIN="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
OUT="/mnt/v/output/zensim/bakes/v48_strict_tanh5_2026-05-26.bin"
LOG="/home/lilith/work/zen/zensimx/benchmarks/v48_strict_tanh5_2026-05-26.log"
CANON="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
TXFORM="/home/lilith/work/zen/zensimx/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
mkdir -p "$(dirname "$OUT")"
"$TRAIN" \
  --group safesyn:$CANON/safesyn.parquet:1.0:0.5 \
  --group cid22_train:$CANON/cid22_train_norm.parquet:1.5:2.0 \
  --group kadid:$CANON/kadid.parquet:0.5:1.0 \
  --group tid:$CANON/tid.parquet:0.5:1.0 \
  --group konjnd_dense:$CANON/konjnd-dense-norm.parquet:1.2:1.5 \
  --hidden 128 --n-hidden-layers 2 --per-sample-alpha-head --epochs 200 \
  --lr 0.001 --l2 0.0001 --seed 17 --target-column human_score --max-features 372 \
  --auto-transforms "$TXFORM" \
  --val-aggregate geomean3 --out-dtype f32 \
  --mse-weight 0.6 --ranknet-weight 0.6 --monotonicity-reg 1.0 \
  --tanh-output-head-scale 5.0 \
  --anchor-parquet $CANON/multiband_anchor_dial100.parquet \
  --anchor-loss-weight 0.01 --anchor-step-p 0.05 \
  --minibatch-size 32 --log-every 25 \
  --monotone-cbc --monotone-feature-mask /home/lilith/work/zen/zensimx/benchmarks/feature_sign_mask_2026-05-26.tsv --monotone-strict \
  --out "$OUT" --log-path "$LOG" 2>&1 | tee "${LOG}.stdout"
echo "DONE: $OUT"
