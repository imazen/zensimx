#!/usr/bin/env bash
# EXACT reproduction of the zensim-a (V39) recipe FAMILY — recovered from
# the prior session transcript (133ab28d), NOT the g5 konjnd-agg script my
# failed v45 reconstruction was based on. This is the v_tuner_v11 command
# (V0_3, CID22 0.860); V39 = the V32 variant of it (--mse 0.6 --ranknet 0.6,
# seed 17) + a post-hoc spline injection.
#
# KEY difference from my failed v45: --target-column mix_cv40_iw60 (a single
# consistent cvvdp×iwssim metric across all groups) — NOT human_score (which
# is per-group-inconsistent → overfit, CID22 0.295). Plus only 2 --group
# inputs (safesyn + cid22_train) + konjnd-aggregation + cross-codec-eq +
# dynamic-range-floor; NOT 5 explicit human-MOS groups.
set -euo pipefail
TRAIN="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
OUT="/mnt/v/output/zensim/bakes/repro_tuner_v11_s1_2026-05-26.bin"
LOG="/home/lilith/work/zen/zensimx/benchmarks/repro_tuner_v11_s1_2026-05-26.log"
mkdir -p "$(dirname "$OUT")"
"$TRAIN" \
  --group safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet:1.0:0.0 \
  --group cid22_train:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/cid22_train.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
  --val-policy min --early-stop-patience 0 --max-features 372 --minibatch-size 32 \
  --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
  --per-sample-alpha-head --tanh-output-head-scale 20.0 \
  --ranknet-weight 0.0 --mse-weight 1.0 --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
  --anchor-parquet /mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet \
  --anchor-loss-weight 0.5 --anchor-target-score 60.0 --anchor-step-p 0.30 \
  --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
  --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 --cross-codec-rank-preserve-weight 0.2 \
  --dynamic-range-floor-weight 0.3 --dynamic-range-sigma-threshold 25.0 --dynamic-range-step-p 0.05 --dynamic-range-probe-n 40 \
  --konjnd-aggregation-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/konjnd-dense.parquet \
  --konjnd-aggregation-weight 0.05 --konjnd-aggregation-step-p 0.10 --konjnd-aggregation-samples-per-ref 5 --konjnd-aggregation-refs-per-step 8 \
  --seed 1 --out "$OUT" --log-path "$LOG" 2>&1 | tee "${LOG}.stdout"
echo "DONE: $OUT"
