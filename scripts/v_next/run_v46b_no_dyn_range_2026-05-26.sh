#!/usr/bin/env bash
# V46 — first attempt at a monotone-by-construction A on the REAL
# V32-faithful recipe (recovered from session 133ab28d transcript,
# now also captured in scripts/v_next/repro_zensim_a_recipe_2026-05-26.sh).
#
# Key differences from my earlier (failed) v45 attempts:
#   - Correct recipe: target mix_cv40_iw60, 2 groups (safesyn + cid22_train),
#     konjnd-aggregation 0.05, cross-codec-eq, dynamic-range-floor 0.3,
#     anchor 0.5, lr 5.66e-3, tanh-scale 20, cid22_train.parquet (not _norm).
#     My v45-series used target=human_score + 5 explicit groups (wrong) →
#     CID22 overfit to 0.295. This recipe achieved CID22 0.860 for V0_3.
#   - V32 settings on top: --mse-weight 0.6 --ranknet-weight 0.6 --seed 17
#     (vs tuner_v11's mse 1.0 / rn 0.0 / seed 1) → CID22 0.8879 unconstrained.
#   - --monotone-cbc: soft sign-penalty during training + final hard
#     projection at every bake-emission (committed bf92de5e) →
#     encoder ≥0, rank_w ≤0, α≡1 → bake monotone-by-construction in every
#     non-negative dissimilarity feature.
set -euo pipefail
TRAIN="/home/lilith/work/zen/zensimx/target/release/zensim_mlp_train"
OUT="/mnt/v/output/zensim/bakes/v46b_no_dyn_range_seed17_2026-05-26.bin"
LOG="/home/lilith/work/zen/zensimx/benchmarks/v46b_no_dyn_range_seed17_2026-05-26.log"
mkdir -p "$(dirname "$OUT")"
"$TRAIN" \
  --group safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet:1.0:0.0 \
  --group cid22_train:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/cid22_train.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
  --val-policy min --early-stop-patience 0 --max-features 372 --minibatch-size 32 \
  --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
  --per-sample-alpha-head --tanh-output-head-scale 20.0 \
  --mse-weight 0.6 --ranknet-weight 0.6 \
  --monotonicity-reg 1.0 --monotonicity-margin 0.0 \
  --anchor-parquet /mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet \
  --anchor-loss-weight 0.5 --anchor-target-score 60.0 --anchor-step-p 0.30 \
  --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
  --cross-codec-eq-weight 1.0 --cross-codec-eq-step-p 0.10 --cross-codec-rank-preserve-weight 0.2 \
  --konjnd-aggregation-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/konjnd-dense.parquet \
  --konjnd-aggregation-weight 0.05 --konjnd-aggregation-step-p 0.10 --konjnd-aggregation-samples-per-ref 5 --konjnd-aggregation-refs-per-step 8 \
  --monotone-cbc \
  --seed 17 --out "$OUT" --log-path "$LOG" 2>&1 | tee "${LOG}.stdout"
echo "DONE: $OUT"
