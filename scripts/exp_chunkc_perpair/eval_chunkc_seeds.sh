#!/usr/bin/env bash
# Eval 5 chunkc seeds + ship bakes. New bakes use 343col parquets,
# ship bakes use 372col parquets — both row-aligned per inspection.
set -euo pipefail

VERDICT=/home/lilith/work/zen/zensimx/target/release/bake_verdict
BAKE_DIR=/mnt/v/zen/zensim-eval/exp_chunkc_perpair_2026-05-18
SHIP_WEIGHTS=/home/lilith/work/zen/zensimx/zensim/weights
FEAT_343=/tmp/exp_chunkc_perpair_features_root
FEAT_372=/mnt/v/zen/zensim-training/2026-05-15-full-features
OUT_DIR=$BAKE_DIR/verdicts
mkdir -p "$OUT_DIR"

for s in 1 2 3 4 5; do
  bake="$BAKE_DIR/chunkc_s${s}_h128.bin"
  if [ ! -f "$bake" ]; then echo "MISSING: $bake"; continue; fi
  echo "=== chunkc s$s verdict ==="
  "$VERDICT" --bake "$bake" --features-root "$FEAT_343" --output "$OUT_DIR/chunkc_s${s}_verdict.md" 2>&1 | tail -3
done

# Ship: Balanced V_22-mix-LARGE+iwssim (372 feat)
echo "=== ship Balanced (V_22-mix-LARGE) ==="
"$VERDICT" --bake "$SHIP_WEIGHTS/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin" \
  --features-root "$FEAT_372" --output "$OUT_DIR/ship_balanced_verdict.md" 2>&1 | tail -3

# Ship: Compression V_24-per-sample-α s4 (372 feat)
echo "=== ship Compression (V_24-per-sample-α) ==="
"$VERDICT" --bake "$SHIP_WEIGHTS/v_compression_persample_2026-05-18.bin" \
  --features-root "$FEAT_372" --output "$OUT_DIR/ship_compression_verdict.md" 2>&1 | tail -3

echo "DONE. Verdicts at: $OUT_DIR/"
