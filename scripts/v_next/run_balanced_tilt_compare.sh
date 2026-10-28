#!/usr/bin/env bash
# Run bake_compare A vs B over all 5 corpora with 1000-bootstrap.
# Args: <bake_a> <bake_b> <output_md>
set -euo pipefail

BAKE_A="${1:?bake_a}"
BAKE_B="${2:?bake_b}"
OUT_MD="${3:?out_md}"

BC=/home/lilith/work/zen/zensimx/target/release/bake_compare

"$BC" --a "$BAKE_A" --b "$BAKE_B" \
  --corpora cid22,kadid,tid,konjnd,aic3 \
  --bands 10 \
  --bootstrap-resamples 1000 \
  --features-root /mnt/v/zen/zensim-training/2026-05-15-full-features \
  --seed 42 \
  --output "$OUT_MD"

echo "DONE compare $BAKE_A vs $BAKE_B -> $OUT_MD"
