#!/usr/bin/env bash
# Score a bake along the OOD blur ladder (identity → blur8) on 4 synthetic
# contents and check degradation-monotonicity + self-identity-max + dial
# range. Featmats emitted by tests/feature_distortion_direction.rs.
#   usage: eval_bake_blur_monotonicity.sh <bake.bin> <label>
set -euo pipefail
BAKE="$1"; LABEL="${2:-$(basename "$BAKE")}"
PRED=/home/lilith/work/zen/zensimx/target/release/predict_features_with_bake
echo "=== $LABEL  ($BAKE) ==="
TOTAL_INV=0; TOTAL_ABOVE_ID=0
GMIN=1e9; GMAX=-1e9
for c in color_blocks checker mandelbrot value_noise; do
  fm="/tmp/blur_ladder_${c}.featmat"
  [ -f "$fm" ] || { echo "  missing $fm — run the analysis test first"; exit 2; }
  # raw scores (post-spline, pre-clamp) so we see the true monotonicity incl. negatives / >100.
  scores=$("$PRED" --bake "$BAKE" --bake-post raw --features-file "$fm")
  readarray -t S <<< "$scores"
  id=${S[0]}
  line="  $c: "
  inv=0; above=0
  prev=999999
  for s in "${S[@]}"; do
    line+="$(printf '%.1f ' "$s")"
    # inversion: score rose vs previous (degradation must not raise score)
    awk "BEGIN{exit !($s > $prev + 0.01)}" && inv=$((inv+1)) || true
    awk "BEGIN{exit !($s > $id + 0.01)}" && above=$((above+1)) || true
    awk "BEGIN{exit !($s < $GMIN)}" && GMIN=$s || true
    awk "BEGIN{exit !($s > $GMAX)}" && GMAX=$s || true
    prev=$s
  done
  # 'above' counts identity itself (s>id+.01 is false for id), so it's real above-identity count
  line+=" | inversions=$inv above_identity=$above"
  echo "$line"
  TOTAL_INV=$((TOTAL_INV+inv)); TOTAL_ABOVE_ID=$((TOTAL_ABOVE_ID+above))
done
echo "  >>> $LABEL: total_inversions=$TOTAL_INV  total_above_identity=$TOTAL_ABOVE_ID  score_range=[$GMIN, $GMAX]"
echo "      (correct-by-construction → 0 inversions, 0 above-identity, wide range)"
