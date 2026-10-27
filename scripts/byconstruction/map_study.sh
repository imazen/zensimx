#!/usr/bin/env bash
# HISTORICAL: blocked by default; prepend --historical-replay for intentional reproduction.
# Map-study dumps after the 2026-10-24 map fixes (record: "Attribution-map correctness").
# TAGS=a,b runs only those tags.
set -euo pipefail
# Fail before argument expansion, output creation, data reads or builds.
python3 "$(dirname "${BASH_SOURCE[0]}")/historical_guard.py" "${BASH_SOURCE[0]}" "$@"
shift # explicit --historical-replay, validated above
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
BIN=zensim-bench/target/release/examples/map_dump
S=/mnt/v/output/zensim/byconstruction-2026-10-22/maps/study
TGA=$HOME/tmp/byc/parity/tga_full_rev6.bin
B=$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin
unset ZENSIM_FORMULA_REV ZENSIM_RING_FORM ZENSIM_MAP_WINDOW ZENSIM_MAP_PAINT ZENSIM_MAP_SPLIT ZENSIM_MAP_GUIDE ZENSIM_MAP_SMOOTH
# Columns: tag, bake, window, paint, split, smoothing sigma (default 0), guide norm (default sq).
while read -r tag bake window paint split smooth guide; do
  if [ -n "${TAGS:-}" ] && [[ ",$TAGS," != *",$tag,"* ]]; then continue; fi
  for set in main leak; do
    pairs=$S/pairs.tsv; [ "$set" = leak ] && pairs=$S/leak/pairs.tsv
    t0=$(date +%s)
    $BIN --pairs "$pairs" --out "$S/maps" --bake "$bake" --tag "$tag" --bin 1 --window "$window" --paint "$paint" --split "$split" --smooth "${smooth:-0}" --guide "${guide:-sq}" > "$S/maps/$tag.$set.jsonl"
    echo "$tag $set done in $(( $(date +%s) - t0 ))s"
  done
done <<LIST
tga_rbox $TGA box nearest repair
tga_ebox $TGA box nearest exact
tga_ecenter $TGA center nearest exact
tga_eguided $TGA guided nearest exact
tga_eguidedpx $TGA guided guided exact
tga_rguidedpx $TGA guided guided repair
b_rbox $B box nearest repair
b_eguidedpx $B guided guided exact
tga_ecenterpx $TGA center guided exact
b_ecenterpx $B center guided exact
tga_ecpx_s075 $TGA center guided exact 0.75
tga_ecpx_s15 $TGA center guided exact 1.5
tga_ecpx_s3 $TGA center guided exact 3
b_ecpx_s15 $B center guided exact 1.5
b_ecpx_s3 $B center guided exact 3
tga_ecpx_abs $TGA center guided exact 0 abs
tga_ecpx_abs_s15 $TGA center guided exact 1.5 abs
tga_ecpx_abs_s3 $TGA center guided exact 3 abs
b_ecpx_abs $B center guided exact 0 abs
b_ecpx_abs_s3 $B center guided exact 3 abs
LIST
echo ALL-DONE
