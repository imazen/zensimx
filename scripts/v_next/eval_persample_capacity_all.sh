#!/usr/bin/env bash
# Evaluate every bake in exp_persample_capacity dir + emit a CI table.
set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/exp_persample_capacity_2026-05-18"
BV="/home/lilith/work/zen/zensimx/target/release/bake_verdict"

mkdir -p "$OUT_DIR/verdicts"

for bake in "$OUT_DIR"/*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    verdict="$OUT_DIR/verdicts/${name}.md"
    if [ ! -f "$verdict" ]; then
        echo "scoring $name"
        "$BV" --bake "$bake" --output "$verdict" 2>&1 | tail -n 4
    fi
done

echo
echo "== summary table =="
echo
printf '%-30s %-10s %-10s %-10s %-10s %-10s\n' "bake" "CID22" "KADID" "TID" "KonJND" "AIC-3"
printf '%-30s %-10s %-10s %-10s %-10s %-10s\n' "-----" "-----" "-----" "----" "------" "-----"
for v in "$OUT_DIR"/verdicts/*.md; do
    [ -f "$v" ] || continue
    name=$(basename "$v" .md)
    cid22=$(grep -E "^\| CID22 " "$v" | head -n 1 | awk -F'|' '{print $4}' | tr -d ' ')
    kadid=$(grep -E "^\| KADID " "$v" | head -n 1 | awk -F'|' '{print $4}' | tr -d ' ')
    tid=$(grep -E "^\| TID " "$v" | head -n 1 | awk -F'|' '{print $4}' | tr -d ' ')
    konjnd=$(grep -E "^\| KonJND " "$v" | head -n 1 | awk -F'|' '{print $4}' | tr -d ' ')
    aic3=$(grep -E "^\| AIC-3 " "$v" | head -n 1 | awk -F'|' '{print $4}' | tr -d ' ')
    printf '%-30s %-10s %-10s %-10s %-10s %-10s\n' "$name" "${cid22:-n/a}" "${kadid:-n/a}" "${tid:-n/a}" "${konjnd:-n/a}" "${aic3:-n/a}"
done
