#!/usr/bin/env bash
# Eval all 5 control seeds via bake_verdict, extract CID22 SROCC.
set -euo pipefail
BAKE_VERDICT=/home/lilith/work/zen/zensimx/target/release/bake_verdict
DIR=/mnt/v/zen/zensim-eval/exp_multi_codec_2026-05-18/control
VERDICTS=/mnt/v/zen/zensim-eval/exp_multi_codec_2026-05-18/verdicts
mkdir -p "$VERDICTS"

for s in 1 2 3 4 5; do
    BAKE="$DIR/persample_s${s}_h128.bin"
    OUT="$VERDICTS/persample_s${s}_h128.md"
    if [[ ! -f "$BAKE" ]]; then
        echo "MISSING: $BAKE"
        continue
    fi
    echo "=== seed $s ==="
    "$BAKE_VERDICT" --bake "$BAKE" --corpora cid22,kadid,tid,konjnd,aic3 --output "$OUT" 2>&1 | tail -5
done

echo
echo "=== CID22 SROCC per seed ==="
for s in 1 2 3 4 5; do
    OUT="$VERDICTS/persample_s${s}_h128.md"
    if [[ -f "$OUT" ]]; then
        cid=$(grep -E '\| CID22 \|' "$OUT" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        kad=$(grep -E '\| KADID \|' "$OUT" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        tid=$(grep -E '\| TID \|' "$OUT" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        konjnd=$(grep -E '\| KonJND \|' "$OUT" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        aic3=$(grep -E '\| AIC-3 \|' "$OUT" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        echo "s$s: CID22=$cid KADID=$kad TID=$tid KonJND=$konjnd AIC-3=$aic3"
    fi
done
