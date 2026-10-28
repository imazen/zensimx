#!/usr/bin/env bash
# Pack + harvest every replication-wave fit AS IT LANDS (LATENCY discipline:
# a late wake-up must cost nothing). Fails LOUD: a failure writes a marker,
# appends to FAILURES, and makes the final exit nonzero.
set -uo pipefail
REPO=/home/lilith/work/zen/zensimx
W=/mnt/v/output/zensim/replication-2026-09-05
CANON=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
export ZENSIM_FULLEVAL_OUT=$W/fulleval
export ZENSIM_VERDICT_DIR=$W/verdicts
mkdir -p "$ZENSIM_FULLEVAL_OUT" "$ZENSIM_VERDICT_DIR" "$W/logs"
P=$W/logs/POSTPROCESS.txt; F=$W/logs/POSTPROCESS_FAILURES.txt
: > "$F"; echo "$(date -u +%FT%TZ) postprocess start" >> "$P"
FAILS=0
for _ in $(seq 1 480); do
  N=$(python3 -c "import json;print(len(json.load(open('/home/lilith/tmp/replicate/fits.json'))))")
  PENDING=0
  for i in $(seq 0 $((N-1))); do
    TAG=$(python3 -c "import json;print(json.load(open('/home/lilith/tmp/replicate/fits.json'))[$i]['tag'])")
    RAW=$W/bakes/$TAG.bin; PK=$W/bakes/${TAG}_packed.bin
    [ -f "$W/logs/$TAG.done" ] || { PENDING=1; continue; }
    grep -q 'rc=0' "$W/logs/$TAG.done" || continue
    [ -f "$ZENSIM_FULLEVAL_OUT/${TAG}_packed.fulleval.json" ] && continue
    [ -f "$RAW" ] || { echo "$(date -u +%FT%TZ) MISSING RAW $TAG" | tee -a "$P" >> "$F"; FAILS=$((FAILS+1)); continue; }
    if [ ! -f "$PK" ]; then
      echo "$(date -u +%FT%TZ) pack $TAG" >> "$P"
      nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack --in "$RAW" --out "$PK" --neg-tail \
        --anchor "$CANON/anchor944_dial.parquet" --target-col target_score \
        --verify "$CANON/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
        >> "$W/logs/pack.log" 2>&1 || { echo "$(date -u +%FT%TZ) PACK FAILED $TAG" | tee -a "$P" >> "$F"; FAILS=$((FAILS+1)); continue; }
    fi
    echo "$(date -u +%FT%TZ) harvest ${TAG}_packed" >> "$P"
    "$REPO/scripts/harvest_bakes.sh" --bake "$PK" --regime 944 >> "$W/logs/harvest.log" 2>&1 \
      || { echo "$(date -u +%FT%TZ) HARVEST FAILED ${TAG}_packed" | tee -a "$P" >> "$F"; FAILS=$((FAILS+1)); }
  done
  if [ "$PENDING" = 0 ] && [ -f "$W/logs/CHAIN_DONE" ]; then break; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) postprocess done fails=$FAILS" >> "$P"
echo "fails=$FAILS" > "$W/logs/POSTPROCESS_DONE"
