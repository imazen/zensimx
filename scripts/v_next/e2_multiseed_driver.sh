#!/bin/bash
# E2 multi-seed confirmation: run both twin arms at seeds {7,13} (seed 1 already
# done), verdict each at --regime 720, so the pre-registered E2 bands get a
# seed-σ. SERIAL — machine-safety rule bans concurrent heavy jobs. Each arm is
# ~13 min train + ~25 s verdict; 4 arms ≈ 55 min.
#
# Usage: bash scripts/v_next/e2_multiseed_driver.sh
set -u
ZM=target/release/zensim_mlp_train
BV=target/release/bake_verdict
BAKES=/mnt/v/output/zensim/bakes
LOGD=/home/lilith/tmp
MARK=/home/lilith/work/zen/zensimx/.workongoing

for SEED in 7 13; do
  for ARM in ext720 v1372; do
    printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "claude-session-b30a" "E2 $ARM s$SEED train" > "$MARK"
    echo "=== [$( date -u +%H:%M:%S )] TRAIN $ARM seed $SEED ==="
    # trainer writes e2_${ARM}_s${SEED}_2026-07-23.bin
    ARM=$ARM SEED=$SEED ~/work/zen/scripts/run-heavy --mem 40G -- \
      bash scripts/v_next/e2_train_720_twin.sh > "$LOGD/e2_${ARM}_s${SEED}.log" 2>&1
    rc=$?
    echo "  train rc=$rc"
    BAKE="$BAKES/e2_${ARM}_s${SEED}_2026-07-23.bin"
    if [ $rc -ne 0 ] || [ ! -f "$BAKE" ]; then echo "  ARM FAILED — skip verdict"; continue; fi
    echo "=== [$( date -u +%H:%M:%S )] VERDICT $ARM seed $SEED ==="
    "$BV" --regime 720 --bake "$BAKE" \
      --output "$LOGD/verdict_${ARM}_s${SEED}.md" --json "$LOGD/verdict_${ARM}_s${SEED}.json" \
      > "$LOGD/verdict_${ARM}_s${SEED}.stdout" 2>&1
    echo "  verdict rc=$? -> $LOGD/verdict_${ARM}_s${SEED}.md"
  done
done
echo "E2-MULTISEED-DONE $(date -u +%H:%M:%S)"
