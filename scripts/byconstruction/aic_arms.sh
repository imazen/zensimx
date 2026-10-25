#!/bin/bash
# E21 cells that also fit AIC2026 TRAIN (pooled rank, one AIC fold held out per cell, research replay) on the widened
# hpm + descriptor root. Exploration record: benchmarks/byconstruction_2026-10-22.md, section "AIC2026 TRAIN".
# Usage: aic_arms.sh GLOBAFF_ROOT BIN_DIR JOBS AIC_WEIGHT ARM... where ARM is LABEL=TOKENS (e.g. Tga=mono:thr:freeglob).
# BASE=s1 uses fx1gs1 (no scale-0 slot) instead of fx1g.
set -euo pipefail
GROOT=$1; BIN=$2; J=$3; W=$4; shift 4
H=$(cd "$(dirname "$0")" && pwd)/../rev4_featpot
export REV4_V2_BIN_DIR=$BIN RAYON_NUM_THREADS=1
G=sel:0e6f70c166e6@h32:H128:cv16:cf98:fx1g; GC=$HOME/tmp/byc/cols_fx1g.txt
if [ "${BASE:-}" = s1 ]; then G=sel:d6e48dbce419@h32:H128:cv16:cf98:fx1gs1; GC=$HOME/tmp/byc/cols_fx1gs1.txt; fi
for arm in "$@"; do
  label=${arm%%=*}; tokens=${arm#*=}
  BYC_ROOT=$GROOT python3 "$H/byc_cells.py" run --arm "$tokens" --base-spec $G --cols $GC --label "$label" \
    --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs "$J" --aic-weight "$W" > "$HOME/tmp/byc/run_$label.log" 2>&1
  echo "$label done $(date -u +%H:%M:%S)"
done
echo AIC-ARMS-DONE
