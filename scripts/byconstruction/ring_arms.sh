#!/bin/bash
# Train recipe arms on a derived root (ring_root.py), 12 E21 cells each. Default arms: the ringing-corrected (hpm)
# set, B's signed recipe, certified-monotone M and FR (ringing sign-free); or LABEL=TOKENS pairs, e.g.
# M64lin=mono:thr:mp64 on the linear-pyramid root. Exploration record: benchmarks/byconstruction_2026-10-22.md.
# Usage: ring_arms.sh ROOT BIN_DIR JOBS_PER_ARM [LABEL=TOKENS ...]
set -euo pipefail
ROOT=$1; BIN=$2; J=${3:-8}
shift 3 || true
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=("Bhpm=" "Mhpm=mono:thr:mp32" "FRhpm=mono:thr:freering")
H=$(cd "$(dirname "$0")" && pwd)/../rev4_featpot
export BYC_ROOT=$ROOT REV4_V2_BIN_DIR=$BIN RAYON_NUM_THREADS=1
# BASE / COLS select another input declaration, e.g. the orientation arm's
# BASE=sel:46516c473ae7@h32:H128:cv16:cf98:fx1o COLS=$HOME/tmp/byc/cols_fx1o.txt
BASE=${BASE:-sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1}
COLS=${COLS:-$HOME/tmp/byc/cols_c.txt}
FOLDS=kadid,tid2013,konfig,cid22_a25
for arm in "${ARMS[@]}"; do
  label=${arm%%=*}; tokens=${arm#*=}
  python3 "$H/byc_cells.py" run --arm "$tokens" --base-spec $BASE --cols $COLS --label "$label" --folds $FOLDS \
    --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_$label.log" 2>&1 &
done
wait
echo ALL-ARMS-DONE
