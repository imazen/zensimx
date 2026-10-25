#!/bin/bash
# Relaunch of the two byc_globaff descriptor arms with the research forward path (byc_predict).
# Exploration record: benchmarks/byconstruction_2026-10-22.md. Usage: globaff_arms2.sh GLOBAFF_ROOT BIN_DIR JOBS
set -euo pipefail
GROOT=$1; BIN=$2; J=${3:-12}
H=$(cd "$(dirname "$0")" && pwd)/../rev4_featpot
export REV4_V2_BIN_DIR=$BIN RAYON_NUM_THREADS=1 BYC_ROOT=$GROOT
FOLDS=kadid,tid2013,konfig,cid22_a25
G=sel:0e6f70c166e6@h32:H128:cv16:cf98:fx1g; GC=$HOME/tmp/byc/cols_fx1g.txt
python3 "$H/byc_cells.py" run --arm mono:thr:freeglob --base-spec $G --cols $GC --label Tghpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Tghpm.log" 2>&1 &
python3 "$H/byc_cells.py" run --arm "" --base-spec $G --cols $GC --label Bghpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Bghpm.log" 2>&1 &
wait
echo ALL-ARMS-DONE
