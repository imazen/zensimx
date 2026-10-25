#!/bin/bash
# Train the byc_globaff descriptor arms on the widened hpm root (Tghpm: monotone with the descriptor sign-free;
# Bghpm: B's signed recipe plus the descriptor) and the plain monotone control Thpm on the hpm root.
# Exploration record: benchmarks/byconstruction_2026-10-22.md. Usage: globaff_arms.sh HPM_ROOT GLOBAFF_ROOT BIN_DIR JOBS
set -euo pipefail
HROOT=$1; GROOT=$2; BIN=$3; J=${4:-8}
H=$(cd "$(dirname "$0")" && pwd)/../rev4_featpot
export REV4_V2_BIN_DIR=$BIN RAYON_NUM_THREADS=1
FOLDS=kadid,tid2013,konfig,cid22_a25
G=sel:0e6f70c166e6@h32:H128:cv16:cf98:fx1g; GC=$HOME/tmp/byc/cols_fx1g.txt
B=sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1; BC=$HOME/tmp/byc/cols_c.txt
BYC_ROOT=$GROOT python3 "$H/byc_cells.py" run --arm mono:thr:freeglob --base-spec $G --cols $GC --label Tghpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Tghpm.log" 2>&1 &
BYC_ROOT=$GROOT python3 "$H/byc_cells.py" run --arm "" --base-spec $G --cols $GC --label Bghpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Bghpm.log" 2>&1 &
BYC_ROOT=$HROOT python3 "$H/byc_cells.py" run --arm mono:thr --base-spec $B --cols $BC --label Thpm --folds $FOLDS --seeds 0,1,2 --jobs $J > "$HOME/tmp/byc/run_Thpm.log" 2>&1 &
wait
echo ALL-ARMS-DONE
