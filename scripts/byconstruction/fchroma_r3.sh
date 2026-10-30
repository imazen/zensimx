#!/usr/bin/env bash
# Profile F round 3 (registered in benchmarks/byconstruction_2026-10-22.md, "Profile F round 3"): the four chroma-leg
# weight arms on the filtered roots (no fitting view with a long edge of 64 px or less), on one host.
#
#   fchroma_r3.sh prepare   pinned binaries and columns in every arm (fchroma_study.sh prepare with BIN_SRC)
#   fchroma_r3.sh run P     the per-host control, the full fits (weight 1: seeds 0-2; the other weights: seed 0) and the
#                           48 E21 cells, P at a time, in that order; every job leaves R3/jobs/<job>.{done,failed}
#   fchroma_r3.sh status    job counts and the control verdict
#
# R3/<arm>/root is built beforehand by ring_root.py chroma --drop-long-edge-at-most 64 and selection-rebind
# --declaration fx2. The caller holds ~/tmp/heavy.lock and run-heavy. Labels stay Mlicfx2 so the continuation owners
# (aic_matched_continuation.py fx2-plan) and fchroma_study.sh compare accept the arms unchanged.
set -euo pipefail
cmd=$1
R3=${R3:-/var/tmp/byc-fchroma-20261029/r3}
BIN_SRC=${BIN_SRC:-/var/tmp/zensim-bin-b0e70f30}
ARMS=(w100 w50 w25 w12)
repo=$(cd "$(dirname "$0")/../.." && pwd)
mixes=/var/tmp/byc-mixes-20261025

spec() {
  python3 -B -c "import sys; sys.path.insert(0, '$repo/scripts/rev4_featpot')
from v2_common import selection_id
print(selection_id([int(x) for x in open('$1/frozen_configuration/cols_fx2.txt').read().split(',')]))"
}

job() {  # job KIND ARM [FOLD] SEED
  local kind=$1 arm=$2 st=$R3/$2 name
  name=$(IFS=_; echo "$*")
  export BYC_ROOT=$st/root REV4_V2_ROOT=$st/root REV4_V2_BIN_DIR=$st/bin BYC_CELLS=$st/cells BYC_SCRATCH=$st/scratch
  export ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN=$st/bin/panel
  local rc=0 sel
  sel="sel:$(spec "$st")@h32:H128:cv16:cf98:fx2"
  case $kind in
    control) bash "$repo/scripts/byconstruction/fchroma_study.sh" control "$st" > "$R3/jobs/$name.log" 2>&1 || rc=$? ;;
    full)
      local seed=$3 dest=$st/full/Mlicfx2/full_s$3
      mkdir -p "$st/full/Mlicfx2" "$st/scratch/full_s$seed"
      TMPDIR=$st/scratch/full_s$seed python3 -B "$repo/scripts/rev4_featpot/v2_confirm_fit.py" --root "$st/root" \
        --spec "$sel:mono:thr:mp32" --columns "$(cat "$st/frozen_configuration/cols_fx2.txt")" --head N --seed-index "$seed" \
        --strict-admission --data-role-decision "$st/root/human_role_decision.json" --dest "$dest" --train-only \
        --pack-production --e33-output-stage > "$R3/jobs/$name.log" 2>&1 || rc=$?
      if [ $rc = 0 ]; then
        "$st/bin/bake_dial_refit" certify-monotone --bake "$dest/refit/production-f16.bin" > "$st/full/full_s$seed.certify.json" || rc=$?
      fi ;;
    cell)
      python3 -B "$repo/scripts/rev4_featpot/byc_cells.py" run --arm mono:thr:mp32 --base-spec "$sel" \
        --cols "$st/frozen_configuration/cols_fx2.txt" --label Mlicfx2 --folds "$3" --seeds "$4" --jobs 1 \
        > "$R3/jobs/$name.log" 2>&1 || rc=$? ;;
  esac
  if [ $rc = 0 ]; then touch "$R3/jobs/$name.done"; else echo "rc=$rc" > "$R3/jobs/$name.failed"; fi
  echo "$(date -u +%H:%M:%S) $name rc=$rc" >> "$R3/PROGRESS.log"
}

case "$cmd" in
  prepare)
    for arm in "${ARMS[@]}"; do ARM=fx2 BIN_SRC=$BIN_SRC bash "$repo/scripts/byconstruction/fchroma_study.sh" prepare "$R3/$arm"; done
    for arm in "${ARMS[@]}"; do cmp "$R3/w100/frozen_configuration/cols_fx2.txt" "$R3/$arm/frozen_configuration/cols_fx2.txt"; done ;;
  run)
    P=$2
    mkdir -p "$R3/jobs"
    jobs=("control w100")
    for seed in 0 1 2; do jobs+=("full w100 $seed"); done
    for arm in w50 w25 w12; do jobs+=("full $arm 0"); done
    for seed in 0 1 2; do for arm in "${ARMS[@]}"; do for fold in kadid tid2013 konfig cid22_a25; do
      jobs+=("cell $arm $fold $seed"); done; done; done
    export -f job spec
    export R3 repo mixes
    printf '%s\n' "${jobs[@]}" | while read -r line; do
      name=$(echo "$line" | tr ' ' _)
      [ -e "$R3/jobs/$name.done" ] || echo "$line"
    done | xargs -P "$P" -L 1 bash -c 'job "$@"' _ ;;
  status)
    echo "done $(ls "$R3"/jobs/*.done 2>/dev/null | wc -l) failed $(ls "$R3"/jobs/*.failed 2>/dev/null | wc -l) of 55"
    python3 -c "import json; print('control', json.load(open('$R3/w100/CONTROL.json'))['pass'])" 2>/dev/null || echo "control pending" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
