#!/usr/bin/env bash
# Profile F round 3 (registered in benchmarks/byconstruction_2026-10-22.md, "Profile F round 3"): the continuations of the
# weight-1 arm (F3 general e59, F3codec e29) and the continuation source-fold study, on the filtered root.
#
#   fchroma_r3_cont.sh ready     prerequisites: w100 control PASS, full seeds 0-2 packed and certified, 12 E21 cells
#   fchroma_r3_cont.sh plans     fx2-plan (source folds), then full-plan general e59 and codec e29 (round 1's pattern)
#   fchroma_r3_cont.sh run NAME  folds | general | codec: run that plan under the dev compute lock (run-heavy)
#
# Everything lands in R3/stage2/. Labels stay Mlicfx2 (the arms are the filtered-root fx2 study). dev only, behind heavy.lock.
set -euo pipefail
cmd=$1
S=${S:-/var/tmp/byc-fchroma-20261029}
R3=$S/r3 ST=$R3/w100 R=$R3/stage2
C=${SOURCE_COMMIT:-84dfebc85ab8}
O=/var/tmp/byc-fx2-20261028/stage2/fx2-full-xrefaic256e59-run01/bin
REPRO=/var/tmp/byc-fx2-20261028/stage2/fx1-reproduction-run02
repo=$(cd "$(dirname "$0")/../.." && pwd)
ac=$repo/scripts/byconstruction/aic_matched_continuation.py
export PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs OPENBLAS_NUM_THREADS=1

case $cmd in
  ready)
    python3 -B -c "import json,sys; d=json.load(open('$ST/CONTROL.json')); sys.exit(0 if d['pass'] else 1)" || { echo "MISSING control"; exit 1; }
    for s in 0 1 2; do
      test -f $ST/full/Mlicfx2/full_s$s/refit/production-f16.bin && test -s $ST/full/full_s$s.certify.json || { echo "MISSING full seed $s"; exit 1; }
    done
    n=$(ls $ST/cells/Mlicfx2/*/refit/last.bin 2>/dev/null | wc -l); [ "$n" -ge 12 ] || { echo "MISSING E21 cells $n/12"; exit 1; }
    echo READY ;;
  plans)
    mkdir -p $R
    ( cd $repo
      python3 -B $ac fx2-plan $R/fx2-folds --source-commit $C --trainer $ST/bin/zensim_mlp_train --meta-tool $O/bake_dial_refit --study $ST --reproduction $REPRO
      python3 -B $ac full-plan $R/fx2-full-general-e59 --source-commit $C --trainer $ST/bin/zensim_mlp_train --meta-tool $O/bake_dial_refit --point $R/fx2-folds XrefAIC256 59 --after-run $R/fx2-folds
      python3 -B $ac full-plan $R/fx2-full-codec-e29 --source-commit $C --trainer $ST/bin/zensim_mlp_train --meta-tool $O/bake_dial_refit --point $R/fx2-folds AICOnly 29 --after-run $R/fx2-full-general-e59 ) ;;
  run)
    name=$2
    case $name in
      folds) dir=$R/fx2-folds mode=run jobs=4 ;;
      general) dir=$R/fx2-full-general-e59 mode=full-run jobs=12 ;;
      codec) dir=$R/fx2-full-codec-e29 mode=full-run jobs=12 ;;
    esac
    export REV4_V2_BIN_DIR=$dir/bin REV4_V2_ROOT=$ST/root BYC_ROOT=$ST/root AIC_BIN=$dir/bin ZEN_PANEL_BIN=$dir/bin/panel TMPDIR=$HOME/tmp
    exec flock $HOME/tmp/heavy.lock $HOME/work/claudehints/scripts/run-heavy --mem 30G --jobs $jobs -- python3 -B $dir/source/scripts/byconstruction/aic_matched_continuation.py $mode $dir/PLAN.json ;;
esac
