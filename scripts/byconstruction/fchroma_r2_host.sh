#!/usr/bin/env bash
# round 2, one chroma-leg weight on one host: per-host control with the study's trainer, then 12 E21 cells and the seed-0 full fit+pack.
# usage: fchroma-r2-host.sh TAG JOBS
set -uo pipefail
TAG=$1; JOBS=$2; H=$(hostname)
S=/var/tmp/byc-fchroma-20261029; ST=$S/r2w$TAG; M=/var/tmp/byc-mixes-20261025; R=/home/lilith/work/zen/zensimx--fchroma
export PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs
log(){ echo "$(date -u +%H:%M:%S) $*" >> $ST/PROGRESS.log; }
# 1. control (original Mlic root, stored Mlic cell)
C=$ST/hostctl-cells-$H; rm -rf $C; mkdir -p $C $ST/scratch-hostctl
env BYC_ROOT=$M/Mlic/root REV4_V2_ROOT=$M/Mlic/root REV4_V2_BIN_DIR=$ST/bin BYC_CELLS=$C BYC_SCRATCH=$ST/scratch-hostctl ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN=$ST/bin/panel TMPDIR=$HOME/tmp \
  python3 -B $R/scripts/rev4_featpot/byc_cells.py run --arm mono:thr:mp32 --base-spec sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1 --cols $M/frozen_configuration/cols_c.txt --label Mlic --folds kadid --seeds 0 --jobs 1 || { log control-run-failed; exit 1; }
new=$C/Mlic/without_kadid_s0; old=$M/cells/Mlic/without_kadid_s0
$ST/bin/bake_dial_refit strip --in $new/refit/last.bin --out $ST/hostctl_$H.new.bin --key zentrain.repro >/dev/null
$ST/bin/bake_dial_refit strip --in $old/refit/last.bin --out $ST/hostctl_$H.old.bin --key zentrain.repro >/dev/null
python3 - $new $old $ST $H <<'P' || { log control-FAILED; exit 1; }
import sys,json,hashlib
new,old,S,H=sys.argv[1:]
sha=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
o={"host":H,"trainer_sha256":sha(S+"/bin/zensim_mlp_train"),"max_tier":"v3","prediction":[sha(new+"/pred_kadid.tsv"),sha(old+"/pred_kadid.tsv")],"bake_without_repro":[sha(S+f"/hostctl_{H}.new.bin"),sha(S+f"/hostctl_{H}.old.bin")]}
o["pass"]=o["prediction"][0]==o["prediction"][1] and o["bake_without_repro"][0]==o["bake_without_repro"][1]
json.dump(o,open(S+f"/HOSTCTL_{H}.json","w"),indent=1);print(json.dumps(o))
sys.exit(0 if o["pass"] else 1)
P
log control-pass
# 2. cells and the seed-0 full fit in parallel
export BYC_ROOT=$ST/root REV4_V2_ROOT=$ST/root REV4_V2_BIN_DIR=$ST/bin BYC_CELLS=$ST/cells BYC_SCRATCH=$ST/scratch ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN=$ST/bin/panel TMPDIR=$HOME/tmp
mkdir -p $ST/cells $ST/scratch $ST/full/Mlicfx2 $ST/scratch/full_s0
SPEC=sel:45acffa603e8@h32:H128:cv16:cf98:fx2
( TMPDIR=$ST/scratch/full_s0 python3 -B $R/scripts/rev4_featpot/v2_confirm_fit.py --root $ST/root --spec "$SPEC:mono:thr:mp32" --columns "$(cat $ST/frozen_configuration/cols_fx2.txt)" --head N --seed-index 0 --strict-admission --data-role-decision $ST/root/human_role_decision.json --dest $ST/full/Mlicfx2/full_s0 --train-only --pack-production --e33-output-stage > $ST/full/full_s0.log 2>&1; log "full-s0-rc=$?" ) &
python3 -B $R/scripts/rev4_featpot/byc_cells.py run --arm mono:thr:mp32 --base-spec $SPEC --cols $ST/frozen_configuration/cols_fx2.txt --label Mlicfx2 --folds kadid,tid2013,konfig,cid22_a25 --seeds 0,1,2 --jobs $JOBS > $ST/cells.log 2>&1; log "cells-rc=$?"
wait
$ST/bin/bake_dial_refit certify-monotone --bake $ST/full/Mlicfx2/full_s0/refit/production-f16.bin > $ST/full/full_s0.certify.json; log "ALL-DONE"
