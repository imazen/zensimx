#!/usr/bin/env bash
# W12 cells for given seeds on a host that already passed the control with this trainer (the HOSTCTL of the host's own arm is copied in)
TAG=$1; SEEDS=$2; JOBS=$3; ST=/var/tmp/byc-fchroma-20261029/r2w$TAG; R=/home/lilith/work/zen/zensimx--fchroma
export PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs BYC_ROOT=$ST/root REV4_V2_ROOT=$ST/root REV4_V2_BIN_DIR=$ST/bin BYC_CELLS=$ST/cells BYC_SCRATCH=$ST/scratch ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZEN_PANEL_BIN=$ST/bin/panel TMPDIR=$HOME/tmp
mkdir -p $ST/cells $ST/scratch
echo "$(date -u +%H:%M:%S) cells seeds=$SEEDS start" >> $ST/PROGRESS.log
python3 -B $R/scripts/rev4_featpot/byc_cells.py run --arm mono:thr:mp32 --base-spec sel:45acffa603e8@h32:H128:cv16:cf98:fx2 --cols $ST/frozen_configuration/cols_fx2.txt --label Mlicfx2 --folds kadid,tid2013,konfig,cid22_a25 --seeds $SEEDS --jobs $JOBS > $ST/cells-$SEEDS.log 2>&1
echo "$(date -u +%H:%M:%S) cells seeds=$SEEDS rc=$?" >> $ST/PROGRESS.log
