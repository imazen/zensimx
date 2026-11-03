#!/bin/bash
# usage: pack.sh DIR  — F3's pack step (zensim-training recipes/F3.json "pack", program a2ac1e1c) on DIR/ckpt/ckpt_epoch059.bin,
# then the recipe's strip step to DIR/shipped.bin; prints the stripped-keys sha256 of production-f16.bin.
set -euo pipefail
D=$1; S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; R=${ROOT:-/var/tmp/byc-fchroma-20261029/r3/w100/root}
P=~/work/zen/zensim-training/recipes/programs/a2ac1e1c460c99634d058ac4c50d8843fbf622a5c41089fe5ad8f4086bd09fc8.py
test ! -e $D/pack_e059; mkdir -p $D/pack_e059/refit  # the recipe runner creates refit/ before the program runs
mkdir -p ~/tmp/e36test/scratch/pack-$(basename $D)
env ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 REV4_V2_BIN_DIR=$S/bin REV4_V2_ROOT=$R \
  ZEN_PANEL_BIN=$S/bin/panel PYTHONPATH=$S/source/scripts/rev4_featpot:$S/source/scripts TMPDIR=~/tmp/e36test/scratch/pack-$(basename $D) \
  \
  ~/.cache/zensim-training-env/py3147-20261031/bin/python -B $P $D/ckpt/ckpt_epoch059.bin $D/pack_e059 $R/wide/main/real/cid22_fit.parquet --root $R
B=$S/bin/bake_dial_refit
$B strip --in $D/pack_e059/refit/production-f16.bin --out $D/shipped.bin.tmp --key zentrain.repro
$B strip --in $D/shipped.bin.tmp --out $D/shipped.bin --key zentrain.sample_coverage && rm $D/shipped.bin.tmp
echo "PACKED $D shipped=$(sha256sum < $D/shipped.bin | cut -c1-64)"
