#!/bin/bash
# M1 paired panels for arms already measured: p30L vs control, p30L + map 300 vs p30L, mix + p30L + map 300 vs p30L.
cd ~/work/zen/zensimx
run() { d=~/tmp/e36test/m1/$1; [ -e $d ] && return; ZEN_PANEL_BIN=$HOME/tmp/e36test/bin/panel RAYON_NUM_THREADS=6 nice -n19 python3 scripts/byconstruction/dataset_eval.py paired --out ~/tmp/e36test/eval --dest $d --a $2 --vs $3 --resamples 1000 > $d.log 2>&1 || echo "FAIL $1"; }
for s in 0 1 2; do
  run p30L-vs-control-s$s p30L_s$s control_s$s
  run map300-vs-p30L-s$s p30L_map300_s$s p30L_s$s
  run mixmap300-vs-p30L-s$s tid2013_mdid_p30L_map300_s$s p30L_s$s
  run mixnomap-vs-p30L-s$s tid2013_mdid_p30L_s$s p30L_s$s
done
echo done > ~/tmp/e36test/m1/existing.DONE
