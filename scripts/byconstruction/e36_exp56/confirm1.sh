#!/bin/bash
# Confirmation read 1 (zensimx record, registered 2026-11-03 before the read): CLIC 2021 validation.
set -euo pipefail
cd $HOME/work/zen/zensimx
m=""
for s in 0 1 2; do
  c=$([ $s = 0 ] && echo $HOME/tmp/e36test/control-s0 || echo $HOME/tmp/e36test/arms/control-s$s)
  m="$m --model control_s$s=$c/pack_e059/refit/production-f16.bin"
  m="$m --model p30L_s$s=$HOME/tmp/e36test/arms/p30L-s$s/pack_e059/refit/production-f16.bin"
  m="$m --model p30L_map300_s$s=$HOME/tmp/e36test/exp3m/runs/p30L-map300-s$s/pack_e059/refit/production-f16.bin"
  m="$m --model mix_p30L_map300_s$s=$HOME/tmp/e36test/exp5c/runs/tid2013_mdid-p30L-map300-s$s/pack_e059/refit/production-f16.bin"
done
ZEN_PANEL_BIN=$HOME/tmp/e36test/bin/panel flock $HOME/tmp/heavy.lock $HOME/work/claudehints/scripts/run-heavy --mem 12G --jobs 8 -- \
  python3 scripts/byconstruction/e36_exp56/clic.py confirm $HOME/tmp/e36test/clic21-confirm-1 --bank /mnt/v/zensim/dataset-features-rev7-clic-20261103/bank/clic21_valid \
  --predict /var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59/bin/bake_dial_refit $m --vs p30L_s@ \
  --arm "F3 control=control_s0@0,control_s1@1,control_s2@2" \
  --arm "p30L + map 300=p30L_map300_s0@0,p30L_map300_s1@1,p30L_map300_s2@2" \
  --arm "mix + p30L + map 300=mix_p30L_map300_s0@0,mix_p30L_map300_s1@1,mix_p30L_map300_s2@2"
