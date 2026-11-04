#!/bin/bash
# R10 amendment 3: the CIQA confirmation read, registered in benchmarks/e36_subset_test_2026-11-01.md before it runs.
# Waits for the label-free feature bank (rev5_bank.py extract ciqa), then reads pairs.tsv once for the four dial
# candidates (F3 control, p30L, mixF3, mix + p30L; seeds 0-2) against F3's control at the same seed. Everything it
# writes stays under the private CIQA root.
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; R=/mnt/v/zensim/ciqa-private/e36-read
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59
while [ ! -f $R/EXTRACT.DONE ]; do sleep 60; done
grep -q "^extract rc=0" $R/bank_ciqa.log || { echo "extraction failed" >&2; exit 1; }
m=""
for s in 0 1 2; do
  c=$( [ $s = 0 ] && echo $T/control-s0 || echo $T/arms/control-s$s)
  m="$m --model control_s$s=$c/pack_e059/refit/production-f16.bin"
  m="$m --model p30L_s$s=$T/arms/p30L-s$s/pack_e059/refit/production-f16.bin"
  m="$m --model mixF3_s$s=$T/r10/runs/mixF3-s$s/pack_e059/refit/production-f16.bin"
  m="$m --model mix_p30L_s$s=$T/exp5c/runs/tid2013_mdid-p30L-s$s/pack_e059/refit/production-f16.bin"
done
cd $H/work/zen/zensimx
ZEN_PANEL_BIN=$T/bin/panel flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy --mem 12G --jobs 8 -- \
  python3 scripts/byconstruction/e36_exp56/ciqa.py confirm $R/confirm-1 --bank $R/bank/ciqa \
  --predict $S/bin/bake_dial_refit $m --vs control_s@ --registration "R10 amendment 3 (CIQA confirmation read)" \
  --arm "p30L=p30L_s0@0,p30L_s1@1,p30L_s2@2" --arm "mix on F3's recipe=mixF3_s0@0,mixF3_s1@1,mixF3_s2@2" \
  --arm "mix + p30L=mix_p30L_s0@0,mix_p30L_s1@1,mix_p30L_s2@2"
