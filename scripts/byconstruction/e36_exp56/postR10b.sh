#!/bin/bash
# R10 amendment 2: the product gates postR10.sh did not run for mixF3 (seeds 0-2): G-TRADE (a), the dial contract and
# identity (fchroma-gates.sh verdict,identity), NEARID at seeds 1-2; then confirmation read 3 (CLIC 2021 validation).
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; L=$T/postR10b; mkdir -p $L
LOCK="flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy"
PRED=/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59
NEW="mixF3-s0 mixF3-s1 mixF3-s2"; CTRL="control-s0 control-s1 control-s2"
bake() { case $1 in control-s0) echo $T/control-s0/pack_e059/refit/production-f16.bin;; control-*) echo $T/arms/$1/pack_e059/refit/production-f16.bin;; *) echo $T/r10/runs/$1/pack_e059/refit/production-f16.bin;; esac; }
col() { echo "$1" | tr '-' '_'; }
step() { echo "== $(date -u +%H:%M:%SZ) $1"; }
cd $H/work/zen/zensimx
step "G-TRADE"; G=$T/gtrade; TAG=R10; args=""; keys=""; for n in $NEW; do args="$args --bake $(col $n)=$(bake $n)"; keys="$keys --key $(col $n)=e36-$n"; done
$LOCK --mem 36G --jobs 24 -- python3 scripts/byconstruction/gauntlet_features.py score /mnt/v/zensim/gauntlet-features/2026-10-31/bank $G/scores-$TAG --predict $PRED $args > $L/gtrade-score.log 2>&1 || echo FAIL gtrade-score
nice -n19 python3 scripts/byconstruction/gauntlet_features.py augment $G/scores-$TAG $G/aug $G/aug-$TAG $keys > $L/gtrade-augment.log 2>&1 || echo FAIL gtrade-augment
M="zensim-e zensim-f3"; for n in $CTRL $NEW; do M="$M e36-$n"; done
(cd $H/work/zen/evif && $LOCK --mem 24G -- python3 tools/trade_gate.py $(ls -d $G/aug-$TAG/*/) --dims $H/tmp/gauntlet-features/view-dims.json --out $G/gate-$TAG --models $M > $L/gtrade-gate.log 2>&1) || echo FAIL gtrade-gate
python3 $T/gtrade_guard.py $G/gate-$TAG/TALLIES.json $M > $L/gtrade-guard.txt 2>&1 || echo FAIL gtrade-guard
step "dial contract and identity"; : > $T/gates/models-R10b.txt; for n in $NEW; do echo "$(col $n) $(bake $n)" >> $T/gates/models-R10b.txt; done
(cd $T && $LOCK --mem 36G --jobs 24 -- bash -c 'xargs -P3 -L1 bash -c '"'"'STEPS=verdict,identity bash ~/tmp/fchroma-gates.sh $0 $1 ~/tmp/e36test/gates/$0-vi > ~/tmp/e36test/gates/$0-vi.log 2>&1 || echo FAIL $0'"'"' < gates/models-R10b.txt' > $L/gates-vi.log 2>&1) || echo FAIL gates-vi
step "NEARID (seeds 1-2)"; R=/mnt/tower/output/zensim/mlic-nearid-20261027-run01; B=$T/nearid/bin/serve_custom_bake; : > $T/nearid/jobs-R10b.txt
for n in mixF3-s1 mixF3-s2; do b=$(bake $n); echo "$(echo $n | tr -d -- "-_" | tr A-Z a-z) $b $(sha256sum < $b | cut -c1-64)" >> $T/nearid/jobs-R10b.txt; done
$LOCK --mem 24G --jobs 24 -- bash -c 'while read l b h; do just dataset-nearid-candidate '$B' '$R'/PACKET.json ~/tmp/e36test/nearid/out-$l $l $b $h > ~/tmp/e36test/nearid/$l.log 2>&1 & done < ~/tmp/e36test/nearid/jobs-R10b.txt; wait' > $L/nearid.log 2>&1 || echo FAIL nearid
step "confirmation read 3 (CLIC 2021 validation)"; m=""
for s in 0 1 2; do c=$( [ $s = 0 ] && echo $T/control-s0 || echo $T/arms/control-s$s); m="$m --model control_s$s=$c/pack_e059/refit/production-f16.bin --model mixF3_s$s=$(bake mixF3-s$s)"
  m="$m --model mix_p30L_s$s=$T/exp5c/runs/tid2013_mdid-p30L-s$s/pack_e059/refit/production-f16.bin"; done
ZEN_PANEL_BIN=$T/bin/panel $LOCK --mem 12G --jobs 8 -- python3 scripts/byconstruction/e36_exp56/clic.py confirm $T/clic21-confirm-3 \
  --bank /mnt/v/zensim/dataset-features-rev7-clic-20261103/bank/clic21_valid --predict $S/bin/bake_dial_refit $m --vs control_s@ \
  --registration "confirmation read 3 (R10 amendment 2)" \
  --arm "mix on F3's recipe=mixF3_s0@0,mixF3_s1@1,mixF3_s2@2" --arm "mix + p30L=mix_p30L_s0@0,mix_p30L_s1@1,mix_p30L_s2@2" > $L/confirm3.log 2>&1 || echo FAIL confirm3
step "done"
