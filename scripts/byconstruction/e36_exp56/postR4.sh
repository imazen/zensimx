#!/bin/bash
# After R4 (DVIFM's painted map as the map-term target; option b1, the DVIFM-only arm): the amendment-2 set, compression rows, M1 against the twin
# (p30L + map 300 with butteraugli's target), held-out map agreement against butteraugli's and DVIFM's block sums,
# NEARID at seed 0, and the private AIC2026 out-of-fold check.   PRESET=<dvifm preset> postR4.sh
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; O=/mnt/v/zensim/e36-features/2026-11-01; L=$T/postR4; mkdir -p $L
PRESET=${PRESET:?set PRESET}
LOCK="flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy"
PRED=/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; P=$H/.cache/zensim-training-env/py3147-20261031/bin/python
NEW="p30L-map300dv-s0 p30L-map300dv-s1 p30L-map300dv-s2"
bake() { echo $T/r4/runs/$1/pack_e059/refit/production-f16.bin; }
col() { echo "$1" | tr -- '-+' '__'; }
step() { echo "== $(date -u +%H:%M:%SZ) $1"; }
cd $H/work/zen/zensimx
step "pack"; (cd $T && $LOCK --mem 36G --jobs 8 -- bash -c "for n in $NEW; do ./pack_nolock.sh r4/runs/\$n > r4/runs/\$n.pack.log 2>&1 & done; wait") > $L/pack.log 2>&1
for n in $NEW; do tail -1 $T/r4/runs/$n.pack.log | grep -q PACKED || echo "PACK FAIL $n"; done
D=/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes; for n in $NEW; do mkdir -p $D/r4-$n; cp --update=none $(bake $n) $T/r4/runs/$n/shipped.bin $D/r4-$n/; done
(cd $D && sha256sum */*.bin > SHA256SUMS && cp -r --update=none $D/. /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cp SHA256SUMS /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cd /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes && sha256sum -c --quiet SHA256SUMS && echo "bakes kept: $(wc -l < SHA256SUMS) files verified on tower")
step "held-out map agreement (butteraugli, then DVIFM $PRESET)"; E=$T/exp3; A=scripts/byconstruction/e36_exp56/map_agreement.py
for n in $NEW; do for b in 8 16; do $LOCK --mem 12G --jobs 8 -- $E/integrand_dump.bin --pairs $E/heldout/pairs.tsv --out $E/heldout/$n-$b --bake $(bake $n) --block $b > $E/heldout/$n-$b.jsonl 2>/dev/null
  echo "$n $b $(python3 $A $E/heldout/$n-$b.jsonl $E/heldout/$n-$b)"; done; done > $L/agreement.txt 2>&1
for n in f3 p30L p30L-map300-s0 p30L-map300-s1 p30L-map300-s2 $NEW; do for b in 8 16; do
  echo "$n $b $(python3 $A $E/heldout/$n-$b.jsonl $E/heldout/$n-$b $T/r4/maps/$PRESET/heldout-$b 2>/dev/null)"; done; done > $L/agreement-dvifm.txt 2>&1
step "E36 score, seeds 0-2"; args=""
for s in 0 1 2; do args="$args --bake control_s$s=$( [ $s = 0 ] && echo $T/control-s0 || echo $T/arms/control-s$s)/pack_e059/refit/production-f16.bin --bake p30L_s$s=$T/arms/p30L-s$s/pack_e059/refit/production-f16.bin"
  args="$args --bake p30L_map300_s$s=$T/exp3m/runs/p30L-map300-s$s/pack_e059/refit/production-f16.bin"; done
for n in $NEW; do args="$args --bake $(col $n)=$(bake $n)"; done
$LOCK --mem 36G --jobs 24 -- python3 scripts/byconstruction/e36_features.py score $O/bank $O/scores-R4 --predict $PRED $args > $L/score.log 2>&1 || echo FAIL score
m=""; for s in 0 1 2; do m="$m --model control_s$s --model p30L_s$s --model p30L_map300_s$s"; done; for n in $NEW; do m="$m --model $(col $n)"; done
$LOCK --mem 30G --jobs 8 -- python3 scripts/byconstruction/e36_features.py inversions $O/scores-R4 $H/tmp/e36feat/full /mnt/v/output/zensim/e36-bulk-2026-10-17/pools-full-v1 $O/inversions-R4 $m > $L/inversions.log 2>&1 || echo FAIL inversions
python3 $T/primary/primary.py $O/inversions-R4/pairs_scored.parquet $H/tmp/e36feat/full/selection.jsonl $T/primary/PRIMARY_R4.json control,p30L,p30L-map300,p30L-map300dv > $L/primary.log 2>&1 || echo FAIL primary
step "EVAL + error vs human + compression rows"; m=""; for n in $NEW; do m="$m --model $(col $n)=$(bake $n)"; done
$LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m > $L/eval-score.log 2>&1 || echo FAIL eval-score
M=""; for s in 0 1 2; do M="$M,p30L@$s=p30L_s$s,map300@$s=p30L_map300_s$s,map300dv@$s=p30L_map300dv_s$s"; done
ZEN_PANEL_BIN=$T/bin/panel $P $T/exp5/readout.py $T/eval ${M#,} $L/readout.json > $L/human_error.md 2>&1 || echo FAIL readout
ZEN_PANEL_BIN=$T/bin/panel $P scripts/byconstruction/e36_exp56/ceval.py $T/eval $L/ceval.json ${M#,} > $L/ceval.md 2>&1 || echo FAIL ceval
step "M1 paired against the twin"
for s in 0 1 2; do for lab in map300dv; do d=$T/m1/$lab-vs-twin-s$s
  [ -e $d ] || ZEN_PANEL_BIN=$T/bin/panel RAYON_NUM_THREADS=8 $LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py paired --out $T/eval --dest $d --a p30L_${lab}_s$s --vs p30L_map300_s$s --resamples 1000 > $d.log 2>&1 || echo FAIL paired $lab $s
done; done
python3 scripts/byconstruction/e36_exp56/m1_summary.py "DVIFM target minus butteraugli target=$T/m1/map300dv-vs-twin-s0,$T/m1/map300dv-vs-twin-s1,$T/m1/map300dv-vs-twin-s2" > $L/m1.md 2>&1 || echo FAIL m1
step "G-CHROMA-L"; args="--bake control_s0=$T/control-s0/pack_e059/refit/production-f16.bin"; for n in $NEW; do args="$args --bake $(col $n)=$(bake $n)"; done
$LOCK --mem 12G --jobs 4 -- python3 scripts/byconstruction/chroma_ladder_gate.py score --bank /var/tmp/byc-fchroma-20261029/c/merged-bank --predict $PRED --roles dev $args --out $T/gchroma/R4-dev --drop-long-edge-at-most 64 > $L/gchroma.log 2>&1 || echo FAIL gchroma
python3 scripts/byconstruction/chroma_ladder_gate.py report --out $T/gchroma/R4-dev > $L/gchroma-report.log 2>&1 || echo FAIL gchroma-report
step "G-STEER"; : > $T/gates/models-R4.txt; for n in $NEW; do echo "$(col $n) $(bake $n)" >> $T/gates/models-R4.txt; done
while read n b; do $LOCK --mem 36G --jobs 9 -- python3 scripts/byconstruction/gates_steer_sharded.py --name $n --model $b --out $T/gates/$n --bin /var/tmp/byc-fx2-20261028/stage2/gate-tools-d02d2e07/steer_instrument --formula-rev 7 --shards 9 > $T/gates/$n.steer.log 2>&1 || echo FAIL steer $n
  (cd $H/work/zen/zensimx--fchroma && PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs python3 -B scripts/byconstruction/gates.py --summary --out $T/gates/$n --name $n > $T/gates/$n.summary.json 2>$T/gates/$n.summary.err) || echo FAIL summary $n; done < $T/gates/models-R4.txt
step "NEARID (seed 0)"; R=/mnt/tower/output/zensim/mlic-nearid-20261027-run01; B=$T/nearid/bin/serve_custom_bake; : > $T/nearid/jobs-R4.txt
for n in p30L-map300dv-s0; do b=$(bake $n); echo "$(echo $n | tr -d -- "-_" | tr A-Z a-z) $b $(sha256sum < $b | cut -c1-64)" >> $T/nearid/jobs-R4.txt; done
$LOCK --mem 24G --jobs 24 -- bash -c 'while read l b h; do just dataset-nearid-candidate '$B' '$R'/PACKET.json ~/tmp/e36test/nearid/out-$l $l $b $h > ~/tmp/e36test/nearid/$l.log 2>&1 & done < ~/tmp/e36test/nearid/jobs-R4.txt; wait' > $L/nearid.log 2>&1 || echo FAIL nearid
step "AIC2026 out of fold (private)"
(cd $T && $LOCK --mem 36G --jobs 16 -- bash -c 'for d in expaic/runs/map300dv-k?; do ./pack_nolock.sh $d > $d.pack.log 2>&1 & while [ $(jobs -r | wc -l) -ge 7 ]; do sleep 2; done; done; wait') > $L/aic-pack.log 2>&1
for d in $T/expaic/runs/map300dv-k?; do tail -1 $d.pack.log | grep -q PACKED || echo "PACK FAIL $d"; done > $L/aic-pack-check.txt
cp $T/expaic/OOF_PANEL.json $T/expaic/OOF_PANEL_before_R4.json
$LOCK --mem 12G --jobs 8 -- python3 $T/expaic/score.py p30L,control,map300,map300dv > $T/expaic/score-R4.log 2>&1 || echo FAIL aic-score
(cd $T/expaic && ZEN_PANEL_BIN=$T/bin/panel python3 paired.py map300,map300dv > paired-R4.log 2>&1 \
  && BASE=map300 ZEN_PANEL_BIN=$T/bin/panel python3 paired.py map300dv >> paired-R4.log 2>&1 \
  && BASE=control ZEN_PANEL_BIN=$T/bin/panel python3 paired.py map300dv >> paired-R4.log 2>&1) || echo FAIL aic-paired
step "done"
