#!/bin/bash
# After wave F: experiment 6 walk seeds (EVAL), experiment 5 stage 3 and map weights 100-1000 (amendment-2 measures).
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; O=/mnt/v/zensim/e36-features/2026-11-01; L=$T/postF; mkdir -p $L
LOCK="flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy"
PRED=/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; P=$H/.cache/zensim-training-env/py3147-20261031/bin/python
CONT="exp3m:p30L-map100-s0 exp3m:p30L-map300-s0 exp3m:p30L-map1000-s0"
dir() { echo $T/${1%%:*}/runs/${1#*:}; }
col() { echo "${1#*:}" | tr -- '-+' '__'; }
bake() { echo $(dir $1)/pack_e059/refit/production-f16.bin; }
step() { echo "== $(date -u +%H:%M:%SZ) $1"; }
cd $H/work/zen/zensimx
step "walk seeds: EVAL"; m=""; for n in walk-s1 walk-s2; do m="$m --model x6_$(echo $n | tr - _)=$T/exp6/runs/$n/refit/production-f16.bin"; done
$LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m > $L/walk-eval.log 2>&1 || echo FAIL walk-eval
grep -h "\[coverage\]" $T/exp6/runs/walk-s*/train.log $T/exp5/runs/control-s*/train.log > $L/coverage.txt
step "pack"; (cd $T && $LOCK --mem 36G --jobs 12 -- bash -c "for x in $CONT; do ./pack_nolock.sh \${x%%:*}/runs/\${x#*:} > \${x%%:*}/runs/\${x#*:}.pack.log 2>&1 & done; wait") > $L/pack.log 2>&1
for x in $CONT; do tail -1 $(dir $x).pack.log | grep -q PACKED || echo "PACK FAIL $x"; done
D=/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes; for x in $CONT; do n=${x#*:}; mkdir -p $D/$n; cp --update=none $(bake $x) $(dir $x)/shipped.bin $D/$n/; done
(cd $D && sha256sum */*.bin > SHA256SUMS && cp -r --update=none $D/. /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cp SHA256SUMS /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cd /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes && sha256sum -c --quiet SHA256SUMS && echo "bakes kept: $(wc -l < SHA256SUMS) files verified on tower")
step "held-out map agreement"; E=$T/exp3
for x in $CONT; do n=${x#*:}; for b in 8 16; do $LOCK --mem 12G --jobs 8 -- $E/integrand_dump.bin --pairs $E/heldout/pairs.tsv --out $E/heldout/$n-$b --bake $(bake $x) --block $b > $E/heldout/$n-$b.jsonl 2>/dev/null
  echo "$n $b $(python3 $E/map_agreement.py $E/heldout/$n-$b.jsonl $E/heldout/$n-$b)"; done; done > $L/agreement.txt 2>&1
step "E36 score"; args=""
for s in 0 1 2; do args="$args --bake control_s$s=$( [ $s = 0 ] && echo $T/control-s0 || echo $T/arms/control-s$s)/pack_e059/refit/production-f16.bin --bake p30L_s$s=$T/arms/p30L-s$s/pack_e059/refit/production-f16.bin"; done
for x in $CONT; do args="$args --bake $(col $x)=$(bake $x)"; done
$LOCK --mem 36G --jobs 24 -- python3 scripts/byconstruction/e36_features.py score $O/bank $O/scores-F --predict $PRED $args > $L/score.log 2>&1 || echo FAIL score
step "inversions"; m=""; for s in 0 1 2; do m="$m --model control_s$s --model p30L_s$s"; done; for x in $CONT; do m="$m --model $(col $x)"; done
$LOCK --mem 30G --jobs 8 -- python3 scripts/byconstruction/e36_features.py inversions $O/scores-F $H/tmp/e36feat/full /mnt/v/output/zensim/e36-bulk-2026-10-17/pools-full-v1 $O/inversions-F $m > $L/inversions.log 2>&1 || echo FAIL inversions
step "primary"
SEEDS=0 python3 $T/primary/primary.py $O/inversions-F/pairs_scored.parquet $H/tmp/e36feat/full/selection.jsonl $T/primary/PRIMARY_F_map.json control,p30L,p30L-map100,p30L-map300,p30L-map1000 > $L/primary-map.log 2>&1 || echo FAIL primary-map
step "EVAL + error vs human"; m=""; for x in $CONT; do m="$m --model $(col $x)=$(bake $x)"; done
$LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m > $L/eval-score.log 2>&1 || echo FAIL eval-score
M=""; for s in 0 1 2; do M="$M,control@$s=control_s$s,p30L@$s=p30L_s$s"; done
for w in 100 300 1000; do M="$M,p30L-map$w@0=p30L_map${w}_s0"; done
ZEN_PANEL_BIN=$T/bin/panel $P $T/exp5/readout.py $T/eval ${M#,} $L/readout.json > $L/human_error.md 2>&1 || echo FAIL readout
step "G-CHROMA-L"; args="--bake control_s0=$T/control-s0/pack_e059/refit/production-f16.bin"; for x in $CONT; do args="$args --bake $(col $x)=$(bake $x)"; done
$LOCK --mem 12G --jobs 4 -- python3 scripts/byconstruction/chroma_ladder_gate.py score --bank /var/tmp/byc-fchroma-20261029/c/merged-bank --predict $PRED --roles dev $args --out $T/gchroma/F-dev --drop-long-edge-at-most 64 > $L/gchroma.log 2>&1 || echo FAIL gchroma
c=""; for x in $CONT; do c="$c --candidate $(col $x)"; done
python3 scripts/byconstruction/chroma_ladder_gate.py level --out $T/gchroma/F-dev --baseline control_s0 $c > $L/gchroma-level.log 2>&1 || echo FAIL gchroma-level
python3 scripts/byconstruction/chroma_ladder_gate.py report --out $T/gchroma/F-dev > $L/gchroma-report.log 2>&1 || echo FAIL gchroma-report
step "G-STEER"; : > $T/gates/models-F.txt; for x in $CONT; do echo "$(col $x) $(bake $x)" >> $T/gates/models-F.txt; done
while read n b; do $LOCK --mem 36G --jobs 9 -- python3 scripts/byconstruction/gates_steer_sharded.py --name $n --model $b --out $T/gates/$n --bin /var/tmp/byc-fx2-20261028/stage2/gate-tools-d02d2e07/steer_instrument --formula-rev 7 --shards 9 > $T/gates/$n.steer.log 2>&1 || echo FAIL steer $n
  (cd $H/work/zen/zensimx--fchroma && PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs python3 -B scripts/byconstruction/gates.py --summary --out $T/gates/$n --name $n > $T/gates/$n.summary.json 2>$T/gates/$n.summary.err) || echo FAIL summary $n; done < $T/gates/models-F.txt
step "done"
