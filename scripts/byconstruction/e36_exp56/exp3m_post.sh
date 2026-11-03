#!/bin/bash
# Experiment 3 step 3, seed 0: the map arms and the no-map reproduction of p30L, measured as amendment 2's arms.
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; A=$T/exp3m/runs; O=/mnt/v/zensim/e36-features/2026-11-01; L=$T/exp3m/post; mkdir -p $L
LOCK="flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy"
PRED=/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake
NEW="p30L-mapbuild-s0 p30L-map1-s0 p30L-map10-s0 p30L-map30-s0"
bake() { case $1 in control-s0) echo $T/control-s0/pack_e059/refit/production-f16.bin;; p30L-s0) echo $T/arms/p30L-s0/pack_e059/refit/production-f16.bin;; *) echo $A/$1/pack_e059/refit/production-f16.bin;; esac; }
col() { echo "$1" | tr '-' '_'; }
step() { echo "== $(date -u +%H:%M:%SZ) $1"; }
cd $H/work/zen/zensimx
step "p30L-mapbuild reproduces p30L-s0"; $T/speed/cmp_ckpts.sh $T/arms/p30L-s0/ckpt $A/p30L-mapbuild-s0/ckpt
step "pack"; (cd $T && $LOCK --mem 36G --jobs 8 -- bash -c "for n in $NEW; do ./pack_nolock.sh exp3m/runs/\$n > exp3m/runs/\$n.pack.log 2>&1 & done; wait") > $L/pack.log 2>&1
for n in $NEW; do tail -1 $A/$n.pack.log | grep -q PACKED || echo "PACK FAIL $n"; done
D=/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes; for n in $NEW; do mkdir -p $D/$n; cp --update=none $A/$n/pack_e059/refit/production-f16.bin $A/$n/shipped.bin $D/$n/; done
(cd $D && sha256sum */*.bin > SHA256SUMS && cp -r --update=none $D/. /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cp SHA256SUMS /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cd /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes && sha256sum -c --quiet SHA256SUMS && echo "bakes kept: $(wc -l < SHA256SUMS) files verified on tower")
step "held-out map agreement"; E=$T/exp3
for n in $NEW; do for b in 8 16; do $LOCK --mem 12G --jobs 8 -- $E/integrand_dump.bin --pairs $E/heldout/pairs.tsv --out $E/heldout/$n-$b --bake $(bake $n) --block $b > $E/heldout/$n-$b.jsonl 2>/dev/null
  echo "$n $b $(python3 $E/map_agreement.py $E/heldout/$n-$b.jsonl $E/heldout/$n-$b)"; done; done > $L/agreement.txt 2>&1
step "E36 score"; args=""; for n in control-s0 p30L-s0 $NEW; do args="$args --bake $(col $n)=$(bake $n)"; done
$LOCK --mem 36G --jobs 24 -- python3 scripts/byconstruction/e36_features.py score $O/bank $O/scores-map --predict $PRED $args > $L/score.log 2>&1 || echo FAIL score
step "inversions"; m=""; for n in control-s0 p30L-s0 $NEW; do m="$m --model $(col $n)"; done
$LOCK --mem 30G --jobs 8 -- python3 scripts/byconstruction/e36_features.py inversions $O/scores-map $H/tmp/e36feat/full /mnt/v/output/zensim/e36-bulk-2026-10-17/pools-full-v1 $O/inversions-map $m > $L/inversions.log 2>&1 || echo FAIL inversions
step "primary (seed 0)"; SEEDS=0 python3 $T/primary/primary.py $O/inversions-map/pairs_scored.parquet $H/tmp/e36feat/full/selection.jsonl $T/primary/PRIMARY_MAP.json control,p30L,p30L-mapbuild,p30L-map1,p30L-map10,p30L-map30 > $L/primary.log 2>&1 || echo FAIL primary
step "EVAL + error vs human"; S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; P=$H/.cache/zensim-training-env/py3147-20261031/bin/python; m=""; for n in $NEW; do m="$m --model $(col $n)=$(bake $n)"; done
$LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m > $L/eval-score.log 2>&1 || echo FAIL eval-score
M="control=control_s0,p30L=p30L_s0"; for n in $NEW; do M="$M,${n%-s0}=$(col $n)"; done
ZEN_PANEL_BIN=$T/bin/panel $P $T/exp5/readout.py $T/eval $M $L/readout.json > $L/human_error.md 2>&1 || echo FAIL human-error
step "G-CHROMA-L"; args=""; for n in control-s0 $NEW; do args="$args --bake $(col $n)=$(bake $n)"; done
$LOCK --mem 12G --jobs 4 -- python3 scripts/byconstruction/chroma_ladder_gate.py score --bank /var/tmp/byc-fchroma-20261029/c/merged-bank --predict $PRED --roles dev $args --out $T/gchroma/map-dev --drop-long-edge-at-most 64 > $L/gchroma.log 2>&1 || echo FAIL gchroma
c=""; for n in $NEW; do c="$c --candidate $(col $n)"; done
python3 scripts/byconstruction/chroma_ladder_gate.py level --out $T/gchroma/map-dev --baseline control_s0 $c > $L/gchroma-level.log 2>&1 || echo FAIL gchroma-level
python3 scripts/byconstruction/chroma_ladder_gate.py report --out $T/gchroma/map-dev > $L/gchroma-report.log 2>&1 || echo FAIL gchroma-report
step "G-STEER sharded, one model at a time"; : > $T/gates/models-map.txt; for n in $NEW; do echo "$(col $n) $(bake $n)" >> $T/gates/models-map.txt; done
while read n b; do $LOCK --mem 36G --jobs 9 -- python3 scripts/byconstruction/gates_steer_sharded.py --name $n --model $b --out $T/gates/$n --bin /var/tmp/byc-fx2-20261028/stage2/gate-tools-d02d2e07/steer_instrument --formula-rev 7 --shards 9 > $T/gates/$n.steer.log 2>&1 || echo FAIL steer $n
  (cd $H/work/zen/zensimx--fchroma && PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs python3 -B scripts/byconstruction/gates.py --summary --out $T/gates/$n --name $n > $T/gates/$n.summary.json 2>$T/gates/$n.summary.err) || echo FAIL summary $n; done < $T/gates/models-map.txt
step "done"
