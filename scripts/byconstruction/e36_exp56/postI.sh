#!/bin/bash
# After wave I: p30L + map 300 / 1000 at seeds 0-2 (amendment 4), with the amendment-2 measures and NEARID.
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; O=/mnt/v/zensim/e36-features/2026-11-01; L=$T/postI; mkdir -p $L
LOCK="flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy"
PRED=/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; P=$H/.cache/zensim-training-env/py3147-20261031/bin/python
NEW="p30L-map300-s1 p30L-map1000-s1 p30L-map300-s2 p30L-map1000-s2"
ALL="p30L-map300-s0 p30L-map1000-s0 $NEW"
bake() { echo $T/exp3m/runs/$1/pack_e059/refit/production-f16.bin; }
col() { echo "$1" | tr -- '-+' '__'; }
step() { echo "== $(date -u +%H:%M:%SZ) $1"; }
cd $H/work/zen/zensimx
step "pack"; (cd $T && $LOCK --mem 36G --jobs 8 -- bash -c "for n in $NEW; do ./pack_nolock.sh exp3m/runs/\$n > exp3m/runs/\$n.pack.log 2>&1 & done; wait") > $L/pack.log 2>&1
for n in $NEW; do tail -1 $T/exp3m/runs/$n.pack.log | grep -q PACKED || echo "PACK FAIL $n"; done
D=/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes; for n in $NEW; do mkdir -p $D/$n; cp --update=none $(bake $n) $T/exp3m/runs/$n/shipped.bin $D/$n/; done
(cd $D && sha256sum */*.bin > SHA256SUMS && cp -r --update=none $D/. /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cp SHA256SUMS /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cd /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes && sha256sum -c --quiet SHA256SUMS && echo "bakes kept: $(wc -l < SHA256SUMS) files verified on tower")
step "held-out map agreement"; E=$T/exp3
for n in $NEW; do for b in 8 16; do $LOCK --mem 12G --jobs 8 -- $E/integrand_dump.bin --pairs $E/heldout/pairs.tsv --out $E/heldout/$n-$b --bake $(bake $n) --block $b > $E/heldout/$n-$b.jsonl 2>/dev/null
  echo "$n $b $(python3 $E/map_agreement.py $E/heldout/$n-$b.jsonl $E/heldout/$n-$b)"; done; done > $L/agreement.txt 2>&1
step "E36 score, seeds 0-2"; args=""
for s in 0 1 2; do args="$args --bake control_s$s=$( [ $s = 0 ] && echo $T/control-s0 || echo $T/arms/control-s$s)/pack_e059/refit/production-f16.bin --bake p30L_s$s=$T/arms/p30L-s$s/pack_e059/refit/production-f16.bin"; done
for n in $ALL; do args="$args --bake $(col $n)=$(bake $n)"; done
$LOCK --mem 36G --jobs 24 -- python3 scripts/byconstruction/e36_features.py score $O/bank $O/scores-I --predict $PRED $args > $L/score.log 2>&1 || echo FAIL score
m=""; for s in 0 1 2; do m="$m --model control_s$s --model p30L_s$s"; done; for n in $ALL; do m="$m --model $(col $n)"; done
$LOCK --mem 30G --jobs 8 -- python3 scripts/byconstruction/e36_features.py inversions $O/scores-I $H/tmp/e36feat/full /mnt/v/output/zensim/e36-bulk-2026-10-17/pools-full-v1 $O/inversions-I $m > $L/inversions.log 2>&1 || echo FAIL inversions
python3 $T/primary/primary.py $O/inversions-I/pairs_scored.parquet $H/tmp/e36feat/full/selection.jsonl $T/primary/PRIMARY_I.json control,p30L,p30L-map300,p30L-map1000 > $L/primary.log 2>&1 || echo FAIL primary
step "EVAL + error vs human"; m=""; for n in $NEW; do m="$m --model $(col $n)=$(bake $n)"; done
$LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m > $L/eval-score.log 2>&1 || echo FAIL eval-score
M=""; for s in 0 1 2; do M="$M,control@$s=control_s$s,p30L@$s=p30L_s$s,p30L-map300@$s=p30L_map300_s$s,p30L-map1000@$s=p30L_map1000_s$s"; done
ZEN_PANEL_BIN=$T/bin/panel $P $T/exp5/readout.py $T/eval ${M#,} $L/readout.json > $L/human_error.md 2>&1 || echo FAIL readout
step "G-CHROMA-L"; args="--bake control_s0=$T/control-s0/pack_e059/refit/production-f16.bin"; for n in $NEW; do args="$args --bake $(col $n)=$(bake $n)"; done
$LOCK --mem 12G --jobs 4 -- python3 scripts/byconstruction/chroma_ladder_gate.py score --bank /var/tmp/byc-fchroma-20261029/c/merged-bank --predict $PRED --roles dev $args --out $T/gchroma/I-dev --drop-long-edge-at-most 64 > $L/gchroma.log 2>&1 || echo FAIL gchroma
python3 scripts/byconstruction/chroma_ladder_gate.py report --out $T/gchroma/I-dev > $L/gchroma-report.log 2>&1 || echo FAIL gchroma-report
step "G-STEER"; : > $T/gates/models-I.txt; for n in $NEW; do echo "$(col $n) $(bake $n)" >> $T/gates/models-I.txt; done
while read n b; do $LOCK --mem 36G --jobs 9 -- python3 scripts/byconstruction/gates_steer_sharded.py --name $n --model $b --out $T/gates/$n --bin /var/tmp/byc-fx2-20261028/stage2/gate-tools-d02d2e07/steer_instrument --formula-rev 7 --shards 9 > $T/gates/$n.steer.log 2>&1 || echo FAIL steer $n
  (cd $H/work/zen/zensimx--fchroma && PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs python3 -B scripts/byconstruction/gates.py --summary --out $T/gates/$n --name $n > $T/gates/$n.summary.json 2>$T/gates/$n.summary.err) || echo FAIL summary $n; done < $T/gates/models-I.txt
step "NEARID (seed 0)"; R=/mnt/tower/output/zensim/mlic-nearid-20261027-run01; B=$T/nearid/bin/serve_custom_bake; : > $T/nearid/jobs-I.txt
for n in p30L-map300-s0 p30L-map1000-s0; do b=$(bake $n); echo "$(echo $n | tr -d - | tr A-Z a-z) $b $(sha256sum < $b | cut -c1-64)" >> $T/nearid/jobs-I.txt; done
$LOCK --mem 24G --jobs 24 -- bash -c 'while read l b h; do just dataset-nearid-candidate '$B' '$R'/PACKET.json ~/tmp/e36test/nearid/out-$l $l $b $h > ~/tmp/e36test/nearid/$l.log 2>&1 & done < ~/tmp/e36test/nearid/jobs-I.txt; wait' > $L/nearid.log 2>&1 || echo FAIL nearid
step "done"
