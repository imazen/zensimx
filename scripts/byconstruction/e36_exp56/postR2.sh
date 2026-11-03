#!/bin/bash
# After R2's base fits (method review M3: human rows by distortion type): EVAL, error against human ratings, the M1
# paired panel per seed, and the E36 primary, each against experiment 6's walk arm (same sampler, all human rows).
set -uo pipefail
H=$HOME; T=$H/tmp/e36test; O=/mnt/v/zensim/e36-features/2026-11-01; L=$T/postR2; mkdir -p $L
LOCK="flock $H/tmp/heavy.lock $H/work/claudehints/scripts/run-heavy"
PRED=/var/tmp/byc-fchroma-20261029/r3/tools/predict_features_with_bake
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; P=$H/.cache/zensim-training-env/py3147-20261031/bin/python
ARMS="m3c m3cn"
step() { echo "== $(date -u +%H:%M:%SZ) $1"; }
cd $H/work/zen/zensimx
for a in $ARMS; do for s in 0 1 2; do test -s $T/exp7/runs/$a-s$s/refit/production-f16.bin || echo "MISSING BAKE $a-s$s"; done; done
step "keep bakes"; D=/mnt/v/output/zensim/e36-subset-test-2026-11-01/bakes
for a in $ARMS; do for s in 0 1 2; do mkdir -p $D/r2-$a-s$s; cp --update=none $T/exp7/runs/$a-s$s/refit/production-f16.bin $D/r2-$a-s$s/; done; done
(cd $D && sha256sum */*.bin > SHA256SUMS && cp -r --update=none $D/. /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cp SHA256SUMS /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes/ && cd /mnt/tower/output/zensim/e36-subset-test-2026-11-01/bakes && sha256sum -c --quiet SHA256SUMS && echo "bakes kept: $(wc -l < SHA256SUMS) files verified on tower")
grep -h "\[coverage\]" $T/exp7/runs/*/train.log $T/exp6/runs/walk-s*/train.log > $L/coverage.txt
step "EVAL"; m=""; for a in $ARMS; do for s in 0 1 2; do m="$m --model x7_${a}_s$s=$T/exp7/runs/$a-s$s/refit/production-f16.bin"; done; done
$LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m > $L/eval-score.log 2>&1 || echo FAIL eval-score
M=""; for s in 0 1 2; do M="$M,walk@$s=x6_walk_s$s,m3c@$s=x7_m3c_s$s,m3cn@$s=x7_m3cn_s$s"; done
ZEN_PANEL_BIN=$T/bin/panel $P $T/exp5/readout.py $T/eval ${M#,} $L/readout.json > $L/human_error.md 2>&1 || echo FAIL readout
step "compression rows (R2 amendment 1)"; C=""; for s in 0 1 2; do C="$C,walk@$s=x6_walk_s$s,m3c@$s=x7_m3c_s$s,m3cn@$s=x7_m3cn_s$s"; done
ZEN_PANEL_BIN=$T/bin/panel $P scripts/byconstruction/e36_exp56/ceval.py $T/eval $L/ceval.json ${C#,} > $L/ceval.md 2>&1 || echo FAIL ceval
step "M1 paired panel per seed"
for s in 0 1 2; do for a in $ARMS; do
  ZEN_PANEL_BIN=$T/bin/panel RAYON_NUM_THREADS=8 $LOCK --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py paired --out $T/eval --dest $L/paired-$a-s$s --a x7_${a}_s$s --vs x6_walk_s$s > $L/paired-$a-s$s.log 2>&1 || echo FAIL paired $a $s
done; done
step "E36 score"; args=""
for s in 0 1 2; do args="$args --bake walk_s$s=$T/exp6/runs/walk-s$s/refit/production-f16.bin"; for a in $ARMS; do args="$args --bake ${a}_s$s=$T/exp7/runs/$a-s$s/refit/production-f16.bin"; done; done
$LOCK --mem 36G --jobs 24 -- python3 scripts/byconstruction/e36_features.py score $O/bank $O/scores-R2 --predict $PRED $args > $L/score.log 2>&1 || echo FAIL score
m=""; for s in 0 1 2; do m="$m --model walk_s$s"; for a in $ARMS; do m="$m --model ${a}_s$s"; done; done
$LOCK --mem 30G --jobs 8 -- python3 scripts/byconstruction/e36_features.py inversions $O/scores-R2 $H/tmp/e36feat/full /mnt/v/output/zensim/e36-bulk-2026-10-17/pools-full-v1 $O/inversions-R2 $m > $L/inversions.log 2>&1 || echo FAIL inversions
CONTROL=walk python3 $T/primary/primary.py $O/inversions-R2/pairs_scored.parquet $H/tmp/e36feat/full/selection.jsonl $T/primary/PRIMARY_R2.json walk,m3c,m3cn > $L/primary.log 2>&1 || echo FAIL primary
python3 scripts/byconstruction/e36_exp56/m1_summary.py "m3c minus walk=$L/paired-m3c-s0,$L/paired-m3c-s1,$L/paired-m3c-s2" "m3cn minus walk=$L/paired-m3cn-s0,$L/paired-m3cn-s1,$L/paired-m3cn-s2" > $L/m1.md 2>&1 || echo FAIL m1-summary
step "done"
