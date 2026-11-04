#!/bin/bash
# R10 amendment 5: score the four confirmation pair lists with jnd-ladder-score (zenmetrics-private
# experiments/jnd-ladder: SSIMULACRA2, butteraugli, PSNR-Y, FSIM/FSIMc, HaarPSI, the two published AIC-4 Haar configs;
# binary built on i265, weights /mnt/v/zenmetrics-private/aic4/models/ecec4e0), write judge files beside the
# candidates with refmetric_judges.py, rerun the confirm statistics and the EVAL readouts. Pair lists: `id ref dist`
# from each set's keys.parquet (EVAL: its judge_pairs.tsv), under $HOME/tmp/e36test/refmetrics/pairs/.
set -uo pipefail
R=$HOME/tmp/e36test/refmetrics; T=$HOME/tmp/e36test; X=$HOME/work/zen/zensimx/scripts/byconstruction/e36_exp56
LOCK="flock $HOME/tmp/heavy.lock $HOME/work/claudehints/scripts/run-heavy"
export AIC4_MODEL_DIR=/mnt/v/zenmetrics-private/aic4/models/ecec4e0
mkdir -p $R/scores $R/logs
# Memory: the AIC-4 Haar pyramids take ~14 GB per pair per thread on EVAL's 8K IQA-ODI JPEGs (eval-odi.tsv: 48 of
# EVAL's 4,678 pairs), so IQA-ODI is scored one pair and one config at a time under a 20G cap (peak RSS 11.2 GiB),
# the other EVAL pairs at 4 threads under 24G, and the three held-out sets at 16 threads under 28G (their images are
# small). dev's cap never exceeds 24G unless the box is verified idle; a 44G cap let the whole machine run out of memory
# on 2026-11-03 before the cap fired. The scorer resumes by id; refmetric_judges.py merges the per-config files.
for s in mmspg clic21 ciqa; do
  echo "== $s $(date -u +%T)"
  $LOCK --mem 28G --jobs 16 -- $HOME/tmp/refmetrics-bin/jnd-ladder-score --threads 16 \
    --configs HaarXYB-SSIM-deltaL1248,HaarXYB-SSIM-mask-deltaL1248 $R/pairs/$s.tsv $R/scores/$s.tsv > $R/logs/score-$s.log 2>&1 || echo "FAIL score $s"
done
echo "== eval (without IQA-ODI) $(date -u +%T)"
grep -v '^iqa_odi' $R/pairs/eval.tsv > $R/pairs/eval-small.tsv
{ head -1 $R/pairs/eval.tsv; grep '^iqa_odi' $R/pairs/eval.tsv; } > $R/pairs/eval-odi.tsv
$LOCK --mem 24G --jobs 4 -- $HOME/tmp/refmetrics-bin/jnd-ladder-score --threads 4 \
  --configs HaarXYB-SSIM-deltaL1248,HaarXYB-SSIM-mask-deltaL1248 $R/pairs/eval-small.tsv $R/scores/eval.tsv > $R/logs/score-eval-small.log 2>&1 || echo "FAIL score eval-small"
for cfg in HaarXYB-SSIM-deltaL1248 HaarXYB-SSIM-mask-deltaL1248; do
  echo "== eval IQA-ODI $cfg $(date -u +%T)"
  $LOCK --mem 20G --jobs 1 -- $HOME/tmp/refmetrics-bin/jnd-ladder-score --threads 1 --configs $cfg \
    $R/pairs/eval-odi.tsv $R/scores/eval-odi-$cfg.tsv > $R/logs/score-eval-odi-$cfg.log 2>&1 || echo "FAIL score eval-odi $cfg"
done
S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59
cd $HOME/work/zen/zensimx
python3 $X/refmetric_judges.py $T/eval $R/pairs/eval.tsv --haar $R/scores/eval.tsv \
  --haar $R/scores/eval-odi-HaarXYB-SSIM-deltaL1248.tsv --haar $R/scores/eval-odi-HaarXYB-SSIM-mask-deltaL1248.tsv \
  --keep-ssim2 > $R/logs/judges-eval.log 2>&1 || echo "FAIL judges eval"
python3 $X/refmetric_judges.py $T/clic21-confirm-3 $R/pairs/clic21.tsv --haar $R/scores/clic21.tsv > $R/logs/judges-clic21.log 2>&1 || echo "FAIL judges clic21"
python3 $X/refmetric_judges.py /mnt/v/zensim/e36-read4-mmspg/confirm-1 $R/pairs/mmspg.tsv --haar $R/scores/mmspg.tsv > $R/logs/judges-mmspg.log 2>&1 || echo "FAIL judges mmspg"
python3 $X/refmetric_judges.py /mnt/v/zensim/ciqa-private/e36-read/confirm-1 $R/pairs/ciqa.tsv --haar $R/scores/ciqa.tsv > $R/logs/judges-ciqa.log 2>&1 || echo "FAIL judges ciqa"
m=""; for s in 0 1 2; do c=$( [ $s = 0 ] && echo $T/control-s0 || echo $T/arms/control-s$s); m="$m --model control_s$s=$c/pack_e059/refit/production-f16.bin --model mixF3_s$s=$T/r10/runs/mixF3-s$s/pack_e059/refit/production-f16.bin --model mix_p30L_s$s=$T/exp5c/runs/tid2013_mdid-p30L-s$s/pack_e059/refit/production-f16.bin"; done
ZEN_PANEL_BIN=$T/bin/panel python3 $X/clic.py confirm $T/clic21-confirm-3 --bank /mnt/v/zensim/dataset-features-rev7-clic-20261103/bank/clic21_valid --predict $S/bin/bake_dial_refit $m --vs control_s@ --registration "confirmation read 3 (R10 amendment 2)" --arm "mix on F3's recipe=mixF3_s0@0,mixF3_s1@1,mixF3_s2@2" --arm "mix + p30L=mix_p30L_s0@0,mix_p30L_s1@1,mix_p30L_s2@2" > $R/logs/confirm-clic21.log 2>&1 || echo "FAIL confirm clic21"
m2="$m"; for s in 0 1 2; do m2="$m2 --model p30L_s$s=$T/arms/p30L-s$s/pack_e059/refit/production-f16.bin"; done
ZEN_PANEL_BIN=$T/bin/panel python3 $X/ciqa.py confirm /mnt/v/zensim/ciqa-private/e36-read/confirm-1 --bank /mnt/v/zensim/ciqa-private/e36-read/bank/ciqa --predict $S/bin/bake_dial_refit $m2 --vs control_s@ --registration "R10 amendment 3 (CIQA confirmation read)" --arm "p30L=p30L_s0@0,p30L_s1@1,p30L_s2@2" --arm "mix on F3's recipe=mixF3_s0@0,mixF3_s1@1,mixF3_s2@2" --arm "mix + p30L=mix_p30L_s0@0,mix_p30L_s1@1,mix_p30L_s2@2" > $R/logs/confirm-ciqa.log 2>&1 || echo "FAIL confirm ciqa"
ZEN_PANEL_BIN=$T/bin/panel python3 $X/mos_confirm.py confirm /mnt/v/zensim/e36-read4-mmspg/confirm-1 --set mmspg --labels /mnt/tower/input/iqa/datasets/mmspg-jpeg-xr/labels.tsv --bank /mnt/v/zensim/e36-read4-mmspg/bank/mmspg --predict $S/bin/bake_dial_refit $m2 --vs control_s@ --registration "R10 amendment 4 (MMSPG JPEG XR confirmation read)" --arm "p30L=p30L_s0@0,p30L_s1@1,p30L_s2@2" --arm "mix on F3's recipe=mixF3_s0@0,mixF3_s1@1,mixF3_s2@2" --arm "mix + p30L=mix_p30L_s0@0,mix_p30L_s1@1,mix_p30L_s2@2" > $R/logs/confirm-mmspg.log 2>&1 || echo "FAIL confirm mmspg"
M="F3@0=control_s0,F3@1=control_s1,F3@2=control_s2,mix@0=mixF3_s0,mix@1=mixF3_s1,mix@2=mixF3_s2,ssim2=ssim2,butteraugli=butteraugli,cvvdp=cvvdp,fsimc=fsimc,fsim_y=fsim_y,haarpsi=haarpsi,haar_ssim=haar_ssim,haar_mask=haar_mask"
ZEN_PANEL_BIN=$T/bin/panel python3 $T/exp5/readout.py $T/eval "$M" $R/eval-readout.json > $R/eval-human_error.md 2>&1 || echo "FAIL readout"
ZEN_PANEL_BIN=$T/bin/panel python3 $X/ceval.py $T/eval $R/eval-ceval.json "$M" > $R/eval-ceval.md 2>&1 || echo "FAIL ceval"
echo "ALL DONE $(date -u +%T)"
