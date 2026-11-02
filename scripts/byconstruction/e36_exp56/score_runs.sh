#!/bin/bash
# usage: score_runs.sh EXP:TAG ...  scores EXP/runs/TAG/refit/production-f16.bin on EVAL as model x<N>_<tag> (N from expN)
set -uo pipefail
T=~/tmp/e36test; S=/var/tmp/byc-fchroma-20261029/r3/stage2/fx2-full-general-e59; P=~/.cache/zensim-training-env/py3147-20261031/bin/python
m=""; for x in "$@"; do e=${x%%:*}; n=${x#*:}; b=$T/$e/runs/$n/refit/production-f16.bin; test -f $b || { echo "no bake $x"; exit 1; }
  m="$m --model x${e#exp}_$(echo $n | tr -- '-+' '__')=$b"; done
cd ~/work/zen/zensimx
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 12G --jobs 8 -- $P scripts/byconstruction/dataset_eval.py score --out $T/eval --predict $S/bin/bake_dial_refit $m
