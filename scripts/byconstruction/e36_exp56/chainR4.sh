#!/bin/bash
# R4 (DVIFM's painted map as the map-term target, option b1): 10 map-term runs (3 full-data continuations + 7 AIC folds), at most
# 4 per wave (each map arm peaks near 4 GB while loading its pool), then postR4.   chainR4.sh PRESET
cd ~/tmp/e36test
P=${1:?preset}
L=($(python3 r4/make_cmds.py $P r4/pools | tr '\n' ' '))
printf '%s\n' "${L[@]}" > r4/tags.txt
i=0; while [ $i -lt ${#L[@]} ]; do ./wave.sh "${L[@]:$i:4}" > waveR4-$i.log 2>&1 || { echo "wave $i failed" > chainR4.FAILED; exit 1; }; i=$((i+4)); done
echo "done $(date -u +%FT%TZ)" > waveR4.DONE
PRESET=$P ./postR4.sh > postR4.log 2>&1
echo "done $(date -u +%FT%TZ)" > postR4.DONE
