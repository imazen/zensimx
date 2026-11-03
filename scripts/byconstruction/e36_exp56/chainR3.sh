#!/bin/bash
# R3 after R2's post: 20 map-term runs (exp8 full data seeds 0-2 + AIC folds), at most 4 per wave, then postR3.
cd ~/tmp/e36test
until [ -e postR2.DONE ]; do sleep 60; done
L=($(cat exp8/tags.txt))
i=0; while [ $i -lt ${#L[@]} ]; do ./wave.sh "${L[@]:$i:4}" > waveR3-$i.log 2>&1; i=$((i+4)); done
echo "waves done $(date -u +%FT%TZ)" > waveR3.DONE
./postR3.sh > postR3.log 2>&1; echo "done $(date -u +%FT%TZ)" > postR3.DONE
