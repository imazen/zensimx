#!/bin/bash
# R10 (the TID2013 + MDID mix on F3's continuation, no hinge): 3 continuations + 7 AIC2026 folds, waves of 7, then postR10.
cd ~/tmp/e36test
L=($(cat r10/tags.txt | tr '\n' ' '))
i=0; while [ $i -lt ${#L[@]} ]; do ./wave.sh "${L[@]:$i:7}" > waveR10-$i.log 2>&1 || { echo "wave $i failed" > chainR10.FAILED; exit 1; }; i=$((i+7)); done
echo "done $(date -u +%FT%TZ)" > waveR10.DONE
./postR10.sh > postR10.log 2>&1
echo "done $(date -u +%FT%TZ)" > postR10.DONE
