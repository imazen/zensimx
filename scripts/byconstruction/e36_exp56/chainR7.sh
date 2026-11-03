#!/bin/bash
# R7: six base fits (waves of 4), then 20 continuations (6 full-data + 14 AIC folds, waves of 4), then postR7.
cd ~/tmp/e36test
B=($(sed -n 1p exp9/tags.txt)); C=($(sed -n 2p exp9/tags.txt))
i=0; while [ $i -lt ${#B[@]} ]; do ./wave.sh "${B[@]:$i:4}" > waveR7b-$i.log 2>&1 || { echo "base wave $i failed" > chainR7.FAILED; exit 1; }; i=$((i+4)); done
for t in ${B[@]}; do test -s ${t%%:*}/runs/${t#*:}/refit/last.bin || { echo "no base checkpoint $t" > chainR7.FAILED; exit 1; }; done
i=0; while [ $i -lt ${#C[@]} ]; do ./wave.sh "${C[@]:$i:4}" > waveR7c-$i.log 2>&1; i=$((i+4)); done
echo "waves done $(date -u +%FT%TZ)" > waveR7.DONE
./postR7.sh > postR7.log 2>&1; echo "done $(date -u +%FT%TZ)" > postR7.DONE
