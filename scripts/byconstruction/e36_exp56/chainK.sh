#!/bin/bash
# Experiment 5 amendment 7: mix + map 30 / 100, full data at seeds 0-2 and the AIC2026 folds; at most 4 map arms per wave.
cd ~/tmp/e36test
L=(exp5c:tid2013_mdid-p30L-map30-s0 exp5c:tid2013_mdid-p30L-map30-s1 exp5c:tid2013_mdid-p30L-map30-s2 exp5c:tid2013_mdid-p30L-map100-s0
   exp5c:tid2013_mdid-p30L-map100-s1 exp5c:tid2013_mdid-p30L-map100-s2)
for k in 0 1 2 3 4 5 6; do L+=(expaic:mixmap30-k$k expaic:mixmap100-k$k); done
i=0; while [ $i -lt ${#L[@]} ]; do ./wave.sh "${L[@]:$i:4}" > waveK-$i.log 2>&1; i=$((i+4)); done
echo "done $(date -u +%FT%TZ)" > waveK.DONE
