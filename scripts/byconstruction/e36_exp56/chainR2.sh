#!/bin/bash
# R2 (method review M3) after amendment 7's post: six walk base fits (4 + 2 per wave), then postR2.
cd ~/tmp/e36test
until [ -e postK.DONE ]; do sleep 60; done
./wave.sh exp7:m3c-s0 exp7:m3cn-s0 exp7:m3c-s1 exp7:m3cn-s1 > waveR2-0.log 2>&1
./wave.sh exp7:m3c-s2 exp7:m3cn-s2 > waveR2-1.log 2>&1
echo "waves done $(date -u +%FT%TZ)" > waveR2.DONE
./postR2.sh > postR2.log 2>&1; echo "done $(date -u +%FT%TZ)" > postR2.DONE
