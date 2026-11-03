#!/bin/bash
# R10 amendment 1: build and calibrate on dev, then encode and judge on i270.
cd ~/work/zen/zensimx
S=$HOME/tmp/e36test/r10loop
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 16 -- bash scripts/byconstruction/e36_loop_r10.sh build $S > $S/stages.log 2>&1; echo "build rc=$?" >> $S/stages.log
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 16 -- bash scripts/byconstruction/e36_loop_r10.sh calibrate $S >> $S/stages.log 2>&1; echo "calibrate rc=$?" >> $S/stages.log
grep -q "calibrate rc=0" $S/stages.log || exit 1
$S/i270_run.sh > $S/i270_launch.log 2>&1; echo "launch rc=$?" >> $S/stages.log
until ssh i270 "test -e $S/I270_DONE"; do sleep 60; done
for f in run-i270.log bdrate-vs-F3lintail.md bdrate-vs-butteraugli.md bdrate-vs-p30Llintail.md REPRODUCTION.txt LINEAR_ARMS.txt; do scp -q "i270:tmp/e36test/r10loop/$f" $S/; done && echo "fetched $(date -u +%FT%TZ)" >> $S/stages.log
