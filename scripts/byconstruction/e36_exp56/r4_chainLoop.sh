#!/bin/bash
# R4: build and calibrate on dev, then encode and judge on i270.
cd ~/work/zen/zensimx
S=$HOME/tmp/e36test/r4loop
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 16 -- bash scripts/byconstruction/e36_loop_r4.sh build $S > $S/stages.log 2>&1; echo "build rc=$?" >> $S/stages.log
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 16 -- bash scripts/byconstruction/e36_loop_r4.sh calibrate $S >> $S/stages.log 2>&1; echo "calibrate rc=$?" >> $S/stages.log
grep -q "calibrate rc=0" $S/stages.log || exit 1
bash scripts/byconstruction/e36_exp56/r4_loop_i270_run.sh > $S/i270_launch.log 2>&1; echo "launch rc=$?" >> $S/stages.log
until ssh i270 "test -e $S/I270_DONE"; do sleep 60; done
for f in run-i270.log bdrate-vs-butteraugli.md bdrate-vs-p30Llintail.md bdrate-vs-map300lintail.md bdrate-vs-F3lintail.md REPRODUCTION.txt LINEAR_ARMS.txt; do scp -q "i270:tmp/e36test/r4loop/$f" $S/; done && echo "fetched $(date -u +%FT%TZ)" >> $S/stages.log
