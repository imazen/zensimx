#!/bin/bash
# usage: wave.sh EXP:TAG ...  runs ~/tmp/e36test/EXP/cmds/TAG.cmd.json concurrently, staggered 15 s, under one heavy lock
cd ~/tmp/e36test
for x in "$@"; do e=${x%%:*}; n=${x#*:}; test -e $e/cmds/$n.cmd.json || { echo "missing $x"; exit 1; }
  test -e $e/runs/$n && { echo "exists $e/runs/$n"; exit 1; }; mkdir -p $e/logs; done
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 40G --jobs $# -- bash -c '
  for x in "$@"; do e=${x%%:*}; n=${x#*:}; (cd $e && exec python3 ../run_cmd.py cmds/$n.cmd.json > logs/$n.log 2>&1) & sleep 15; done
  rc=0; for x in "$@"; do wait -n || rc=1; done; exit $rc' _ "$@"
rc=$?; for x in "$@"; do e=${x%%:*}; n=${x#*:}; echo "$x: $(tail -1 $e/logs/$n.log | cut -c1-160)"; done; exit $rc
