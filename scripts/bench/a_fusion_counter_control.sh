#!/usr/bin/env bash
# Verify that tool duration and FIFO-controlled hardware time have the same scope.
set -euo pipefail
[[ $(hostname) = dev ]]
study=${1:?study root}
out=${2:?new receipt directory}
if [[ ${A_FUSION_CONTROL_LOCKED:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G --jobs 1 -- \
        env A_FUSION_CONTROL_LOCKED=1 bash "$0" "$@"
fi
test ! -e "$out"
mkdir -p "$out"
mkfifo "$out/control"
export TMPDIR="$study/scratch"
taskset -c 0 perf stat -x, -e duration_time,task-clock,cycles --delay=-1 \
    --control="fifo:$out/control" -o "$out/counters.csv" -- \
    python3 - "$out/control" > "$out/run.log" 2>&1 <<'PY'
from pathlib import Path
import sys,time
fifo=Path(sys.argv[1])
time.sleep(1)
with fifo.open('w') as stream:
    stream.write('enable\n');stream.flush()
    sum(range(10_000_000))
    stream.write('disable\n');stream.flush()
time.sleep(1)
PY
python3 - "$out" <<'PY'
import csv,json,pathlib,sys
out=pathlib.Path(sys.argv[1]);values={}
with (out/'counters.csv').open() as stream:
    for row in csv.reader(line for line in stream if not line.startswith('#')):
        if len(row)>=7:
            assert not row[0].startswith('<'),row
            assert float(row[-3])>=99,row
            values[row[2]]=dict(count=float(row[0]),runtime_ns=float(row[-4]))
assert values['cycles']['runtime_ns']>0
result=dict(counters=values,tool_duration_ns=values['duration_time']['count'],
            hardware_runtime_ns=values['cycles']['runtime_ns'])
result['same_window']=result['tool_duration_ns']<2*result['hardware_runtime_ns']
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
