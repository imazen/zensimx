#!/usr/bin/env bash
# Parse saved captures only. This command never launches a scorer or benchmark.
set -euo pipefail
study=${1:?study root}
stage=${2:?perf|ibs|timing|eval-baseline|amd-report|intel-report}
input=${3:?saved capture directory}
out=${4:?new parse-attempt directory}
case $(hostname) in dev) memory=24G ;; i270) memory=16G ;; *) exit 2 ;; esac
if [[ ${A_FUSION_PARSE_LOCKED:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem "$memory" --jobs 1 -- \
        env A_FUSION_PARSE_LOCKED=1 bash "$0" "$@"
fi
test ! -e "$out"
mkdir -p "$out"
export TMPDIR="$study/scratch"
tools=$(cd "$(dirname "$0")" && pwd)
finish() {
    local status=$?
    python3 - "$input" "$out" "$stage" "$status" <<'PY'
import datetime,json,pathlib,sys
source,out=map(pathlib.Path,sys.argv[1:3]);stage=sys.argv[3];status=int(sys.argv[4])
row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,
         input=str(source.resolve()),output=str(out.resolve()),status=status,
         artifacts=[p.name for p in out.iterdir() if p.name in
                    {'samples.txt','self.txt','load-data.txt','memory-symbols.txt','summary.json','report.json'}])
(out/'STATUS.json').write_text(json.dumps(row,indent=2)+'\n')
with (source/'PARSE_RUNS.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
PY
}
trap finish EXIT
exec > "$out/parse.log" 2>&1
sha256sum "$0" "$tools/a_fusion_artifacts.py" > "$out/PARSER_SHA256SUMS"
case "$stage" in
    perf|ibs)
        python3 - "$input/admission.json" <<'PY'
import json,pathlib,sys
row=json.loads(pathlib.Path(sys.argv[1]).read_text());assert row['status']=='accepted',row
PY
        sha256sum "$input/perf.data" "$input/admission.json" > "$out/INPUT_SHA256SUMS"
        perf script -i "$input/perf.data" --no-inline -F period,ip,sym > "$out/samples.txt"
        perf report -i "$input/perf.data" --stdio --no-inline --call-graph none --no-children \
            --percent-limit 0 -F overhead,period,sample,symbol > "$out/self.txt"
        if [[ $stage = ibs ]]; then
            perf mem report -i "$input/perf.data" --stdio --sort=mem,sym > "$out/memory-symbols.txt"
            perf script -i "$input/perf.data" -F comm,pid,tid,cpu,event,ip,sym,addr,weight,data_src > "$out/load-data.txt"
        fi
        ;;
    timing)
        sha256sum "$input/inner.json" "$input/workers.json" "$input/admission.json" > "$out/INPUT_SHA256SUMS"
        sha256sum "$tools/a_extraction_fusion.py" >> "$out/PARSER_SHA256SUMS"
        python3 -B "$tools/a_extraction_fusion.py" timing "$input" "$out/summary.json"
        ;;
    eval-baseline)
        sha256sum "$input/before3/eval.jsonl" "$input/fixtures/eval-frozen-pred-mixF3-s0.tsv" > "$out/INPUT_SHA256SUMS"
        sha256sum "$tools/a_extraction_fusion.py" >> "$out/PARSER_SHA256SUMS"
        python3 -B "$tools/a_extraction_fusion.py" eval-baseline "$input" "$out/report.json"
        ;;
    amd-report)
        sha256sum "$tools/a_fusion_amd_report.py" "$tools/a_family_cost_perf.py" >> "$out/PARSER_SHA256SUMS"
        python3 -B "$tools/a_fusion_amd_report.py" "$input" "$out/report.json"
        ;;
    intel-report)
        sha256sum "$tools/a_fusion_stall_report.py" "$tools/a_family_cost_perf.py" >> "$out/PARSER_SHA256SUMS"
        python3 -B "$tools/a_fusion_stall_report.py" "$input" "$out/report.json"
        ;;
    *) echo 'unsupported parse stage' >&2; exit 2 ;;
esac
