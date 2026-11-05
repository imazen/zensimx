#!/usr/bin/env bash
# One quiet, FIFO-controlled AMD profile window per shared-lock acquisition.
set -euo pipefail
[[ $(hostname) = dev ]] || { echo 'AMD fusion profiles require dev' >&2; exit 2; }
study=${1:?study root}
label=${2:?immutable label}
size=${3:?256|1024|large}
kind=${4:?cycles|counters|pipeline1|pipeline2|l2|l3|memory|fills|ibs|metric-NAME}
[[ $size =~ ^(256|1024|large)$ ]]
if [[ $kind = metric-umc_* && -f $study/STOP_OPTIONAL_PROFILES.json ]]; then
    cat "$study/STOP_OPTIONAL_PROFILES.json" >&2
    echo 'Optional capture cancelled before launch; required profiles and timings retain priority.' >&2
    exit 76
fi
if [[ ${A_FUSION_AMD_LOCK_HELD:-0} != 1 ]]; then
    test -f "$study/A_REPORTED.json"
    while true; do
        status=0
        flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G --jobs 2 -- \
            env A_FUSION_AMD_LOCK_HELD=1 bash "$0" "$@" || status=$?
        [[ $status = 75 ]] || exit "$status"
        sleep 15
    done
fi
if [[ ${A_FUSION_AMD_QUIET:-0} != 1 ]]; then
    exec python3 -B "$study/source/scripts/bench/a_fusion_quiet.py" \
        "$study/$label/quiet-$size-$kind.jsonl" -- \
        env A_FUSION_AMD_QUIET=1 bash "$0" "$@"
fi
export TMPDIR="$study/scratch" ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1
binary="$study/${A_FUSION_BEFORE_LABEL:?baseline label}/a_family_cost"
cell="$study/$label/$size-$kind"
test ! -e "$cell"
mkdir -p "$cell" "$TMPDIR"
mkfifo "$cell/control"
export A_COST_PERF_CONTROL="$cell/control"
"$binary" fusion dispatch > "$cell/DISPATCH.json"
sha256sum "$binary" "$study/production-f16.bin" "$study/inputs-A0.tsv" "$0" > "$cell/SHA256SUMS"
printf '%s %s %s %s pid=%s\n' "$(date -u +%FT%TZ)" "$label" "$size" "$kind" "$$" >> "$study/PROGRESS.txt"
command=(taskset -c 0 perf)
case "$kind" in
    cycles)
        command+=(record -e cycles:u -F 997 -g --call-graph fp -o "$cell/perf.data")
        ;;
    ibs)
        command+=(mem record -e mem-ldst -a -C 0 -W -c 65536 -g --call-graph fp -o "$cell/perf.data")
        ;;
    counters)
        command+=(stat -x, -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses -o "$cell/counters.csv")
        ;;
    fills)
        command+=(stat -x, -e ls_dmnd_fills_from_sys.local_l2,ls_dmnd_fills_from_sys.local_ccx,ls_dmnd_fills_from_sys.near_cache,ls_dmnd_fills_from_sys.dram_io_all -o "$cell/counters.csv")
        ;;
    pipeline1|pipeline2|l2|l3|memory|metric-*)
        case "$kind" in
            pipeline1) metric=PipelineL1 ;; pipeline2) metric=PipelineL2 ;;
            l2) metric=l2_cache ;; l3) metric=l3_cache ;; memory) metric=memory_controller ;;
            metric-*) metric=${kind#metric-}; [[ $metric =~ ^[a-zA-Z0-9_]+$ ]] ;;
        esac
        command+=(stat -x, --metric-no-threshold -M "$metric")
        if [[ $kind = l3 || $kind = memory || $kind = metric-l3_* || $kind = metric-umc_* ]]; then command+=(-a -C 0); fi
        command+=(-o "$cell/counters.csv")
        ;;
    *) echo 'unsupported capture kind' >&2; exit 2 ;;
esac
command+=(--delay=-1 --control="fifo:$cell/control" -- "$binary" profile "$study/pairs" \
    "$study/production-f16.bin" "$study/inputs-A0.tsv" "$size" "$cell" 1 10)
python3 -B "$study/source/scripts/bench/a_family_cost_admit.py" "$cell/admission.json" -- \
    "${command[@]}" > "$cell/run.log" 2>&1
python3 - "$cell/admission.json" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text());assert r['status']=='accepted',r
PY
# Capture completion precedes every derived report. Parse with a_fusion_parse.sh.
printf '%s %s %s capture complete\n' "$(date -u +%FT%TZ)" "$size" "$kind" > "$cell/CAPTURE_COMPLETE.txt"
printf '%s %s %s complete\n' "$(date -u +%FT%TZ)" "$size" "$kind" > "$cell/COMPLETE.txt"
