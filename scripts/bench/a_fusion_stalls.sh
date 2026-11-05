#!/usr/bin/env bash
# Additional Phase-1 attribution, using exactly the original scorer windows.
set -euo pipefail
[[ $(hostname) = i270 ]] || { echo 'Intel stall capture requires i270' >&2; exit 2; }
study=${1:?study root}
label=${2:?immutable capture label}
mode=${3:-capture}
if [[ ${A_FUSION_STALL_LOCK_HELD:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 2 -- \
        env A_FUSION_STALL_LOCK_HELD=1 bash "$0" "$@"
fi
export TMPDIR="$study/scratch" ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1
old="$HOME/tmp/zensim-cost"
out="$study/$label"
if [[ $mode = confirm-l3 || $mode = confirm-sparse ]]; then
    test -f "$out/COMPLETE.txt"
    sha256sum "$0" > "$out/$mode-SCRIPT_SHA256SUMS"
    confirm() {
        local size=$1 event=$2 period=$3 name=$4
        local cell="$out/$name"
        test ! -e "$cell"
        mkdir -p "$cell"
        mkfifo "$cell/control-stat" "$cell/control-record"
        A_COST_PERF_CONTROL="$cell/control-stat" python3 -B "$study/source/scripts/bench/a_family_cost_admit.py" \
            "$cell/count-admission.json" -- taskset -c 2 perf stat -x, -e "$event" --delay=-1 \
            --control="fifo:$cell/control-stat" -o "$cell/count.csv" -- \
            "$old/binaries/a_family_cost" profile "$old/pairs" "$old/production-f16.bin" \
            "$old/inputs-A0.tsv" "$size" "$cell/count-window" 1 10 > "$cell/count-run.log" 2>&1
        A_COST_PERF_CONTROL="$cell/control-record" python3 -B "$study/source/scripts/bench/a_family_cost_admit.py" \
            "$cell/admission.json" -- taskset -c 2 perf record -e "$event:pp" -c "$period" -g --call-graph fp \
            --delay=-1 --control="fifo:$cell/control-record" -o "$cell/perf.data" -- \
            "$old/binaries/a_family_cost" profile "$old/pairs" "$old/production-f16.bin" \
            "$old/inputs-A0.tsv" "$size" "$cell/record-window" 1 10 > "$cell/run.log" 2>&1
        perf report -i "$cell/perf.data" --stdio --no-inline --call-graph none --no-children \
            --percent-limit 0 -F overhead,period,sample,symbol > "$cell/self.txt"
        perf script -i "$cell/perf.data" --no-inline -F period,ip,sym > "$cell/samples.txt"
    }
    if [[ $mode = confirm-l3 ]]; then
        confirm 256 mem_load_retired.l3_miss 1 256-l3-period1
    else
        confirm 256 mem_load_retired.l2_miss 101 256-l2-period101
        confirm 1024 mem_load_retired.l3_miss 101 1024-l3-period101
    fi
    python3 -B "$study/source/scripts/bench/a_fusion_stall_report.py" "$out" "$out/REPORT-$mode.json"
    exit
fi
[[ $mode = capture ]] || { echo "choose capture, confirm-l3 or confirm-sparse" >&2; exit 2; }
test ! -e "$out"
mkdir -p "$out"
sha256sum "$old/binaries/a_family_cost" "$old/production-f16.bin" "$old/inputs-A0.tsv" "$0" > "$out/SHA256SUMS"
perf list --details > "$out/perf-list.txt"
metrics=(
    TopdownL2
    tma_l1_bound,tma_l2_bound,tma_l3_bound,tma_dram_bound
    tma_store_bound,tma_mem_latency,tma_mem_bandwidth
    tma_ports_utilization,tma_divider
)
events=(mem_load_retired.l1_miss:pp mem_load_retired.l2_miss:pp mem_load_retired.l3_miss:pp exe_activity.bound_on_loads)
for size in 256 1024 large; do
    for i in "${!metrics[@]}"; do
        cell="$out/$size-stat-$i"
        mkdir -p "$cell"
        mkfifo "$cell/control"
        export A_COST_PERF_CONTROL="$cell/control"
        printf '%s %s %s stat-%s pid=%s\n' "$(date -u +%FT%TZ)" "$label" "$size" "$i" "$$" >> "$study/PROGRESS.txt"
        printf '%s\n' "${metrics[$i]}" > "$cell/metric.txt"
        python3 -B "$study/source/scripts/bench/a_family_cost_admit.py" "$cell/admission.json" -- \
            taskset -c 2 perf stat -x, --metric-no-threshold -M "${metrics[$i]}" --delay=-1 \
            --control="fifo:$cell/control" -o "$cell/counters.csv" -- \
            "$old/binaries/a_family_cost" profile "$old/pairs" "$old/production-f16.bin" \
            "$old/inputs-A0.tsv" "$size" "$cell" 1 10 > "$cell/run.log" 2>&1
    done
    for event in "${events[@]}"; do
        name=${event%%:*}
        cell="$out/$size-$name"
        mkdir -p "$cell"
        mkfifo "$cell/control"
        export A_COST_PERF_CONTROL="$cell/control"
        printf '%s %s %s %s pid=%s\n' "$(date -u +%FT%TZ)" "$label" "$size" "$event" "$$" >> "$study/PROGRESS.txt"
        printf '%s\n' "$event" > "$cell/event.txt"
        python3 -B "$study/source/scripts/bench/a_family_cost_admit.py" "$cell/admission.json" -- \
            taskset -c 2 perf record -e "$event" -F 997 -g --call-graph fp --delay=-1 \
            --control="fifo:$cell/control" -o "$cell/perf.data" -- \
            "$old/binaries/a_family_cost" profile "$old/pairs" "$old/production-f16.bin" \
            "$old/inputs-A0.tsv" "$size" "$cell" 1 10 > "$cell/run.log" 2>&1
        perf report -i "$cell/perf.data" --stdio --no-inline --call-graph none --no-children \
            --percent-limit 0 -F overhead,period,sample,symbol > "$cell/self.txt"
        perf script -i "$cell/perf.data" --no-inline -F period,ip,sym > "$cell/samples.txt"
    done
done
printf '%s %s complete 24/24 cells\n' "$(date -u +%FT%TZ)" "$label" > "$out/COMPLETE.txt"
