#!/usr/bin/env bash
# Fresh attempt root, one heavy admission per latency or RSS worker.
set -euo pipefail
root=$(realpath -m "${1:?attempt root}")
bin=$(realpath "${2:?runtime binary}")
tier=${3:?v3 or v4x}
host=$(hostname -s)
case "$host:$tier" in
    dev:v3|dev:v4x)
        lock="$HOME/tmp/handoff/20261117/dev-bench.lock"
        mem=24G
        many_cpus=0-7,16-23
        ;;
    i265:v3)
        lock="$HOME/tmp/i265-heavy.lock"
        mem=16G
        many_cpus=0-15
        ;;
    *) echo "Unsupported benchmark placement $host:$tier" >&2; exit 2 ;;
esac
mkdir "$root"
export TMPDIR="$root/scratch"
mkdir "$TMPDIR"
sha256sum "$bin" > "$root/binary.sha256"
printf '%s\t%s\t%s\n' "$host" "$tier" "$$" > "$root/collector.tsv"
lscpu -e=CPU,CORE,MAXMHZ > "$root/cpus.txt"
printf 'queued\n' > "$root/progress.tsv"
failed=0
for threads in ${REV8_THREADS:-16 1}; do
    if [[ $threads == 1 ]]; then
        cpus=0
    else
        [[ $threads == 16 ]]
        cpus=$many_cpus
    fi
    for geometry in ${REV8_SIZES:-64x64 256x256 840x944 1024x1024 4000x3000}; do
        width=${geometry%x*}
        height=${geometry#*x}
        for mode in ${REV8_MODES:-latency zensim-a-rev7 rev8-alg-s4 rev8-alg-s5 rev8-hybrid-s5}; do
            stem="$root/${threads}t_${geometry}_$mode"
            if [[ $mode == latency ]]; then
                args=(latency "$stem.json")
            else
                args=(rss "$mode")
            fi
            printf '%s\tqueued\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$threads" "$geometry" "$mode" "$$" >> "$root/progress.tsv"
            if flock "$lock" bash -c '
                set -euo pipefail
                root=$1; threads=$2; geometry=$3; mode=$4; shift 4
                printf "%s\trunning\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$threads" "$geometry" "$mode" "$$" >> "$root/progress.tsv"
                uptime
                if [[ $mode == latency && -n ${REV8_ADMISSION_PROGRAM:-} ]]; then
                    "$REV8_ADMISSION_PROGRAM"
                fi
                if [[ $(hostname -s) == i265 ]]; then
                    df -B1 / | awk "NR == 2 { exit !(\$4 >= 5368709120) }"
                fi
                exec "$@"
            ' bash "$root" "$threads" "$geometry" "$mode" \
                "$HOME/work/claudehints/scripts/run-heavy" --mem "$mem" --jobs 16 -- \
                env RAYON_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" ZENBENCH_LAUNCHER_PIDS="$$" ZENBENCH_GATE_TRACE=1 ZENBENCH_NO_SAVE=1 \
                /usr/bin/time -v -o "$stem.time.txt" taskset -c "$cpus" "$bin" "$tier" "$threads" "$width" "$height" "${args[@]}" \
                > "$stem.log" 2>&1; then
                printf '%s\tcomplete\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$threads" "$geometry" "$mode" >> "$root/progress.tsv"
            else
                code=$?
                printf '%s\tfailed-%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$code" "$threads" "$geometry" "$mode" >> "$root/progress.tsv"
                failed=1
            fi
        done
    done
done
printf '%s\tfinished\t%s\n' "$(date -u +%FT%TZ)" "$failed" >> "$root/progress.tsv"
exit "$failed"
