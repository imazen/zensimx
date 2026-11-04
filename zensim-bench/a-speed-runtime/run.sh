#!/usr/bin/env bash
# Single-host collector. Every process takes the shared compute lock.
set -euo pipefail
root=$(realpath "${1:?artifact root}")
binary=$(realpath "${2:?benchmark binary}")
cpus=${BENCH_CPUS:?set the same affinity for every arm and thread count}
export TMPDIR="$root/scratch"
mkdir -p "$TMPDIR" "$root/results"
printf '%s\n' "$$" > "$root/collector.pid"
sha256sum "$binary" > "$root/benchmark.sha256"
for threads in ${BENCH_THREADS:-1 16}; do
    [[ $threads = 1 || $threads = 16 ]] || exit 2
    for size in ${BENCH_SIZES:-64 256 1024 4096}; do
        [[ " 64 256 1024 4096 " = *" $size "* ]] || exit 2
        cell="${threads}t_${size}"
        for mode in ${BENCH_MODES:-latency rss-zensim_a rss-fast_ssim2 rss-margarine}; do
            stem="$root/results/${cell}_${mode}"
            test ! -e "$stem.log" || { echo "refusing to overwrite $stem.log" >&2; exit 2; }
            printf '%s %s %s queued collector=%s\n' "$(date -u +%FT%TZ)" "$cell" "$mode" "$$" > "$root/PROGRESS.txt"
            if [[ $mode = latency ]]; then
                args=(latency "$stem.json")
            else
                args=(rss "${mode#rss-}")
            fi
            # Record live quiet-state evidence after acquiring the lock. A
            # foreign heavy job or load >=2 refuses; preserve every attempt.
            flock "$HOME/tmp/heavy.lock" bash -c '
                set -euo pipefail
                root=$1; cell=$2; mode=$3; shift 3
                printf "%s %s %s running lock-holder=%s\n" "$(date -u +%FT%TZ)" "$cell" "$mode" "$$" > "$root/PROGRESS.txt"
                uptime
                ps -eo pid,ppid,comm,pcpu,args --sort=-pcpu | sed -n "1,12p"
                if [[ $mode = latency ]]; then
                    deadline=$((SECONDS+600))
                    until awk "{exit !(\$1 < 2.0)}" /proc/loadavg &&
                        ! pgrep -x "cc1plus|cc1|rustc|cargo|cmake|ninja|make|zensim_mlp_trai" >/dev/null; do
                        (( SECONDS < deadline )) || { echo "quiet load gate refused"; exit 3; }
                        echo "waiting for load1 < 2.0 and no compiler/trainer: $(cat /proc/loadavg)"
                        sleep 10
                    done
                fi
                exec "$@"
            ' bash "$root" "$cell" "$mode" \
                "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 16 -- \
                env RAYON_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" \
                ZENBENCH_LAUNCHER_PIDS="$$" ZENBENCH_GATE_TRACE=1 \
                /usr/bin/time -v -o "$stem.time.txt" taskset -c "$cpus" "$binary" "$root/fixtures" "$threads" "$size" "${args[@]}" \
                2>&1 | tee "$stem.log"
            printf '%s\n' "$cell $mode" >> "$root/COMPLETED.txt"
        done
    done
done
# A targeted retry is not a full matrix. Require every output before declaring
# raw collection complete; the reporter separately checks scientific admission.
complete=1
for threads in 1 16; do
    for size in 64 256 1024 4096; do
        test -s "$root/results/${threads}t_${size}_latency.json" || complete=0
        for arm in zensim_a fast_ssim2 margarine; do
            test -s "$root/results/${threads}t_${size}_rss-${arm}.time.txt" || complete=0
        done
    done
done
printf '%s raw-collection-finished matrix-outputs-present=%s\n' "$(date -u +%FT%TZ)" "$complete" > "$root/PROGRESS.txt"
if [[ $complete = 1 ]]; then touch "$root/COMPLETE"; fi
