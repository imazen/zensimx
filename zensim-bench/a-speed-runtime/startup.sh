#!/usr/bin/env bash
# No-input command startup is separate from the resident-pixel paired matrix.
set -euo pipefail
root=$(realpath "${1:?artifact root}")
startup=$(realpath "${2:?startup instrument}")
export TMPDIR="$root/scratch"
for threads in 1 16; do
    stem="$root/startup-${threads}t"
    test ! -e "$stem.log"
    flock "$HOME/tmp/heavy.lock" bash -c '
        set -euo pipefail
        until awk "{exit !(\$1 < 2.0)}" /proc/loadavg; do sleep 5; done
        exec "$@"
    ' bash "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 16 -- \
        env RAYON_NUM_THREADS="$threads" OMP_NUM_THREADS="$threads" \
        ZENBENCH_LAUNCHER_PIDS="$$" ZENBENCH_GATE_TRACE=1 \
        taskset -c 0-15 "$startup" "$root/bin/margarine-cli" "$stem.json" \
        2>&1 | tee "$stem.log"
done
