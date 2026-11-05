#!/usr/bin/env bash
# Manual continuation only: preserve accepted captures and replay parsing separately.
set -euo pipefail
[[ $(hostname) = i270 ]] || { echo 'run this recipe on i270 when idle' >&2; exit 2; }
study=${1:?absolute frozen study root}
driver=$(cd "$(dirname "$0")" && pwd)
if [[ ${2:-} = --verify ]]; then
    printf '%s  %s\n' 460eedfaa3b8830944a0b969b384f0fabd16077869e0b7d1de6d97cf2512f13c "$study/pairs/pairs.json" \
        185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088 "$study/production-f16.bin" | sha256sum --check -
    for label in baseline-pinned moment-pinned activity-pinned activity-panel-pinned; do
        sha256sum --check "$study/$label/BINARY_SHA256SUMS"
    done
    test -f "$study/activity-panel-pinned/GATE_SOURCE_COMPATIBILITY.json"
    exit
fi
run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$
run="$study/manual-idle-$run_id"
mkdir "$run"
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 1 -- \
    bash "$0" "$study" --verify > "$run/verify.log" 2>&1
cells=('moment-pinned 64 16' 'activity-pinned 64 1' 'activity-pinned 64 16')
for size in 1024 large 256; do
    for threads in 1 16; do cells+=("activity-panel-pinned $size $threads"); done
done
for spec in "${cells[@]}"; do
    read -r label size threads <<< "$spec"
    cell="$study/$label/timing/$size-${threads}t"
    if [[ ! -f $cell/CAPTURE_COMPLETE.txt ]]; then
        if [[ -e $cell ]]; then mv "$cell" "$cell.invalid-before-$run_id"; fi
        printf '%s capture %s\n' "$(date -u +%FT%TZ)" "$spec"
        A_FUSION_NATIVE_DRIVER="$driver/a_fusion_native.sh" A_FUSION_BEFORE_LABEL=baseline-pinned \
            bash "$driver/a_fusion_host_cell.sh" measure-cell "$study" "$label" "$size" "$threads" \
            > "$run/$label-$size-${threads}t.capture.log" 2>&1
    fi
    # Failure here leaves CAPTURE_COMPLETE intact; the next invocation only reparses.
    bash "$driver/a_fusion_parse.sh" "$study" timing "$cell" "$cell/parses/idle-$run_id" \
        > "$run/$label-$size-${threads}t.parse.log" 2>&1
    printf '%s parsed %s\n' "$(date -u +%FT%TZ)" "$spec" | tee -a "$run/PROGRESS.txt"
done
date -u +%FT%TZ > "$run/COMPLETE.txt"
