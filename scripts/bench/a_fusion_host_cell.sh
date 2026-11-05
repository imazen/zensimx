#!/usr/bin/env bash
# Release the shared lock between cells and while waiting for a quiet host.
set -euo pipefail
host=$(hostname)
case "$host" in dev) memory=24G ;; i270) memory=16G ;; *) exit 2 ;; esac
stage=${1:?build|measure-cell}
study=${2:?study root}
label=${3:?immutable label}
case "$stage" in build) jobs=8 ;; measure-cell) jobs=16 ;; *) exit 2 ;; esac
if [[ ${A_FUSION_HOST_LOCK_HELD:-0} != 1 ]]; then
    if [[ $host = dev ]]; then
        while [[ ! -f $study/A_REPORTED.json ]]; do
            printf '%s waiting for the owner-ordered Intel Phase-A report\n' "$(date -u +%FT%TZ)"
            sleep 15
        done
    fi
    while true; do
        status=0
        flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem "$memory" --jobs "$jobs" -- \
            env A_FUSION_HOST_LOCK_HELD=1 bash "$0" "$@" || status=$?
        [[ $status = 75 ]] || exit "$status"
        # A quiet-gate refusal owns no running scorer; queued training can proceed.
        sleep 15
    done
fi
export A_FUSION_LOCK_HELD=1
export TMPDIR="$study/scratch"
mkdir -p "$TMPDIR"
native="$study/source/scripts/bench/a_fusion_native.sh"
if [[ $stage = build ]]; then
    snapshot=${4:?source snapshot}
    mkdir -p "$study/source"
    cp -r "$snapshot/." "$study/source/"
    cp "$snapshot/SOURCE_COMMIT.txt" "$study/SOURCE_COMMIT.txt"
    exec bash "$native" build "$study" "$label"
fi
export A_FUSION_BAKE="$study/production-f16.bin" A_FUSION_PAIRS="$study/pairs"
export A_FUSION_BEFORE_LABEL=${A_FUSION_BEFORE_LABEL:?baseline label}
export A_FUSION_CONTROLLER_LABEL=${A_FUSION_CONTROLLER_LABEL:-$A_FUSION_BEFORE_LABEL}
exec python3 -B "$study/source/scripts/bench/a_fusion_quiet.py" \
    "$study/$label/quiet-${4:?size}-${5:?threads}.jsonl" -- \
    bash "$native" measure-cell "$study" "$label" "$4" "$5"
