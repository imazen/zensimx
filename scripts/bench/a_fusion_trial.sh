#!/usr/bin/env bash
# Serialize one qualification phase; snapshots and executable labels are immutable.
set -euo pipefail
[[ $(hostname) = i270 ]] || { echo 'fusion qualification requires i270' >&2; exit 2; }
phase=${1:?build|gates|era-gates|pixels|era-pixels|measure|era-measure}
study=${2:?study root}
label=${3:?candidate label}
snapshot=${4:?source snapshot containing zensim/ and SOURCE_COMMIT.txt}
jobs=8
[[ $phase != *measure ]] || jobs=16
if [[ ${A_FUSION_TRIAL_LOCK_HELD:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs "$jobs" -- \
        env A_FUSION_TRIAL_LOCK_HELD=1 bash "$0" "$@"
fi
export A_FUSION_LOCK_HELD=1
export A_FUSION_BEFORE_LABEL=${A_FUSION_BEFORE_LABEL:-baseline-final}
export A_FUSION_BEFORE_TEST="$study/baseline-verified/steer_instrument"
export A_FUSION_EVAL_TABLE=/mnt/tower/output/zensim/fusion-eval-pixels-2026-11-04/pairs.tsv
export TMPDIR="$study/scratch"
native="$study/source/scripts/bench/a_fusion_native.sh"
out="$study/$label"
mkdir -p "$out"
record() {
    printf '%s %s %s rc=%s\n' "$(date -u +%FT%TZ)" "$label" "$1" "$2" >> "$study/TRIAL_RECEIPTS.tsv"
}
run() {
    local stage=$1 status=0
    test ! -e "$out/trial-$stage.log"
    bash "$native" "$stage" "$study" "$label" > "$out/trial-$stage.log" 2>&1 || status=$?
    record "$stage" "$status"
    return "$status"
}
case "$phase" in
    build)
        cp -r "$snapshot/." "$study/source/"
        cp "$snapshot/SOURCE_COMMIT.txt" "$study/SOURCE_COMMIT.txt"
        run build
        ;;
    gates|era-gates)
        test -x "$out/a_family_cost"
        for name in blur.rs feature_v2.rs feature_v2_kernels.rs feature_v2_oracle.rs fused.rs featcanon.rs; do
            test -f "$snapshot/zensim/src/$name"
        done
        cp -r "$snapshot/." "$study/source/"
        cp "$snapshot/SOURCE_COMMIT.txt" "$study/SOURCE_COMMIT.txt"
        # Match the formatting used to build the immutable normal executables.
        export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$PATH"
        cd "$study/source"
        cargo fmt -p zensim
        shopt -s nullglob
        for source in zensim/src/blur*.rs zensim/src/fused_basic_fusion.rs; do
            [[ ! -f $source ]] || rustfmt --edition 2024 "$source"
        done
        for stage in crate tiers instrument stable split; do
            if ! run "$stage"; then
                # Era arms preserve exact assertions and record their failures.
                [[ $phase = era-gates && $stage != instrument && $stage != stable ]] || exit 1
            fi
        done
        ;;
    pixels|era-pixels)
        test -x "$out/steer_instrument"
        for stage in identity nearid steer eval; do
            if ! run "$stage"; then
                [[ $phase = era-pixels ]] || exit 1
            fi
        done
        comparison="$out/comparison"
        mkdir -p "$comparison/expected" "$comparison/actual"
        python3 - "$study" "$label" <<'PY'
import pathlib,sys
r=pathlib.Path(sys.argv[1]);label=sys.argv[2]
for arm,source in [('expected','baseline-final'),('actual',label)]:
    target=r/label/'comparison'/arm
    fingerprints='baseline-complete-v2' if arm=='expected' else source
    for name,path in {
        'steer.json':r/source/'pixel/steer135/fusion/result.json',
        'identity.jsonl':r/fingerprints/'identity.jsonl',
        'nearid.jsonl':r/fingerprints/'nearid/candidate-fusion.jsonl',
        'nearid.json':r/source/'nearid/candidate-fusion.GATES.json',
        'eval.jsonl':r/fingerprints/'eval.jsonl',
    }.items():
        assert path.is_file(),path
        (target/name).symlink_to(path)
PY
        reporter="$study/source/scripts/bench/a_extraction_fusion.py"
        if [[ $phase = pixels ]]; then
            for stage in pixel-bits pixel-features eval-bits; do
                python3 -B "$reporter" "$stage" "$comparison" "$out/$stage.json"
            done
        else
            python3 -B "$reporter" drift "$comparison" "$out/drift.json"
        fi
        ;;
    measure|era-measure)
        [[ $phase != era-measure ]] || export A_FUSION_ALLOW_DRIFT=1
        run measure
        ;;
    *) echo 'unknown trial phase' >&2; exit 2 ;;
esac
