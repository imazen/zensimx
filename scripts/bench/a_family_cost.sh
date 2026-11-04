#!/usr/bin/env bash
# Run on the selected quiet benchmark host. Never run measurements on dev.
set -euo pipefail
stage=${1:?build|prepare|bench|profile}
study=${2:?absolute study directory}
runner=$(realpath "$0")
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$PATH"
export TMPDIR="$study/scratch"
export CARGO_TARGET_DIR="$study/target"
export ZENSIM_FORMULA_REV=7
export ZENBENCH_NO_CALIBRATE=1
mkdir -p "$TMPDIR" "$study/logs"
case "$(hostname)" in i270|r5900xt) ;; *) echo 'unsupported measurement host' >&2; exit 2 ;; esac
cd "$study/build-source"
heavy() {
    flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- "$@"
}
bin=${A_COST_BIN:-"$study/binaries/a_family_cost"}
case "$stage" in
    build)
        export CARGO_PROFILE_RELEASE_DEBUG=1
        export RUSTFLAGS='-C force-frame-pointers=yes'
        heavy cargo fmt -p zensim --check 2>&1 | tee "$study/logs/fmt.log"
        heavy cargo clippy --release --locked -p zensim --example a_family_cost --features training,custom-profiles 2>&1 | tee "$study/logs/clippy.log"
        heavy cargo build --release --locked -p zensim --example a_family_cost --features training,custom-profiles 2>&1 | tee "$study/logs/build.log"
        mkdir -p "$study/binaries"
        test ! -e "$bin"
        cp "$CARGO_TARGET_DIR/release/examples/a_family_cost" "$bin"
        sha256sum "$bin" > "$study/binary.sha256"
        ;;
    prepare)
        heavy "$bin" prepare "$study/input/sources.json" "$study/pairs" 2>&1 | tee "$study/logs/prepare.log"
        ;;
    bench)
        threads=${3:?threads}; size=${4:?size}
        cpus=${A_COST_CPUS:?set explicit CPU affinity}
        test ! -e "$study/timing/${size}-${threads}t"
        export RAYON_NUM_THREADS="$threads"
        # Register precisely this exec'd process with zenbench's existing owner
        # mechanism, so its Rayon CPU use is not classified as foreign load.
        heavy env A_COST_HEAVY_WORKER=1 bash "$runner" bench-worker "$study" "$threads" "$size" 2>&1 | tee "$study/logs/bench-${size}-${threads}t.log"
        ;;
    bench-worker)
        test "${A_COST_HEAVY_WORKER:-}" = 1
        threads=${3:?threads}; size=${4:?size}
        export ZENBENCH_LAUNCHER_PIDS=$$
        exec taskset -c "${A_COST_CPUS:?explicit affinity}" "$bin" bench "$study/pairs" "$study/production-f16.bin" "$study/inputs-A0.tsv" "$size" "$study/timing/${size}-${threads}t" "$threads"
        ;;
    profile)
        size=${3:?size}; repeats=${4:?repeats}
        cpus=${A_COST_CPUS:?set explicit CPU affinity}
        export RAYON_NUM_THREADS=1
        mkdir -p "$study/perf/$size"
        control="$study/perf/$size/control"
        test ! -e "$control"
        mkfifo "$control"
        export A_COST_PERF_CONTROL="$control"
        heavy taskset -c "$cpus" perf record -e cycles:u -F 997 -g --call-graph fp --delay=-1 --control="fifo:$control" -o "$study/perf/$size/perf.data" -- "$bin" profile "$study/pairs" "$study/production-f16.bin" "$study/inputs-A0.tsv" "$size" "$study/perf/$size" 1 "$repeats" 2>&1 | tee "$study/logs/perf-${size}.log"
        perf report -i "$study/perf/$size/perf.data" --stdio --no-children --percent-limit 0 -F overhead,symbol > "$study/perf/$size/self.txt"
        perf report -i "$study/perf/$size/perf.data" --stdio --children --percent-limit 0 -F overhead,overhead_children,symbol > "$study/perf/$size/children.txt"
        ;;
    timings)
        shift 2
        for cell in "$@"; do
            size=${cell%:*}; threads=${cell#*:}
            if [ "$threads" = 1 ]; then
                export A_COST_CPUS=${A_COST_CPUS_1:?one-thread affinity}
            else
                test "$threads" = 16
                export A_COST_CPUS=${A_COST_CPUS_16:?sixteen-thread affinity}
            fi
            bash "$runner" bench "$study" "$threads" "$size"
        done
        ;;
    *) exit 2 ;;
esac
