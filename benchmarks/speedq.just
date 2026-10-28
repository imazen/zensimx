# Imported by the root justfile; run these recipes through that file.
# SPEEDQ uses the existing synthetic matrix images and Rust scoring surfaces.
# Explicitly omit optional C++ oracle arms: only existing Rust peers are required.
speedq-build:
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo bench --no-run --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar --features speedq,ssim2-rayon --message-format=json-render-diagnostics

speedq-clippy:
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo clippy --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar --features speedq,ssim2-rayon -- -D warnings

[positional-arguments]
speedq-parity binary dest *options:
    #!/usr/bin/env bash
    binary=$1
    dest=$2
    shift 2
    exec python3 scripts/demos/speedq_run.py parity --binary "$binary" --dest "$dest" "$@"

speedq-test:
    python3 -m unittest discover -s scripts/demos -p 'test_speedq*.py' -v

# Marker counts are progress only; the reporter validates qualification evidence.
[positional-arguments]
speedq-status raw_dir:
    python3 scripts/demos/speedq_run.py status --dest "$1"

# Run one traced segment; keep its destination separate from qualification data.
[positional-arguments]
speedq-diagnose *options:
    python3 scripts/demos/speedq_run.py diagnose "$@"

[positional-arguments]
speedq-mirror source dest:
    nice -n19 ionice -c3 rsync -a --no-owner --no-group "$1/" "$2/"

[positional-arguments]
speedq-timing *options:
    python3 scripts/demos/speedq_run.py timing "$@"

[positional-arguments]
speedq-rss *options:
    python3 scripts/demos/speedq_run.py rss "$@"

# Freeze the executable named by the successful Cargo JSON build receipt.
speedq-freeze build_log dest:
    python3 scripts/demos/speedq_run.py freeze --build-log "{{build_log}}" --dest "{{dest}}"

# Produce the qualification report and optional before/after table.
[positional-arguments]
speedq-report *options:
    python3 scripts/demos/speed_matrix_report.py --speedq "$@"

costcmp-build:
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo bench --no-run --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar --features costcmp --message-format=json-render-diagnostics

costcmp-clippy:
    ZENSIM_BENCH_SKIP_CPP_FFI=1 cargo clippy --manifest-path zensim-bench/Cargo.toml --bench ssim2_speed_bar --features costcmp -- -D warnings

# The curated revision-7 models' runtime grid (costcmp_run.py --r7-grid): its own assessment manifest, built off dev.
costcmp-r7-build:
    CARGO_TARGET_DIR=zensim-bench/target cargo bench --locked --no-run --message-format=json-render-diagnostics --manifest-path zensim-bench/r7-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp

costcmp-r7-clippy:
    CARGO_TARGET_DIR=zensim-bench/target cargo clippy --locked --manifest-path zensim-bench/r7-runtime/Cargo.toml --bench ssim2_speed_bar --features costcmp -- -D warnings

costcmp-test:
    python3 -m unittest discover -s scripts/demos -p 'test_costcmp.py' -v

[positional-arguments]
costcmp-run *options:
    python3 scripts/demos/costcmp_run.py "$@"

[positional-arguments]
costcmp-report *options:
    python3 scripts/demos/speed_matrix_report.py --costcmp "$@"
