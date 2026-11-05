#!/usr/bin/env bash
# Invoked once inside the shared heavy lock and run-heavy scope.
set -euo pipefail
root=$1
bake=$2
certifier=$3
bindgen=$4
source_commit=${VIZ_SOURCE_COMMIT:-$(jj log -r @- --no-graph -T commit_id)}
sdr=${VIZ_SDR_BANK:-/var/tmp/byc-fchroma-20261029/r3/w100/root/wide/main/real/cid22_fit.parquet}
hdr=${VIZ_HDR_BANK:-/mnt/v/output/zensim/byc-hdr-20261104/banks-cube-root-203/teacher-train-r1/features.parquet}
export TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_FORMULA_REV=7
export ZENSIM_WORKBENCH_BAKES="A=$bake"
export ZENSIM_WORKBENCH_LABEL='zensim A revision 7 browser'
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/pages-feasibility-target}"
unset ZENSIM_VIZ_MAP_DIR
out="$root/pages-final-native"
mkdir "$out"
cargo run --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --example viz_parity -- "$bake" "$out/maps"
cargo build --release --locked --manifest-path tools/zensim-viz/Cargo.toml
export VIZ_CERTIFIER="$certifier"
exporter="$CARGO_TARGET_DIR/release/zensim-viz-export"
"$exporter" pages-bank "$bake" --bank "$sdr" --out "$out/sdr"
"$exporter" pages-bank "$bake" --bank "$hdr" --out "$out/hdr"
"$exporter" weights "$bake" --out "$out/weights"
"$exporter" function "$bake" --out "$out/function"
"$exporter" audit "$bake" --bank "$sdr" --out "$out/audit"
cargo run --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --example viz_lens_parity -- "$out" "$out/lens.json"
export VIZ_F_BAKE="${VIZ_F_BAKE:-$root/F0-185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088.bin}"
export VIZ_F_BANK="$sdr"
cargo test --release --locked --manifest-path tools/zensim-viz/Cargo.toml --features real-bake-tests
cargo clippy --release --locked --manifest-path tools/zensim-viz/Cargo.toml --all-targets --features real-bake-tests -- -D warnings
bash tools/zensim-viz/finish_pages.sh "$root" "$bake" "$bindgen" "$source_commit"
