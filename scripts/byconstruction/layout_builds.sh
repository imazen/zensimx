#!/bin/bash
# Build byc_score_timing from the old (pre-ringing-change) and new source trees under several function alignments,
# so whole-walk timing compares code over several placements instead of one. Exploration record:
# benchmarks/byconstruction_2026-10-22.md. Usage: layout_builds.sh LAYOUT_ROOT ALIGN...
set -euo pipefail
L=$1; shift
for a in "$@"; do
  for tree in old new; do
    T=$L/target-$tree-$a
    flags=""; [ "$a" != default ] && flags="-C llvm-args=-align-all-functions=$a"
    (cd $L/$tree && CARGO_TERM_COLOR=never RUSTFLAGS="$flags" CARGO_TARGET_DIR=$T cargo build --release -p zensim-validate --bin byc_score_timing > $T.log 2>&1) || { echo "build $tree $a failed"; tail -5 $T.log; exit 1; }
    mkdir -p $L/bin/$tree-$a && cp -p $T/release/byc_score_timing $L/bin/$tree-$a/
    echo "$tree $a $(sha256sum $L/bin/$tree-$a/byc_score_timing | cut -c1-16)"
  done
done
