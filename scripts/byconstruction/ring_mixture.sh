#!/bin/bash
# Label-free damage-mixture test for every cell of the given arms, scored under a measurement arm (FORM: raw or hpm
# for ZENSIM_RING_FORM, linpyr for ZENSIM_PYRAMID=linear with the raw ring form) with the given scorer build.
# Exploration record: benchmarks/byconstruction_2026-10-22.md.
# Usage: ring_mixture.sh BIN_DIR FORM JOBS ARM [ARM ...]; MIX selects another mixture set (default: the standard one).
# BYC_CELLS selects another cell root, retaining the original default.
set -euo pipefail
BIN=$1; FORM=$2; J=$3; shift 3
H=$(cd "$(dirname "$0")" && pwd)
MIX=${MIX:-/mnt/v/output/zensim/byconstruction-2026-10-22/mixture}
CELLS=${BYC_CELLS:-$HOME/tmp/byc/cells}
FOLDS=${BYC_MIX_FOLDS:-}
case $FORM in
  raw|hpm) ARM_ENV=(ZENSIM_RING_FORM="$FORM") ;;
  linpyr) ARM_ENV=(ZENSIM_RING_FORM=raw ZENSIM_PYRAMID=linear) ;;
  *) echo "unknown form $FORM" >&2; exit 2 ;;
esac
for arm in "$@"; do
  for d in "$CELLS"/"$arm"/without_*_s*; do
    cell=$(basename "$d"); cell=${cell#without_}
    # Optional experiment-specific subset; the default still scores every cell.
    fold=${cell%_s*}
    if [ -n "$FOLDS" ]; then
      case ,$FOLDS, in *,"$fold",*) ;; *) continue ;; esac
    fi
    bakes=("$d"/dense/*.bin)
    [ "${#bakes[@]}" = 1 ] && [ -f "${bakes[0]}" ] || { echo "expected one dense bake: $d" >&2; exit 2; }
    bake=${bakes[0]}
    env "${ARM_ENV[@]}" RAYON_NUM_THREADS=1 python3 "$H/mixture_test.py" score --out "$MIX" \
      --bake "${arm}_${cell}=$bake" --bin-dir "$BIN" --jobs "$J"
  done
done
