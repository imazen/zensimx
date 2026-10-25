#!/bin/bash
# Perf parity of a corrected ringing form: old binary (raw form, pre-change build) vs new binary (FORM), plus controls.
# Exploration record: benchmarks/byconstruction_2026-10-22.md. Usage: ringfix_timing.sh OLD_BIN NEW_BIN BAKE OUT_DIR [FORM]
set -euo pipefail
OLD=$1; NEW=$2; BAKE=$3; OUT=$4; F=${5:-hpm}; H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
run() { echo "== $1 $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)"; shift; "$@"; }
RING_A=raw RING_B=$F run old-vs-$F-512 "$H/ring_timing.sh" "$OLD" "$NEW" "$BAKE" 512 40 16 "$OUT/old_vs_${F}_512.jsonl"
RING_A=raw RING_B=$F run old-vs-$F-2048 "$H/ring_timing.sh" "$OLD" "$NEW" "$BAKE" 2048 8 16 "$OUT/old_vs_${F}_2048.jsonl"
RING_A=raw RING_B=$F run old-vs-$F-64 "$H/ring_timing.sh" "$OLD" "$NEW" "$BAKE" 64 400 16 "$OUT/old_vs_${F}_64.jsonl"
RING_A=raw RING_B=raw run old-vs-newraw-512 "$H/ring_timing.sh" "$OLD" "$NEW" "$BAKE" 512 40 16 "$OUT/old_vs_newraw_512.jsonl"
RING_A=raw RING_B=$F run newraw-vs-$F-512 "$H/ring_timing.sh" "$NEW" "$NEW" "$BAKE" 512 40 16 "$OUT/newraw_vs_${F}_512.jsonl"
THREADS=8 CPUS=0-7 RING_A=raw RING_B=$F run old-vs-$F-2048-8t "$H/ring_timing.sh" "$OLD" "$NEW" "$BAKE" 2048 8 16 "$OUT/old_vs_${F}_2048_8t.jsonl"
echo "== done $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)"
