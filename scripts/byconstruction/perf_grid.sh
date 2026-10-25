#!/usr/bin/env bash
# Paired before/after timing grid for the speed pass: B (revision 5) and Tga (revision 6) at 512² and 2048², at the
# v4x and v3 tiers, one thread pinned to core 2. Each cell is ring_timing.sh (alternating arm order per round) and
# timing_summary.py (median per-round ratio with a bootstrap CI). Record: benchmarks/byconstruction_2026-10-22.md.
# Usage: perf_grid.sh A_BIN B_BIN OUT_DIR [ROUNDS]; RAW (default OUT_DIR) receives the per-round JSONL.
# A_BB / A_TGA set arm A's B and Tga bakes, B_BB / B_TGA arm B's (default: arm A's), e.g. one binary under two
# revision stamps of the same weights. BAKES (default "b tga") and TIERS (default "v4x v3") narrow the grid.
# CASES=<cases.tsv> (perf_cases.py) replaces the two synthetic sizes with its size x content matrix; iterations per
# call then scale with the case's pixel count (8 at 2048², at most 2000).
set -euo pipefail
A=$1
B=$2
OUT=$3
ROUNDS=${4:-10}
RAW=${RAW:-$OUT}
mkdir -p "$OUT" "$RAW"
cd ~/work/zen/zensim--byconstruction
S=scripts/byconstruction
BB=${A_BB:-$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin}
TGA=${A_TGA:-$HOME/tmp/byc/parity/tga_full_rev6.bin}
BB_B=${B_BB:-$BB}
TGA_B=${B_TGA:-$TGA}
unset ZENSIM_FORMULA_REV ZENSIM_RING_FORM
{
  echo "# A=$A sha256=$(sha256sum "$A" | cut -c1-16)"
  echo "# B=$B sha256=$(sha256sum "$B" | cut -c1-16)"
  echo "# A bakes: $BB $TGA; B bakes: $BB_B $TGA_B"
  echo "# host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ) rounds=$ROUNDS load=$(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/grid.meta"
BAKES=${BAKES:-b tga}
TIERS=${TIERS:-v4x v3}
if [ -n "${CASES:-}" ]; then
  echo "# cases: $CASES" >> "$OUT/grid.meta"
  mapfile -t ROWS < <(tail -n +2 "$CASES" | cut -f1-4,6)
else
  ROWS=($'512\tsynthetic\t512\t512\t' $'2048\tsynthetic\t2048\t2048\t')
fi
printf 'bake\ttier\tcase\tclass\tsize\tA_ms\tB_ms\tdelta\tci95_lo\tci95_hi\n' > "$OUT/grid.tsv"
for name in $BAKES; do
  case $name in
    b) bake="b:$BB:$BB_B" ;;
    tga) bake="tga:$TGA:$TGA_B" ;;
    *) echo "unknown bake $name" >&2; exit 2 ;;
  esac
  for tier in $TIERS; do
    for row in "${ROWS[@]}"; do
      IFS=$'\t' read -r label class w h args <<< "$row"
      iters=$(( 8 * 2048 * 2048 / (w * h) ))
      (( iters < 8 )) && iters=8
      (( iters > 2000 )) && iters=2000
      jsonl="$RAW/${bake%%:*}_${tier}_${label}.jsonl"
      arms=${bake#*:}
      CASE_ARGS=$args TIER=$tier bash $S/ring_timing.sh "$A" "$B" "${arms%%:*}" "$w" "$iters" "$ROUNDS" "$jsonl" "${arms#*:}"
      python3 $S/timing_summary.py "$jsonl" | python3 -c "
import json, sys
d = json.load(sys.stdin)
lo, hi = d['ci95_pct']
print('\t'.join(['${bake%%:*}', '$tier', '$label', '$class', '${w}x${h}', f\"{d['A_median_ms']:.4f}\", f\"{d['B_median_ms']:.4f}\",
                 f\"{d['paired_delta_pct']:+.2f}%\", f'{lo:+.2f}%', f'{hi:+.2f}%']))" >> "$OUT/grid.tsv"
      tail -1 "$OUT/grid.tsv"
    done
  done
done
