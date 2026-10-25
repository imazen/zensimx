#!/bin/bash
# Build the orientation-band training root: pairs TSVs of every bank set and the coverage pool (bank row order), the
# byc_orient sidecar for each, then the stored v2e29 root widened with them at f2301-f2336 (ring_root.py orient).
# Exploration record: benchmarks/byconstruction_2026-10-22.md, "Orientation-selective bands".
# Usage: orient_root.sh BIN_DIR BANK_DIR POOL_DIR OUT_DIR THREADS
set -euo pipefail
BIN=$1; BANK=$2; POOL=$3; OUT=$4; THREADS=${5:-20}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
[ -d "$OUT/pairs" ] || python3 "$HERE/ring_root.py" globaff-pairs --bank "$BANK" --pool "$POOL" --out "$OUT/pairs"
mkdir -p "$OUT/sidecars"
for tsv in "$OUT"/pairs/*.tsv; do
  name=$(basename "$tsv" .tsv)
  [ -f "$OUT/sidecars/$name.csv" ] && continue
  echo "== $name $(date -u +%T)"
  RAYON_NUM_THREADS=$THREADS nice -n 15 "$BIN/byc_orient" --pairs "$tsv" --out "$OUT/sidecars/$name.csv.tmp"
  mv "$OUT/sidecars/$name.csv.tmp" "$OUT/sidecars/$name.csv"
done
(cd "$OUT/sidecars" && sha256sum ./*.csv > SHA256SUMS)
echo "== root $(date -u +%T)"
python3 "$HERE/ring_root.py" orient --root /mnt/v/output/zensim/v40r4-2026-10-08/v2e29 --bank "$BANK" \
  --sidecars "$OUT/sidecars" --out "$OUT/root"
echo ALL-DONE
