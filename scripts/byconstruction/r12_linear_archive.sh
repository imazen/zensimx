#!/usr/bin/env bash
# Preserve complete R12 artifacts; Cargo output is a rebuildable cache.
# Caller holds heavy.lock/run-heavy. Never delete destination data.
set -euo pipefail
cd "$(dirname "$0")/../.."
O=/var/tmp/byc-linear-20261104
D=/mnt/tower/output/zensim/r12-linear-20261104
python3 -B scripts/byconstruction/r12_linear_tables.py --root "$O" --with-extra --out "$O/FINAL_TABLES.md"
mkdir -p "$D"
rsync -aLO --no-owner --no-group --exclude=/target/ --exclude=/archive.log --exclude=/archive-check.txt "$O/" "$D/"
rsync -ancLO --no-owner --no-group --itemize-changes --exclude=/target/ --exclude=/archive.log --exclude=/archive-check.txt "$O/" "$D/" > "$O/archive-check.txt"
if test -s "$O/archive-check.txt"; then
  cat "$O/archive-check.txt"
  exit 1
fi
cp "$O/archive-check.txt" "$D/archive-check.txt"
printf 'R12 archive checksum comparison is empty: %s\n' "$D"
