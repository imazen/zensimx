#!/usr/bin/env bash
# Head-only inspection and paired zenbench rounds. Caller holds heavy.lock/run-heavy.
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${1:-/var/tmp/byc-linear-20261104}
B=${R12_HEAD_BIN:-/var/tmp/byc-linear-20261104/target/release/examples/prune_forward_bench}
mkdir -p "$O/heads"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 ZENSIM_MAX_TIER=v3
python3 - "$O" <<'PY' > "$O/heads/jobs.tsv"
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for name,bake in json.load(open(p/'MODELS.json')).items():
    s=name.rsplit('_s',1)[1]
    a=Path.home()/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin'
    print(name,bake,a,sep='\t')
PY
while IFS=$'\t' read -r name bake baseline; do
  printf '%s heads-%s\n' "$(date -u +%FT%TZ)" "$name" > "$O/STAGE"
  "$B" --inspect "$bake" "$O/heads/$name.json"
  "$B" --heads "$baseline" "$bake" 256 "$O/heads/timing-$name.json" > "$O/heads/timing-$name.log" 2>&1
  printf '%s head-timing %s\n' "$(date -u +%FT%TZ)" "$name" >> "$O/PROGRESS"
done < "$O/heads/jobs.tsv"
printf '%s heads-complete\n' "$(date -u +%FT%TZ)" > "$O/STAGE"
