#!/usr/bin/env bash
# Complete R13 evidence, excluding rebuildable Cargo output. Caller holds the resource lock.
set -euo pipefail
cd "$(dirname "$0")/../.."
O=${1:-/var/tmp/byc-width-20261104}
D=${2:-/mnt/tower/output/zensim/r13-width-20261104}
df -h /var/tmp /mnt/tower
ssh tower df -h /mnt/cache
python3 -B - "$O" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); rows=[]
excluded={'archive.log','archive-check.txt','ARCHIVE_VERIFIED.json','ARCHIVE_MANIFEST.json','SHA256SUMS',
          'ACTIVE_JOB.json','SUPERVISION.json','status-last.json','PROGRESS.jsonl','QUEUED_W4_BUILD.json'}
for p in sorted(root.rglob('*')):
    rel=p.relative_to(root)
    if rel.parts[0]=='target' or not p.is_file() or rel.name in excluded or rel.name.startswith('archive-'):
        continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    rows.append((str(rel),h.hexdigest(),p.stat().st_size))
(root/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest,size in rows))
pins=json.loads((root/'SOURCE_PINS.json').read_text())
(root/'ARCHIVE_MANIFEST.json').write_text(json.dumps(dict(build_commit=pins['source_commit'],
    files=len(rows),bytes=sum(r[2] for r in rows)),indent=2)+'\n')
PY
mkdir -p "$D"
rsync -aLO --no-owner --no-group --exclude=/target/ --exclude=/archive.log --exclude=/archive-check.txt --exclude='archive-*' \
  --exclude=/ACTIVE_JOB.json --exclude=/SUPERVISION.json --exclude=/status-last.json --exclude=/PROGRESS.jsonl \
  --exclude=/QUEUED_W4_BUILD.json "$O/" "$D/"
rsync -ancLO --no-owner --no-group --itemize-changes --exclude=/target/ --exclude=/archive.log --exclude=/archive-check.txt --exclude='archive-*' \
  --exclude=/ACTIVE_JOB.json --exclude=/SUPERVISION.json --exclude=/status-last.json --exclude=/PROGRESS.jsonl \
  --exclude=/QUEUED_W4_BUILD.json "$O/" "$D/" > "$O/archive-check.txt"
if test -s "$O/archive-check.txt"; then cat "$O/archive-check.txt"; exit 1; fi
python3 -B - "$O" "$D" <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,random,sys
source,dest=map(Path,sys.argv[1:]);items=(source/'SHA256SUMS').read_text().splitlines()
checks=[]
for row in random.Random(20261104).sample(items,3):
    digest,name=row.split('  ',1);h=hashlib.sha256()
    with (dest/name).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=digest:raise ValueError('mirror random readback differs')
    checks.append(dict(path=name,sha256=digest))
receipt=dict(checked=datetime.now(timezone.utc).isoformat(),source=str(source),destination=str(dest),
             full_rsync_checksum_comparison='empty',random_sha256_readbacks=checks,
             manifest_sha256=hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest())
for root in (source,dest):(root/'ARCHIVE_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY
