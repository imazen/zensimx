"""Experiment 3 step 3 held-out sample, selection split (seed 20261104, bank row ids kept): E36 fit-split views (sizes 128-1024), decoded PNGs pulled from E36's archives by
the rev7 index (sha-verified), references from the E36 rev7 references folder. Writes pairs.tsv for map_dump."""
import json, hashlib, sys, os, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
IDX = Path.home() / 'tmp/e36feat/full'; ROOT = Path('/mnt/tower/output/zensim'); OUT = Path(sys.argv[1]); N = int(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True); (OUT / 'views').mkdir(exist_ok=True)
c = pq.read_table(IDX / 'cells.parquet', columns=['codec', 'content', 'size', 'q', 'split', 'reference_sha256', 'distorted_sha256', 'distorted_pixels_sha256', 'reference_pixels_sha256']).to_pandas()
c = c[(c.split == 'selection') & c['size'].between(128, 1024)].drop_duplicates(['reference_sha256', 'distorted_sha256'])
c['qb'] = np.where(c.q <= 20, 'q0-20', np.where(c.q <= 60, 'q22-60', 'q62-100'))
rng = np.random.default_rng(20261104)
strata = c.groupby(['codec', 'content', 'qb']); per = max(1, N // strata.ngroups)
pick = c.loc[np.concatenate([rng.choice(g.index, size=min(per, len(g)), replace=False) for _, g in strata])]
ref_path = {}
for line in open(IDX / 'selection.jsonl'):
    s = json.loads(line)['source']; ref_path[s['source_sha256']] = s['image_path']
loc = {}
want = set(pick.distorted_sha256)
for f in sorted((IDX / 'index').glob('*.tsv')):
    for line in open(f):
        sha, _, off, size, arch = line.rstrip('\n').split('\t')[:5]
        if sha in want and sha not in loc: loc[sha] = (int(off), int(size), arch)
rows = []; skipped = 0
for _, r in pick.iterrows():
    if r.distorted_sha256 not in loc or r.reference_sha256 not in ref_path: skipped += 1; continue
    off, size, arch = loc[r.distorted_sha256]; p = ROOT / arch
    if arch.endswith('.zst'):
        import zstandard  # noqa
    if arch.endswith('.zst'):
        import subprocess; data = subprocess.run(['zstd', '-dc', str(p)], capture_output=True, check=True).stdout[off:off + size]
    else:
        with open(p, 'rb') as h: h.seek(off); data = h.read(size)
    if hashlib.sha256(data).hexdigest() != r.distorted_sha256: print('HASH MISMATCH', r.distorted_sha256, arch); continue
    dst = OUT / 'views' / f'{r.distorted_sha256}.png'; dst.write_bytes(data)
    key = f'{r.codec}-{r["size"]}-{r.q}-{r.distorted_sha256[:8]}'
    rows.append((key, ref_path[r.reference_sha256], str(dst), f'{r.codec}/{r.content}/{r.qb}', r.distorted_pixels_sha256, r.reference_pixels_sha256))
with open(OUT / 'pairs.tsv', 'w') as f:
    f.write('key\tref_path\tdist_path\tgroup\tdistorted_pixels_sha256\treference_pixels_sha256\n'); [f.write('\t'.join(x) + '\n') for x in rows]
print(len(rows), 'pairs;', skipped, 'skipped (no archive entry or reference path);', collections.Counter(g.split('/')[0] for *_, g in rows))
