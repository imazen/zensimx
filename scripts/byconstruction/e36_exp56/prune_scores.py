"""E36 experiment 6: per-row within-reference discordance of the human fit table under the F3 base-fit seed-0 bake.

d_i = sum_j |y_i - y_j| * [sign(p_i - p_j) != sign(y_i - y_j)] / sum_j |y_i - y_j|, over rows j of row i's reference
with y_j != y_i (y = human_score, p = bake score; ties in p count as half). Writes discordance.parquet (row, member_set,
ref, d) and prints the per-source distribution.

    prune_scores.py TABLE BAKE PRED OUT_DIR
"""
import hashlib, json, struct, subprocess, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

table, bake, pred, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
t = pq.read_table(table)
keys = pq.read_table(table.with_suffix('.keys.parquet'))
fcols = [c for c in t.column_names if c.startswith('f') and c[1:].isdigit()]
assert fcols == [f'f{i}' for i in range(len(fcols))]
X = np.column_stack([t[c].to_numpy(zero_copy_only=False) for c in fcols]).astype('<f4')
ff = out / 'features.f32'
with open(ff, 'wb') as f:
    f.write(struct.pack('<II', X.shape[1], X.shape[0])); f.write(X.tobytes())
r = subprocess.run([pred, '--bake', str(bake), '--features-file', str(ff)], capture_output=True, text=True, check=True)
p = np.array([float(v) for v in r.stdout.split()]); ff.unlink()
assert len(p) == len(X) and np.isfinite(p).all()
y = t['human_score'].to_numpy(); ref = np.array(t['ref_basename'].to_pylist())
d = np.zeros(len(y))
for rf in np.unique(ref):
    idx = np.flatnonzero(ref == rf)
    dy = y[idx][:, None] - y[idx][None, :]; dp = p[idx][:, None] - p[idx][None, :]
    w = np.abs(dy)
    wrong = np.where(dp == 0, 0.5, (np.sign(dp) != np.sign(dy)).astype(float)) * (dy != 0)
    d[idx] = (w * wrong).sum(1) / np.maximum(w.sum(1), 1e-12)
member = keys['member_set'].to_pylist()
pq.write_table(pa.table({'row': np.arange(len(y)), 'member_set': member, 'ref': ref, 'score': p, 'd': d}), out / 'discordance.parquet')
meta = dict(table=str(table), table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(), bake=str(bake),
            bake_sha256=hashlib.sha256(bake.read_bytes()).hexdigest(), rows=len(y),
            quantiles={q: float(np.quantile(d, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)})
(out / 'discordance.meta.json').write_text(json.dumps(meta, indent=1) + '\n')
print(json.dumps(meta['quantiles']))
m = np.array(member)
for s in sorted(set(member)):
    v = d[m == s]
    print(f'{s:14s} n={len(v):5d} mean={v.mean():.3f} top10%-share={np.mean(v >= np.quantile(d, 0.9)):.3f}')
