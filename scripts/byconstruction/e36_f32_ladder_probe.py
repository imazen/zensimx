"""Does f64 var_loss order the rungs that f32 collapses? Per ladder (reference, codec, mode_key) sorted by q,
adjacent rungs whose var_loss differs in f64 but is equal in f32: does f64's order agree with q's order
(higher q = less loss) and with butteraugli_pnorm3's order?"""
import sys, json, random, collections, io
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.csv as pc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gauntlet_features as gf
ledger, archive, index, n_jobs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
COLS = [int(c[1:]) for c in sys.argv[5].split(',')]
rows = [r for r in pq.read_table(sorted(ledger.glob('*.parquet'))).to_pylist() if r['status'] == 'done']
uniq = {}
for r in rows: uniq.setdefault(r['job_id'], r)
random.seed(20261101); jobs = random.sample(sorted(uniq.values(), key=lambda r: r['job_id']), n_jobs)
cells = pq.read_table(index / 'cells.parquet', columns=['reference_sha256', 'distorted_sha256', 'codec', 'mode_key', 'q', 'butteraugli_pnorm3']).to_pandas()
cells = cells.drop_duplicates(['reference_sha256', 'distorted_sha256']).set_index(['reference_sha256', 'distorted_sha256'])
recs = []
for r in jobs:
    ref = json.loads(r['knob_tuple_json'])['source_sha256']
    res = json.loads(gf.read_object(archive, r['output_sha']))
    csv = gf.read_object(archive, res['sidecars']['features.csv'])
    t = pc.read_csv(io.BytesIO(csv), convert_options=pc.ConvertOptions(column_types={f'f{c}': pa.float64() for c in COLS}))
    X = np.column_stack([t[f'f{c}'].to_numpy() for c in COLS])
    for c in res['cells']:
        try: m = cells.loc[(ref, c['evif_sha256'])]
        except KeyError: continue
        recs.append((ref, m['codec'], m['mode_key'], int(m['q']), float(m['butteraugli_pnorm3']), X[c['row_id']]))
lad = collections.defaultdict(list)
for ref, codec, mode, q, ba, x in recs: lad[(ref, codec, mode)].append((q, ba, x))
stats = {c: collections.Counter() for c in COLS}
for k, v in lad.items():
    v.sort(key=lambda t: t[0])
    for (q1, b1, x1), (q2, b2, x2) in zip(v, v[1:]):
        if q1 == q2: continue
        for i, c in enumerate(COLS):
            a, b = x1[i], x2[i]
            s = stats[c]; s['adjacent'] += 1
            if a == b: s['equal_f64'] += 1; continue
            if np.float32(a) != np.float32(b): s['distinct_f32'] += 1; continue
            s['collapsed'] += 1
            s['collapsed_f64_agrees_q'] += int(b < a)  # higher q -> less variance loss
            if b1 != b2: s['collapsed_ba_pairs'] += 1; s['collapsed_f64_agrees_ba'] += int((b < a) == (b2 < b1))
            s['collapsed_codec_' + k[1]] += 1
print('jobs', n_jobs, 'views', len(recs), 'ladders', len(lad))
for c in COLS:
    s = stats[c]; n = max(s['collapsed'], 1)
    print(f'f{c}', dict(adjacent=s['adjacent'], equal_f64=s['equal_f64'], distinct_f32=s['distinct_f32'], collapsed=s['collapsed'],
          agrees_q=f"{s['collapsed_f64_agrees_q'] / n:.3f}", agrees_ba=f"{s['collapsed_f64_agrees_ba'] / max(s['collapsed_ba_pairs'], 1):.3f}"),
          {k[16:]: v for k, v in s.most_common() if k.startswith('collapsed_codec_')})
