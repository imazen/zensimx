"""Where a hinge arm's E36 gain sits: the inversion share of consensus pairs by the gap between their two quality
settings (q_high - q_low, bands (0,2], (2,4], (4,8], (8,16], (16,32], (32,100]), per split, seed means, for a control
and an arm, and each band's share of the arm's total reduction (pairs x difference).

    inversion_by_gap.py PAIRS_SCORED.parquet CONTROL ARM     # model prefixes as in the table: control, p30L, ...
"""
import sys

import pandas as pd
import pyarrow.parquet as pq

path, base, arm = sys.argv[1:4]
cols = pq.read_schema(path).names
inv = {m: sorted(c for c in cols if c.startswith(f'{m}_s') and c.endswith('_inv') and c[len(m) + 2:-4].isdigit())
       for m in (base, arm)}
if not inv[base] or not inv[arm]:
    raise SystemExit(f'no inversion columns for {base} or {arm}')
d = pq.read_table(path, columns=['split', 'q_low', 'q_high', 'scored', *inv[base], *inv[arm]]).to_pandas()
d = d[d.scored]
d['gap'] = pd.cut(d.q_high - d.q_low, [0, 2, 4, 8, 16, 32, 100])
for split in ('fit', 'selection'):
    g = d[d.split == split].groupby('gap', observed=True)
    out = pd.DataFrame({'pairs': g.size()})
    for m in (base, arm):
        out[m] = sum(g[c].mean() for c in inv[m]) / len(inv[m]) * 100
    out['diff'] = out[arm] - out[base]
    out['share_of_reduction'] = (out['diff'] * out.pairs) / (out['diff'] * out.pairs).sum()
    print(f'== {split} ({len(inv[base])} and {len(inv[arm])} seeds), inversion share %')
    print(out.round(3).to_string())
