"""Mean E21 pooled SROCC per seed (0-2) and its spread, per arm and E34 stored arms."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells = Path.home()/'tmp/byc/cells'
def per_seed(arm):
    out = []
    for s in (0, 1, 2):
        v = []
        for d in panel['datasets']:
            f = d['name']; t = np.array(d['target'])
            p = np.array(d['predictions'][arm][s]) if arm in d['predictions'] else pd.read_csv(cells/arm/f'without_{f}_s{s}'/f'pred_{f}.tsv', sep='\t').pred.to_numpy()
            v.append(spearmanr(p, t).statistic)
        out.append(np.mean(v))
    return np.array(out)
for arm in ('base-c', 'w-c', 'mono-thr', 'H1-mono', 'mono-thr-rp32', 'mono-thr-gate', 'mono-thr-mp32', 'mono-thr-fr32-fb5', 'mono-thr-fr32-fb20'):
    v = per_seed(arm)
    print(f"{arm:20s} mean E21 SROCC {v.mean():.4f}  per-seed {np.round(v,4)}  sd {v.std(ddof=1):.4f}")
