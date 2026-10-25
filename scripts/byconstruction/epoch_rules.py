"""Pooled SROCC per E21 dataset under the last-epoch rule vs refit/best.bin (best dev geomean3), per arm, seeds 0-2."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells, best = Path.home()/'tmp/byc/cells', Path.home()/'tmp/byc/best'
for arm in ('base-c', 'mono-thr', 'mono-thr-fr32-fb5', 'mono-thr-fr32-fb20', 'mono-thr-rp32'):
    row = {}
    for d in panel['datasets']:
        f = d['name']; t = np.array(d['target']); L, B = [], []
        for s in (0, 1, 2):
            pl = np.array(d['predictions']['base-c'][s]) if arm == 'base-c' else pd.read_csv(cells/arm/f'without_{f}_s{s}'/f'pred_{f}.tsv', sep='\t').pred.to_numpy()
            pb = pd.read_csv(best/arm/f'without_{f}_s{s}'/f'pred_{f}.tsv', sep='\t').pred.to_numpy()
            L.append(spearmanr(pl, t).statistic); B.append(spearmanr(pb, t).statistic)
        row[f] = (np.mean(L), np.mean(B))
    mean_l = np.mean([v[0] for v in row.values()]); mean_b = np.mean([v[1] for v in row.values()])
    print(f"{arm:20s} mean last={mean_l:.4f} best={mean_b:.4f} ({mean_b-mean_l:+.4f}) | " + " ".join(f"{k}:{v[0]:.4f}->{v[1]:.4f}" for k, v in row.items()))
