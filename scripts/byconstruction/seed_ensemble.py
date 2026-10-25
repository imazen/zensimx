"""E21 pooled SROCC of 3-seed ensembles (mean raw prediction over seeds 0-2), arms vs base-c."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells = Path.home()/'tmp/byc/cells'
def ens(arm, d):
    f = d['name']
    ps = [np.array(d['predictions'][arm][s]) if arm in d['predictions'] else pd.read_csv(cells/arm/f'without_{f}_s{s}'/f'pred_{f}.tsv', sep='\t').pred.to_numpy() for s in (0,1,2)]
    return np.mean(ps, 0)
rows = {}
for arm in ('base-c', 'mono-thr', 'mono-thr-mp32', 'mono-thr-fr32-fb20'):
    rows[arm] = {d['name']: spearmanr(ens(arm, d), np.array(d['target'])).statistic for d in panel['datasets']}
for arm, r in rows.items():
    m = np.mean(list(r.values())); b = np.mean(list(rows['base-c'].values()))
    print(f"{arm:20s} 3-seed ensemble mean SROCC {m:.4f} (Δ vs base-c ensemble {m-b:+.4f}) " + " ".join(f"{k}:{v:.4f}" for k,v in r.items()))
