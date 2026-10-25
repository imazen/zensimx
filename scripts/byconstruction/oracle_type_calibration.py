"""Oracle per-type affine calibration (uses type labels and targets; an upper bound, not a model)."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells = Path.home()/'tmp/byc/cells'
def calib(p, t, cod):
    out = np.empty_like(p)
    for c in np.unique(cod):
        m = cod == c
        A = np.stack([p[m], np.ones(m.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, t[m], rcond=None)
        out[m] = A @ coef
    return out
for arm in ('mono-thr', 'mono-thr-rp32', 'mono-thr-fr32-fb5'):
    for d in panel['datasets']:
        f = d['name']; t = np.array(d['target']); cod = np.array(d['codec'])
        raw, cal = [], []
        for s in (0, 1, 2):
            pa = pd.read_csv(cells/arm/f'without_{f}_s{s}'/f'pred_{f}.tsv', sep='\t').pred.to_numpy()
            pc = np.array(d['predictions']['base-c'][s])
            raw.append(spearmanr(pa, t).statistic - spearmanr(pc, t).statistic)
            cal.append(spearmanr(calib(pa, t, cod), t).statistic - spearmanr(calib(pc, t, cod), t).statistic)
        print(f"{arm:18s} {f:9s} raw Δ={np.mean(raw):+.4f}   per-type-oracle-calibrated Δ={np.mean(cal):+.4f}")
