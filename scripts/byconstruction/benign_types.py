"""Post-hoc E21 pooled-SROCC delta with benign global-change distortion types removed (KADID brighten, darken, mean shift,
saturation 1/2, contrast change, non-eccentricity patch; TID2013 mean shift, contrast change, saturation change)."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells = Path.home()/'tmp/byc/cells'
# label = type - 1 (sorted two-digit type codes)
BENIGN = {'kadid': {16, 17, 18, 7, 8, 25, 20}, 'tid2013': {16, 17, 18}}
for arm in ('mono-thr', 'mono-thr-rp32'):
    for d in panel['datasets']:
        f = d['name']
        if f not in BENIGN: continue
        t = np.array(d['target']); cod = np.array(d['codec']) + 1
        keep = ~np.isin(cod, list(BENIGN[f]))
        da, dk = [], []
        for s in (0, 1, 2):
            pa = pd.read_csv(cells/arm/f'without_{f}_s{s}'/f'pred_{f}.tsv', sep='\t').pred.to_numpy()
            pc = np.array(d['predictions']['base-c'][s])
            da.append(spearmanr(pa, t).statistic - spearmanr(pc, t).statistic)
            dk.append(spearmanr(pa[keep], t[keep]).statistic - spearmanr(pc[keep], t[keep]).statistic)
        print(f"{arm:14s} {f:8s} all types Δ={np.mean(da):+.4f}  | without benign types ({(~keep).sum()} rows removed) Δ={np.mean(dk):+.4f}")
