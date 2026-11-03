"""Experiment 3 outcome 1: per held-out view, Spearman between a model's map and butteraugli's block sums.
'full' = sum_k s_k I_k over every dumped id (the frozen density the map term fits); 'served' = the served density.
    map_agreement.py DUMP_JSONL DUMP_DIR"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
jl, d = sys.argv[1], sys.argv[2]
full, served = [], []
for l in open(jl):
    j = json.loads(l); k = j['key']; nb = j['blocks'][0] * j['blocks'][1]; n = len(j['ids'])
    I = np.fromfile(f'{d}/{k}.integrands.f32', '<f4').reshape(n, nb).astype(float)
    b = np.fromfile(f'{d}/{k}.butter_sum.f32', '<f4')
    if b.std() == 0 or nb < 4:
        continue
    full.append(spearmanr(np.array(j['sensitivities']) @ I, b).statistic)
    served.append(spearmanr(np.fromfile(f'{d}/{k}.served.f32', '<f4'), b).statistic)
f, s = np.array(full), np.array(served)
print(json.dumps(dict(views=len(f), full_median=float(np.nanmedian(f)), full_p10=float(np.nanquantile(f, 0.1)),
                      served_median=float(np.nanmedian(s)), served_p10=float(np.nanquantile(s, 0.1)))))
