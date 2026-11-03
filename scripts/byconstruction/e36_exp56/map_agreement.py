"""Experiment 3 outcome 1: per held-out view, Spearman between a model's map and butteraugli's block sums.
'full' = sum_k s_k I_k over every dumped id (the frozen density the map term fits); 'served' = the served density.
Views whose reference sums are all equal are skipped; `views` counts the views with a defined correlation.
With TARGET_DIR the reference is another map's block sums on the same grid, `TARGET_DIR/<key>.<SUFFIX>.f32` (R4: DVIFM's,
written by dvifmish `examples/block_sums.rs` as `<key>.dvifm_sum.f32`), in place of the dump's butteraugli sums.
    map_agreement.py DUMP_JSONL DUMP_DIR [TARGET_DIR [SUFFIX]]"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
jl, d = sys.argv[1], sys.argv[2]
td = sys.argv[3] if len(sys.argv) > 3 else d
suffix = sys.argv[4] if len(sys.argv) > 4 else ('dvifm_sum' if len(sys.argv) > 3 else 'butter_sum')
full, served = [], []
for l in open(jl):
    j = json.loads(l); k = j['key']; nb = j['blocks'][0] * j['blocks'][1]; n = len(j['ids'])
    I = np.fromfile(f'{d}/{k}.integrands.f32', '<f4').reshape(n, nb).astype(float)
    b = np.fromfile(f'{td}/{k}.{suffix}.f32', '<f4')
    if b.size != nb:
        raise SystemExit(f'{k}: {b.size} reference sums for {nb} blocks')
    if b.std() == 0 or nb < 4:
        continue
    full.append(spearmanr(np.array(j['sensitivities']) @ I, b).statistic)
    served.append(spearmanr(np.fromfile(f'{d}/{k}.served.f32', '<f4'), b).statistic)
f, s = np.array(full), np.array(served)
out = dict(views=int(np.isfinite(f).sum()), full_median=float(np.nanmedian(f)), full_p10=float(np.nanquantile(f, 0.1)),
           served_median=float(np.nanmedian(s)), served_p10=float(np.nanquantile(s, 0.1)))
print(json.dumps(out if td == d else dict(out, reference=f'{td}/*.{suffix}.f32')))
