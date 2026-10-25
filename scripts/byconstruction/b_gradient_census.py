"""Gradient-sign census of a plain-N bake on real rows: which inputs (and, through the fx1 products' chain rule, which
raw damage features) raise the score when they increase?

  python3 b_gradient_census.py INSPECT.json DATASET [DATASET...]

INSPECT.json is `zenpredict-inspect <bake> --weights` output (zenanalyze). Datasets are E21 tables under ROOT.
Exploration record: benchmarks/byconstruction_2026-10-22.md.
"""
import json, numpy as np, pyarrow.parquet as pq, sys
ROOT = '/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/wide/main/real'
d = json.load(open(sys.argv[1]))
sys.argv = sys.argv[1:]
L = d['layers']; H = 128
W1 = np.array(L[0]['weights'], dtype=np.float64).reshape(820, H); b1 = np.array(L[0]['biases']); w2 = np.array(L[1]['weights'])
scale = np.array(d['scaler_scale'])
der = [l.split() for l in d['metadata'][6]['value_text'].splitlines()[1:]]
ids = sorted({int(t) for row in der for t in row[1:]})
def inputs(path):
    tab = pq.read_table(path, columns=[f'f{i}' for i in ids])
    cols = {i: tab[f'f{i}'].to_numpy().astype(np.float64) for i in ids}
    X = np.stack([cols[int(r[1])] if r[0]=='in' else cols[int(r[1])]*cols[int(r[2])] for r in der], axis=1)
    return X / scale
names = {}
# v2 slot names
v2 = "ssim_mean ssim_dev2 ssim_dev4 art det mse hf_gain hf_loss hf_mag_loss ssim_soft_peak art_soft_peak det_soft_peak masked_ssim masked_art masked_det masked_mse iw_ssim iw_art iw_det iw_mse pjnd_transducer pjnd_fragility gms transducer_lowk transducer_highk blockiness ringing banding edge_width".split()
def fname(i):
    if i < 156:
        s, rem = divmod(i, 39); c, k = divmod(rem, 13); return f"basic.s{s}.{'XYB'[c]}.slot{k}"
    if 372 <= i < 372+29*12:
        cell, k = divmod(i-372, 29); s, c = divmod(cell, 3); return f"v2.s{s}.{'XYB'[c]}.{v2[k] if k < len(v2) else k}"
    return f"f{i}"
def label(r):
    return fname(int(r[1])) if r[0]=='in' else fname(int(r[1]))+'×frag'
out = {}
for ds in sys.argv[1:]:
    X = inputs(f'{ROOT}/{ds}.parquet')
    Z = X @ W1 + b1
    A = (Z > 0).astype(np.float64)          # rows x H
    G = (A * w2) @ W1.T                       # d raw / d x_std, rows x 820  (raw = pin + sum w2 relu)
    pos = (G > 1e-12).mean(axis=0); neg = (G < -1e-12).mean(axis=0)
    out[ds] = (pos, neg, G.mean(axis=0), X.mean(axis=0))
    print(ds, 'rows', X.shape[0], 'active units/row mean', A.sum(1).mean().round(1),
          '| inputs raising score on >50% rows:', int((pos>0.5).sum()), 'on >90%:', int((pos>0.9).sum()),
          '| rows with >=1 score-raising input:', round(float(((G>1e-12).sum(1)>0).mean()),3),
          '| mean share of inputs raising per row:', round(float((G>1e-12).mean()),3))
# consistently reversed inputs on both datasets, ranked by mean positive contribution
pos_k, _, gk, xk = out['kadid']; pos_t, _, gt, xt = out['tid2013']
rank = np.argsort(-(gk*xk + gt*xt))
print('\ntop inputs by mean score-RAISING contribution (grad*mean input), kadid+tid:')
for k in rank[:25]:
    print(f"  {label(der[k]):40s} pos_kadid={pos_k[k]:.2f} pos_tid={pos_t[k]:.2f} contrib_k={gk[k]*xk[k]:+.3f} contrib_t={gt[k]*xt[k]:+.3f}")
from collections import Counter
fam = Counter()
for k in range(820):
    if pos_k[k] > 0.5 and pos_t[k] > 0.5:
        nm = label(der[k]); fam[nm.split('.')[-1].replace('×frag','') + ('×frag' if '×frag' in nm else '')] += 1
print('\nfamilies of inputs B uses as beneficial on >50% of rows of BOTH sets:', fam.most_common(30))

print('\n=== chain rule: d raw / d (raw damage feature d_k) = g_direct/s_direct + g_product*frag/s_product ===')
direct_pos = {}
for k, r in enumerate(der):
    if r[0] == 'in': direct_pos[int(r[1])] = k
prod_pos = {int(r[1]): (k, int(r[2])) for k, r in enumerate(der) if r[0] == 'product'}
for ds in sys.argv[1:]:
    path = f'{ROOT}/{ds}.parquet'
    tab = pq.read_table(path, columns=[f'f{i}' for i in ids])
    cols = {i: tab[f'f{i}'].to_numpy().astype(np.float64) for i in ids}
    X = inputs(path); Z = X @ W1 + b1; A = (Z > 0).astype(np.float64); G = (A * w2) @ W1.T
    feats = sorted(direct_pos)
    Gd = np.stack([G[:, direct_pos[i]] / scale[direct_pos[i]] + (G[:, prod_pos[i][0]] * cols[prod_pos[i][1]] / scale[prod_pos[i][0]] if i in prod_pos else 0) for i in feats], axis=1)
    pos = (Gd > 0).mean(0)
    print(f"{ds}: damage features={len(feats)}; raising score on >50% rows: {(pos>0.5).sum()}, >90%: {(pos>0.9).sum()}; "
          f"mean share of damage features raising score per row: {(Gd>0).mean():.3f}; rows with any: {((Gd>0).sum(1)>0).mean():.3f}")
