"""Registered primary: selection-split inversion share, mean of seeds 0-2, arm minus control, 95% bootstrap over the
40 selection sources (10,000 draws, seed 20261101)."""
import json, sys, numpy as np, pyarrow.parquet as pq
pairs, selection, out = sys.argv[1:4]
origin = {}
for line in open(selection):
    s = json.loads(line)['source']
    origin[s['source_sha256']] = json.loads(s['identity'])['origin']
import os
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]  # SEEDS=0 for a seed-0 screen
CONTROL = os.environ.get('CONTROL', 'control')  # the arm every other arm is compared with
arms = {a: [f"{a.replace('-', '_')}_s{s}" for s in SEEDS] for a in (sys.argv[4].split(',') if len(sys.argv) > 4 else ['control', 'pool-1', 'pool-10'])}
# R13's owner cut retains three frozen control seeds and one candidate seed.
# Only inventory changes; the registered seed mean and source bootstrap below do not.
if 'ARM_SEEDS' in os.environ:
    inventory = json.loads(os.environ['ARM_SEEDS'])
    if set(inventory) != set(arms) or any(not isinstance(v,list) or not v
            or any(type(s) is not int or s < 0 for s in v) or len(v)!=len(set(v))
            for v in inventory.values()):
        raise ValueError('arm seed inventory must bind every arm to unique nonnegative seeds')
    arms = {a:[f"{a.replace('-', '_')}_s{s}" for s in inventory[a]] for a in arms}
cols = ['reference_sha256', 'split', 'scored'] + [f'{m}_inv' for v in arms.values() for m in v]
p = pq.read_table(pairs, columns=cols).to_pandas()
p = p[(p.split == 'selection') & p.scored]
p['origin'] = p.reference_sha256.map(origin)
assert p.origin.notna().all()
g = p.groupby('origin')
n = g.size()
origins = n.index.to_numpy(); N = n.to_numpy(dtype=float)
K = {m: g[f'{m}_inv'].sum().reindex(origins).to_numpy(dtype=float) for v in arms.values() for m in v}
print('selection sources', len(origins), 'pairs', int(N.sum()))
def share(w, arm):  # mean over seeds of weighted inversion share
    return np.mean([(w @ K[m]) / (w @ N) for m in arms[arm]], axis=0)
rng = np.random.default_rng(20261101)
W = rng.multinomial(len(origins), np.full(len(origins), 1 / len(origins)), size=10000).astype(float).T  # origins x draws
res = {}
base = share(np.ones(len(origins)), CONTROL)
ctrl_b = np.mean([(K[m] @ W) / (N @ W) for m in arms[CONTROL]], axis=0)
for arm in [a for a in arms if a != CONTROL]:
    est = share(np.ones(len(origins)), arm) - base
    arm_b = np.mean([(K[m] @ W) / (N @ W) for m in arms[arm]], axis=0)
    d = arm_b - ctrl_b
    lo, hi = np.quantile(d, [0.025, 0.975])
    seeds = {m: float(K[m].sum() / N.sum()) for m in arms[arm]}
    res[arm] = dict(diff=float(est), ci95=[float(lo), float(hi)], arm_share=float(share(np.ones(len(origins)), arm)), seeds=seeds)
res[CONTROL] = dict(share=float(base), seeds={m: float(K[m].sum() / N.sum()) for m in arms[CONTROL]})
res['meta'] = dict(sources=len(origins), pairs=int(N.sum()), draws=10000, seed=20261101, unit='selection source (origin)')
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=1))
