"""E36 experiment 6 row lists: which human fit rows each pruning arm keeps (parent row indices, ascending)."""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path.home() / 'tmp/e36test/src-exp5/scripts/rev4_featpot'))
from v2_human_role import MEMBER_SOURCE
D = pq.read_table(Path(__file__).parent / 'disc/discordance.parquet').to_pandas()
n = len(D); order = np.lexsort((D.row.to_numpy(), -D.d.to_numpy()))  # most discordant first, ties by row
src = D.member_set.map(MEMBER_SOURCE).to_numpy()
def drop_global(frac): return set(order[:round(frac * n)].tolist())
def drop_per_source(frac):
    out = set()
    for s in np.unique(src):
        o = [i for i in order if src[i] == s]
        out |= set(o[:round(frac * len(o))])
    return out
arms = {'prune-g5': drop_global(0.05), 'prune-g10': drop_global(0.10), 'prune-s10': drop_per_source(0.10),
        'prune-r10': set(np.random.default_rng(20261102).choice(n, round(0.10 * n), replace=False).tolist())}
for name, drop in arms.items():
    keep = [i for i in range(n) if i not in drop]
    Path(f'{name}.json').write_text(json.dumps(dict(label=name.replace('-', ''), keep=keep, dropped=len(drop),
        rule=name, discordance='exp6/disc/discordance.parquet')) + '\n')
    kept_src = {s: int((src[keep] == s).sum()) for s in np.unique(src)}
    print(name, 'kept', len(keep), kept_src)
