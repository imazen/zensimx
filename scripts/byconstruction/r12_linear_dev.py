"""R12 predeclared TRAIN development feasibility check; no EVAL selection."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    root = Path('/var/tmp/byc-fchroma-20261029/r3/w100/root/wide/main/real')
    models = {f'F_s{s}':str(Path.home()/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin') for s in range(3)}
    models.update({f'L1_s{s}':str(out.parent/f'fits/L1-s{s}/refit/production-f16.bin') for s in range(3)})
    rows = {}
    for group in ('safesyn','cid22'):
        table = root/f'{group}_dev.parquet'
        y = np.array(pq.read_table(table,columns=['human_score'])['human_score'])
        if not np.isfinite(y).all(): raise ValueError('nonfinite dev target')
        rows[group] = dict(n=len(y), target_min=float(y.min()),target_max=float(y.max()),mae={})
        for name, model in models.items():
            dest = out/f'{group}_{name}.tsv'
            with (out/f'{group}_{name}.log').open('x') as log:
                subprocess.run(['/var/tmp/byc-hdr-20261104/bin/bake_dial_refit','predict','--bake',model,
                                '--corpus',str(table),'--score-units','--out',str(dest)],check=True,
                               stdout=log,stderr=subprocess.STDOUT)
            p = np.loadtxt(dest,delimiter='\t',skiprows=1,usecols=[1])
            if len(p)!=len(y) or not np.isfinite(p).all(): raise ValueError('invalid complete predictions')
            rows[group]['mae'][name]=float(np.abs(p-y).mean())
    total = sum(r['n'] for r in rows.values())
    pooled = {n:sum(r['mae'][n]*r['n'] for r in rows.values())/total for n in models}
    means = {arm:float(np.mean([pooled[f'{arm}_s{s}'] for s in range(3)])) for arm in ('F','L1')}
    delta=means['L1']-means['F']
    result=dict(groups=rows,pooled=pooled,seed_means=means,delta_score_points=delta,L3_run=bool(delta<=1),
                rule='L1 seed mean pooled teacher development absolute score error <= F + 1 point',
                role='training-side development; no EVAL read')
    (out/'DECISION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
