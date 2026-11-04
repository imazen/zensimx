"""R13 frozen-bake capacity, TRAIN development and paired head measurements.

All forward/ablation/hidden execution uses the pinned native audit owners.
Python classifies exported weights and summarizes observations only.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

import r13_width as owner

O = owner.O


def models():
    return json.loads((O/'MODELS.json').read_text())


def capacity():
    if owner.sha(owner.EXPORT) != 'd9e1ca3e9624036f567c29dbc5cd79668a697fe4a63b4bb33c9cf30fdaed2f6f':
        raise ValueError('native weight-audit exporter changed')
    all_rows = {}
    for bank, (path, digest) in owner.BANKS.items():
        if owner.sha(path) != digest:
            raise ValueError('audit bank changed')
        for name, bake in models().items():
            out = O/'capacity'/bank/name
            owner.job(f'audit-{bank}-{name}', [owner.EXPORT,'audit',bake,'--bank',path,'--out',out/'audit'])
            owner.job(f'ablation-{bank}-{name}', [owner.EXPORT,'mlp-bank',bake,'--bank',path,'--out',out/'bank'])
            d = json.loads((out/'audit/audit.json').read_text())
            w = json.loads((out/'audit/weights.json').read_text())
            first,last = w['layers']
            width = first['out_dim']
            encoder = np.array(first['weights'],dtype=np.float32).reshape(-1,width)
            output = np.array(last['weights'],dtype=np.float32)
            dead = (np.max(abs(encoder),axis=0)==0) & (np.array(first['biases'])<=0)
            residual = ~dead & (abs(output)<1e-3*max(abs(output)))
            live = ~dead & ~residual
            contribution = {u['unit']:abs(float(output[u['unit']]))*u['mean_abs_activation'] for u in d['hidden_units']}
            total = sum(contribution.values())
            pair_masses = [(pair,sum(contribution[j] for j in pair)) for pair in d['constraints']['max_pairs']
                           if any(live[j] for j in pair)]
            pair_masses.sort(key=lambda row:-row[1])
            s = json.loads((out/'bank/summary.json').read_text())
            effects = sorted((v['mean_abs_delta'] for v in s['input_attribution']),reverse=True)
            if (len(effects)!=492 or s['bank_sha256']!=digest or d['bank_sha256']!=digest
                    or d['sampled_rows']!=d['bank_rows'] or len(s['row_indices'])!=d['bank_rows']):
                raise ValueError('incomplete or changed audit population')
            cumulative = np.cumsum(effects)
            ablation_mass = sum(effects)
            all_rows[f'{bank}/{name}'] = dict(hidden=width,dead=int(dead.sum()),residual=int(residual.sum()),live=int(live.sum()),
                 live_pairs=len(pair_masses), largest_pair=pair_masses[0][0] if pair_masses else None,
                 largest_pair_share=pair_masses[0][1]/total if pair_masses and total else None,
                 concentration={str(p):int(np.searchsorted(cumulative,p*ablation_mass)+1) if ablation_mass>0 else None
                                for p in (.5,.9,.99)},
                 ablation_mass=ablation_mass, rows=d['sampled_rows'],
                 bank_sha256=digest,bake_sha256=owner.sha(bake))
            owner.write(O/'CAPACITY.json',all_rows)
            print(json.dumps({f'{bank}/{name}':all_rows[f'{bank}/{name}']}),flush=True)


def development():
    out = O/'development'
    out.mkdir(exist_ok=True)
    current = models()
    current.update({f'F_s{s}':str(owner.H/f'tmp/e36test/r10/runs/mixF3-s{s}/pack_e059/refit/production-f16.bin') for s in range(3)})
    rows = {}
    for group in ('safesyn','cid22'):
        path = owner.R/f'wide/main/real/{group}_dev.parquet'
        y = np.asarray(pq.read_table(path,columns=['human_score'])['human_score'])
        if not np.isfinite(y).all(): raise ValueError('invalid development targets')
        rows[group] = dict(n=len(y), table_sha256=owner.sha(path),mae={})
        for name,bake in current.items():
            dest = out/f'{group}-{name}.tsv'
            owner.job(f'development-{group}-{name}', [O/'bin/bake_dial_refit','predict','--bake',bake,
                       '--corpus',path,'--score-units','--out',dest])
            p = np.loadtxt(dest,delimiter='\t',skiprows=1,usecols=[1])
            if len(p)!=len(y) or not np.isfinite(p).all(): raise ValueError('invalid predictions')
            rows[group]['mae'][name]=float(np.abs(p-y).mean())
    total=sum(v['n'] for v in rows.values())
    pooled={n:sum(v['mae'][n]*v['n'] for v in rows.values())/total for n in current}
    base_final_dev = {}
    for name in current:
        seed = name.rsplit('_s',1)[1]
        path = (owner.H/f'tmp/e36test/exp5/runs/tid2013_mdid-s{seed}/result.json' if name.startswith('F_') else
                O/'fits'/name.replace('_s','-s')/'base/result.json')
        result=json.loads(path.read_text())
        if result['selection']['selected_epoch'] != 119:
            raise ValueError('base fit did not retain registered final epoch')
        base_final_dev[name]=result['dev_curve']['119']
    owner.write(O/'DEVELOPMENT.json',dict(groups=rows,pooled=pooled,units='stored target score points; no 100x conversion',
                                         base_final_dev_geomean3=base_final_dev,
                                         role='TRAIN development diagnostic; fixed final epoch; no selection'))


def heads():
    if owner.sha(owner.HEAD) != 'aa409c8e689323f7ed0f8ca4f07fa34565d673efa18e26806f98572677af3a33':
        raise ValueError('R12 paired head instrument changed')
    out = O/'heads'
    out.mkdir(exist_ok=True)
    for name,bake in models().items():
        seed = name.rsplit('_s',1)[1]
        a = owner.H/f'tmp/e36test/r10/runs/mixF3-s{seed}/pack_e059/refit/production-f16.bin'
        owner.job(f'head-{name}', [owner.HEAD,'--heads',a,bake,256,out/f'timing-{name}.json'])
    import r12_linear_head_summary
    r12_linear_head_summary.timing(O)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('capacity','development','heads'))
    a=ap.parse_args()
    globals()[a.command]()


if __name__=='__main__':main()
