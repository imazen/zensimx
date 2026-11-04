"""R14 TRAIN-only transform statistics, using the existing greedy screen owner.

No fit or assessment is launched here. The authority is the frozen A base
recipe; validation-only groups are explicitly excluded. Products use IEEE f32.
"""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

X = Path(__file__).resolve().parents[2]
O = Path('/var/tmp/byc-transforms-20261104')
R = Path('/var/tmp/byc-fchroma-20261029/r3/w100/root')
A = Path.home()/'tmp/e36test/r10/runs/mixF3-s0'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def main():
    out=O/'screen'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'SUMMARY.json').exists():
        raise ValueError('screen already frozen; do not overwrite')
    spec=importlib.util.spec_from_file_location('greedy_screen',X/'scripts/v_next/v0_20_feature_transform_greedy_screen.py')
    screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
    authority=json.loads((A/'dataset-authority.json').read_text())
    cell=authority['cells']['full-s0']
    groups=[dict(g) for g in cell['groups'] if g['train_weight']>0]
    assert all(g['val_weight']==0 for g in groups)
    sys.path.insert(0,str(X/'scripts/rev4_featpot'))
    from v2_teacher import coverage_leg
    (out/'coverage').mkdir(exist_ok=True)
    coverage, receipt=coverage_leg(152,out/'coverage',admitted_root=R)
    assert receipt==json.loads((A/'result.json').read_text())['coverage_leg']
    for g in groups:
        if g['name']=='coverage': g['path']=str(coverage)
    declaration=json.loads((X/'scripts/rev4_featpot/color_noise_fx2_declaration.json').read_text())
    direct=declaration['direct'];products=declaration['products']
    assert direct==cell['feature_ids'] and len(direct)==492 and len(products)==492
    ids=sorted(set(direct)|{v for p in products for v in p})
    counts=[pq.ParquetFile(g['path']).metadata.num_rows for g in groups]
    n=sum(counts)
    # A disk-backed, column-major pool permits one complete input at a time.
    pool=np.memmap(out/'TRAIN.f32',dtype=np.float32,mode='w+',shape=(984,n))
    labels=np.memmap(out/'human_score.f64',dtype=np.float64,mode='w+',shape=(n,))
    offset=0;pins=[]
    for g,count in zip(groups,counts):
        p=Path(g['path']); start=offset
        for b in pq.ParquetFile(p).iter_batches(batch_size=4096,columns=['human_score',*[f'f{i}' for i in ids]]):
            cols={i: b.column(b.schema.get_field_index(f'f{i}')).to_numpy().astype(np.float32) for i in ids}
            end=offset+b.num_rows
            y=b.column(b.schema.get_field_index('human_score')).to_numpy().astype(np.float64)
            assert np.isfinite(y).all()
            labels[offset:end]=y
            for k,i in enumerate(direct):pool[k,offset:end]=cols[i]
            for k,(i,j) in enumerate(products):pool[492+k,offset:end]=cols[i]*cols[j]
            offset=end
        assert offset-start==count
        pins.append(dict(name=g['name'],path=str(p),sha256=sha(p),rows=count,train_weight=g['train_weight']))
        print(json.dumps(dict(stage='pool',group=g['name'],completed_rows=offset,total_rows=n)),flush=True)
    pool.flush();labels.flush()
    records=[];arms={a:[] for a in ('T1a','T1b','T1c')}
    columns=direct+list(range(1853,1853+492))
    for k in range(984):
        col=np.asarray(pool[k],dtype=np.float64)
        if not np.isfinite(col).all():raise ValueError(f'nonfinite TRAIN input{k}')
        quantiles=np.percentile(col,[5,25,50,75,90,99]).astype(np.float32)
        p05,p25,p50,p75,p90,p99=map(float,quantiles)
        constant=float(col.var())<1e-10
        negative=int(np.count_nonzero(col<0))
        lmb=None if constant else float(np.float32(screen.fit_yj_lambda(col)))
        t1a=('identity',[]) if constant else ('yeo_johnson',[lmb])
        t1b=('identity',[]) if constant or p50<=0 else ('soft_clip',[p50,p50])
        if constant:
            rec=dict(best_transform='identity',params_csv='',lift=0.)
        else:
            allowed=tuple(t for t in screen.R14_VARIANTS if not (negative and t=='log1p'))
            rec,_=screen.screen_one_feature_all_transforms(k,col,labels,variants=allowed,yj_lambda=lmb)
        t1c=(rec['best_transform'],[float(np.float32(v)) for v in rec['params_csv'].split(',') if v]) if rec['lift']>=0.005 else ('identity',[])
        for a,(token,ps) in [('T1a',t1a),('T1b',t1b),('T1c',t1c)]:
            arms[a].append(dict(feat_idx=columns[k],best_transform=token,params_csv=','.join(map(str,ps)),lift=rec['lift'] if a=='T1c' else 1.))
        records.append(dict(caller_input=k,trainer_column=columns[k],kind='direct' if k<492 else 'product',
            feature=direct[k] if k<492 else products[k-492],negative_rows=negative,minimum=float(col.min()),
            variance=float(col.var()),constant=constant,p05=p05,p25=p25,p50=p50,p75=p75,p90=p90,p99=p99,
            yeo_johnson_lambda=lmb,T1c=rec))
        if k%16==0:print(json.dumps(dict(stage='screen',completed_inputs=k+1,total_inputs=984)),flush=True)
    for arm,rows in arms.items():
        with (out/f'{arm}.tsv').open('x') as f:
            w=csv.DictWriter(f,fieldnames=['feat_idx','best_transform','params_csv','lift'],delimiter='\t');w.writeheader();w.writerows(rows)
    write(out/'INPUTS.json',records)
    lambdas=[r['yeo_johnson_lambda'] for r in records if r['yeo_johnson_lambda'] is not None]
    histogram,edges=np.histogram(lambdas,bins=[-2,-1.5,-1,-.5,0,.5,1,1.5,2])
    summary=dict(registration='5b5a013482db4944042883995ad6892f92244c69',finished=datetime.now(timezone.utc).isoformat(),
        rows=n,inputs=984,target_column='human_score',target_scale=1,train_groups=pins,
        authority_sha256=sha(A/'dataset-authority.json'),declaration_sha256=sha(X/'scripts/rev4_featpot/color_noise_fx2_declaration.json'),
        source_sha256=sha(__file__),screen_owner_sha256=sha(X/'scripts/v_next/v0_20_feature_transform_greedy_screen.py'),
        constant_inputs=sum(r['constant'] for r in records),negative_inputs=[r['caller_input'] for r in records if r['negative_rows']],
        token_counts={a:dict(Counter(r['best_transform'] for r in rows)) for a,rows in arms.items()},
        lambda_histogram=dict(edges=edges.tolist(),counts=histogram.tolist()),
        tables={a:dict(path=str(out/f'{a}.tsv'),sha256=sha(out/f'{a}.tsv')) for a in arms})
    write(out/'SUMMARY.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
