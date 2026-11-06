"""Registered revision-7 shaped-linear study, dispatched by linear_probe.py.

Reuses admitted tree-study arrays/target mapping, Rust Gram/ZNPR/calibration,
canonical Rust statistics, and existing tree/neural comparison owners.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['TMPDIR'] = str(Path.home()/'tmp')
tempfile.tempdir = os.environ['TMPDIR']
import numpy as np
import pyarrow.parquet as pq
from activation_curriculum import sha, write
from tree_study import weights_and_targets, canonical_ref, RAW_IDS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch, panel_batch_indexed

REPO=Path(__file__).resolve().parents[2]
OLD=Path('/mnt/v/output/zensim/tree-study-20261106/search')
META=OLD.parent/'peers/metadata.parquet'
BASE=Path('/srv/zen-research/human-ssim2-depth-20261106/trial/source-and-report/HumanSSIM2-depth-v1.trial.json')
PREP=BASE.parent.parent/'PREPARED.json'
FAMILIES=('free_raw','free_products','positive_products','positive_shaped_direct','positive_shaped_products')
POSITIVE=('positive_products','positive_shaped_products')
LAMBDAS=(1e-6,1e-4,1e-2)
SEEDS=(20261106,20261107)
HEAVY=Path.home()/'work/claudehints/scripts/run-heavy'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def command(argv, log):
    with Path(log).open('w') as f:
        f.write(json.dumps([str(v) for v in argv])+'\n');f.flush()
        p=subprocess.run([str(v) for v in argv],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'rc={p.returncode}: {log}')


def terms_for(family, declaration):
    positions={v:i for i,v in enumerate(RAW_IDS)}
    direct=[[positions[v],None] for v in declaration['direct']]
    products=[[positions[a],positions[b]] for a,b in declaration['products']]
    if family=='free_raw':return [[i,None] for i in range(len(RAW_IDS))]
    if family=='context':return [[positions[v],None] for v in sorted({b for _,b in declaration['products']})]
    return direct if family=='positive_shaped_direct' else direct+products


def term_matrix(x,terms):
    return np.ascontiguousarray(np.column_stack([x[:,a] if b is None else x[:,a]*x[:,b] for a,b in terms]),dtype=np.float32)


def apply_basis(x,spec):
    t=term_matrix(x,spec['terms'])/np.asarray(spec['scales'],np.float32)
    if spec['knots']:
        k=np.asarray(spec['knots'],np.float32)
        t=np.stack((np.minimum(t,k[:,0]),np.minimum(np.maximum(t-k[:,0],np.float32(0)),k[:,1]-k[:,0]),np.maximum(t-k[:,1],np.float32(0))),axis=2).reshape(len(x),-1)
    return np.ascontiguousarray(t/np.asarray(spec['basis_scales'],np.float32),dtype=np.float32)


def rms(x,w):
    v=np.sqrt(np.einsum('ij,i,ij->j',x.astype(np.float64),w,x.astype(np.float64))/w.sum())
    v[v==0]=1
    return v.astype(np.float32)


def fit_spec(x,ids,w,family,declaration):
    terms=terms_for(family,declaration)
    t=term_matrix(x[ids],terms)
    scales=rms(t,w)
    knots=[]
    if 'shaped' in family:
        t=t/scales
        for col in t.T:
            positive=col[col>0]
            a,b=np.quantile(positive,[.5,.9]).astype(np.float32) if len(positive) else (np.float32(1),np.float32(2))
            if b<=a:b=np.float32(2)*a
            if not 0<a<b or not np.isfinite(b):raise ValueError('invalid deterministic knot widening')
            knots.append([float(a),float(b)])
    spec=dict(schema='shaped-linear-basis-v1',terms=terms,scales=scales.tolist(),knots=knots,
              basis_scales=[1.]*(len(terms)*(3 if knots else 1)))
    b=apply_basis(x[ids],spec)
    spec['basis_scales']=rms(b,w).tolist()
    return spec


def load(root):
    p=json.loads((root/'PLAN.json').read_text())
    d={k:np.load(OLD/'data'/f'{k}.npy',mmap_mode='r') for k in ('raw','y','pop','refs','dev')}
    d.update({k:np.load(root/'data'/f'{k}.npy',mmap_mode='r') for k in ('component','fold','inner','family')})
    d['x']=np.load(root/'data/inputs.npz')['x']
    return p,d


def split_components(component,pop,ids,n,tag):
    names=sorted(set(component[ids]))
    populations=sorted(set(pop[ids]))
    membership={s:np.array([int(np.any((component[ids]==s)&(pop[ids]==p))) for p in populations]) for s in names}
    totals=np.sum(list(membership.values()),axis=0)
    names.sort(key=lambda s:(min(totals[membership[s]>0]),digest([tag,s])))
    count=np.zeros((n,len(populations)))
    assigned={}
    for s in names:
        v=membership[s]
        costs=[np.sum(((count[k]+v)/np.maximum(totals,1))**2-(count[k]/np.maximum(totals,1))**2) for k in range(n)]
        k=min(range(n),key=lambda k:(costs[k],digest([tag,s,k])))
        assigned[s]=k;count[k]+=v
    return assigned


def prepare(root,binary):
    if root.exists():raise FileExistsError(root)
    old=json.loads((OLD/'PLAN.json').read_text())
    pins={str(BASE):'80ec04c1cba1ef851192e63c0a03687a9da019ae483713c978c1dc0daa7ec915',str(PREP):'04963be1bf38e792def29d05fa382a49687614c829265d29561e02d4d0cc85f8',str(META):'3829ec748def4eb549c5a2a85c024a21d4b11151c6afc8c8188c34180d2545e9',**old['inputs'],**{str(OLD/'data'/k):v for k,v in old['data_sha256'].items()}}
    for path,pin in pins.items():
        if sha(path)!=pin:raise ValueError('input changed: '+path)
    meta=pq.read_table(META).to_pydict()
    prep=json.loads(PREP.read_text());offset=0
    for g in prep['groups']:
        keys=pq.read_table(Path(g['path']).with_suffix('.keys.parquet'),columns=['pair_key'])['pair_key'].to_pylist()
        if meta['pair_key'][offset:offset+len(keys)]!=keys:raise ValueError('key order drift')
        offset+=len(keys)
    data={k:np.load(OLD/'data'/f'{k}.npy') for k in ('raw','y','pop','refs','dev')}
    if offset!=len(data['y']):raise ValueError('row mismatch')
    parent={s:s for s in meta['ref_pixels_sha256']}
    def find(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    aliases={};distorted={}
    for i,s in enumerate(meta['ref_pixels_sha256']):
        pop=str(data['pop'][i]);namespace='cid22' if pop.startswith('cid22') else pop
        alias=(namespace,canonical_ref(str(data['refs'][i])).removeprefix('cid22:'))
        if alias in aliases:union(s,aliases[alias])
        aliases[alias]=s
        h=meta['dist_pixels_sha256'][i]
        if h in distorted:union(s,distorted[h])
        distorted[h]=s
    component=np.array([find(s) for s in meta['ref_pixels_sha256']])
    fit=np.flatnonzero(~data['dev']);dev=np.flatnonzero(data['dev'])
    if set(component[fit])&set(component[dev]):raise ValueError('component crosses frozen FIT/development roles')
    outer=split_components(component,data['pop'],fit,3,'outer-v1')
    fold=np.array([outer.get(s,-1) for s in component],dtype=np.int8)
    inner=np.full((3,len(fold)),-1,dtype=np.int8)
    for o in range(3):
        ids=fit[fold[fit]!=o];mapping=split_components(component,data['pop'],ids,2,f'inner-{o}-v1')
        inner[o]=[mapping.get(s,-1) for s in component]
    root.mkdir(parents=True);(root/'data').mkdir();(root/'bin').mkdir();(root/'checks').mkdir()
    for k,v in dict(component=component,fold=fold,inner=inner,family=np.array(meta['codec'])).items():np.save(root/'data'/f'{k}.npy',v)
    # Diagnostic probes depend on FIT inputs, never target labels or development.
    median=np.maximum(np.median(data['raw'][fit],axis=0),np.float32(1e-6))
    rng=np.random.default_rng(20261106)
    small=(rng.uniform(.01,2,(128,576))*median).astype(np.float32);large=small.copy()
    cols=rng.integers(576,size=128);large[np.arange(128),cols]+=median[cols]*np.float32(.25)
    ray=np.array([0,1e-8,1e-6,1e-4,.01,.1,1,10,1e4],np.float32)[:,None]*median
    inputs=np.concatenate([data['raw'],small,large,ray])
    np.savez_compressed(root/'data/inputs.npz',x=inputs)
    binaries={'bake_dial_refit':binary,'panel':Path('/mnt/v/output/zensim/monotone-architectures-20261106/bin/panel'),
              'zensim_mlp_train':Path('/mnt/v/output/zensim/monotone-architectures-20261106/isolated-order-bin/zensim_mlp_train'),
              'corrhead_parity':binary.with_name('corrhead_parity')}
    for name,path in binaries.items():shutil.copy2(path,root/'bin'/name)
    declaration_path=REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json'
    declaration=json.loads(declaration_path.read_text())
    plan=dict(schema='shaped-linear-study-v1',inputs=pins,base_plan=str(BASE),prepared=str(PREP),metadata=str(META),
        original_plan_sha256=sha(OLD/'PLAN.json'),data_sha256={str(p):sha(p) for p in sorted((root/'data').iterdir())},
        binary_sha256={str(root/'bin'/name):sha(root/'bin'/name) for name in binaries},
        driver_sha256=sha(__file__),declaration=declaration,declaration_sha256=sha(declaration_path),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        rows=len(data['y']),fit_components=len(set(component[fit])),development_components=len(set(component[dev])),
        families=FAMILIES,lambdas=LAMBDAS,seeds=SEEDS,workers=4,knots='positive-value q50/q90; equal upper=2*lower; zero column=[1,2]',
        basis_arithmetic='f32 products, division and segments; weighted RMS assembled f64 then rounded f32',
        certificate_scope='regularized f64 objective of supplied PSD Gram, numerical floating point check',
        required=dict(S1=155,S2_pairs=8,S2_calibration=20,S3=60,S4_linear=60,S4_comparators=36,S5_family=48,S5_context=3,S5_influence_max=20),
        comparison=dict(extra=dict(family='extra',features='raw',params=dict(min_samples_leaf=8,max_features=.33)),neural=dict(epochs=120,pairs=5000,widths=[128,64],l2=1e-5)),
        probe_rows=dict(start=len(data['y']),pairs=128,ray=9),created=datetime.now(timezone.utc).isoformat())
    write(root/'PLAN.json',plan)
    counts=[]
    for p in sorted(set(data['pop'])):
        for o in range(3):
            ids=np.flatnonzero((data['pop']==p)&(~data['dev'])&(fold==o))
            counts.append(dict(population=p,outer=o,rows=len(ids),components=len(set(component[ids]))))
    write(root/'FOLDS.json',dict(counts=counts,components={s:dict(outer=outer.get(s,-1),rows=np.flatnonzero(component==s).tolist()) for s in sorted(set(component))}))
    print(json.dumps(dict(stage='prepared',rows=plan['rows'],fit_components=plan['fit_components'],development_components=plan['development_components'])),flush=True)


def numeric_gate(root):
    from scipy.optimize import nnls
    os.environ['ZEN_PANEL_BIN']=str(root/'bin/panel')
    rng=np.random.default_rng(1187);results=[]
    for case in range(4):
        x=rng.uniform(0,2,(73,8)).astype(np.float32);x[:,6]=x[:,2];x[:,7]=0
        if case%2:x[:,3]=x[:,2]+np.float32(1e-5)*x[:,3]
        terms=[[i,None] for i in range(8)]+[[0,1],[2,4]]
        spec=dict(schema='shaped-linear-basis-v1',terms=terms,scales=[1.]*10,knots=[] if case<2 else [[.3,1.4]]*10,basis_scales=[1.]*(10 if case<2 else 30))
        x=np.concatenate([x,np.zeros((1,8),np.float32)])
        b=apply_basis(x,spec).astype(np.float64);w=rng.uniform(.1,2,len(x));target=b@rng.normal(0,.1,b.shape[1])+rng.normal(0,.03,len(x))
        weighted=b*np.sqrt(w/w.sum())[:,None];yt=target*np.sqrt(w/w.sum())
        g=weighted.T@weighted;g=(g+g.T)/2;c=weighted.T@yt;y2=yt@yt
        dest=root/'checks'/f'oracle-{case}';dest.mkdir()
        np.savez_compressed(dest/'inputs.npz',x=x);np.savez_compressed(dest/'gram.npz',g=g,c=c,y2=y2);write(dest/'spec.json',spec)
        for positive in (False,True):
            lam=1e-4;out=dest/str(positive)
            command([root/'bin/bake_dial_refit','linear-study','--inputs',dest/'inputs.npz','--gram',dest/'gram.npz','--spec',dest/'spec.json','--lambda',str(lam),'--out',out]+(['--positive'] if positive else []),dest/f'{positive}.log')
            oracle=nnls(np.vstack([weighted,np.eye(b.shape[1])*np.sqrt(lam)]),np.r_[yt,np.zeros(b.shape[1])],maxiter=10000)[0] if positive else np.linalg.solve(g+lam*np.eye(b.shape[1]),c)
            result=json.loads((out/'RESULT.json').read_text());coef=np.array(result['coefficients'])
            delta=float(np.max(np.abs(coef-oracle)))
            assert delta<1e-7,(case,positive,delta)
            assert result['basis_sha256']==hashlib.sha256(apply_basis(x,spec).tobytes()).hexdigest()
            pred=np.load(out/'predictions.npz')
            assert np.max(np.abs(pred['optimum']-100*(1-b@oracle)))<1e-6
            assert result['identity']==100
            results.append(dict(case=case,positive=positive,coefficient_max_error=delta,certificate=result['certificate']))
    p=panel_batch([('negative',[1,2,3],[3,2,1])],stats='srocc',raw_errors=True)[0]
    assert p['srocc_signed']==-1 and abs(p['mae_raw']-4/3)<1e-6,p
    legacy=panel_batch([('legacy',[1,2,3],[3,2,1])],stats='srocc')[0]
    assert 'mae_raw' not in legacy and legacy['srocc_signed']==-1
    write(root/'checks/NUMERIC_GATE.json',dict(status='PASS',systems=results,signed_raw_panel=p,legacy_panel=legacy))
    print(json.dumps(dict(stage='numeric_gate',status='PASS',systems=len(results))),flush=True)


def mapped_labels(y,pop,maps):
    out=np.empty(len(y))
    for p,m in maps.items():
        ids=np.flatnonzero(pop==p);lo,hi=m['values'][0],m['values'][-1]
        out[ids]=(y[ids]-lo)/(hi-lo)
    return out


def metrics(pred,y,ids,d,maps):
    jobs=[];summary={};mapped=mapped_labels(y,d['pop'],maps)*100
    unmapped=pred.copy()
    for p,m in maps.items():
        sel=d['pop']==p;lo,hi=m['values'][0],m['values'][-1];unmapped[sel]=lo+pred[sel]/100*(hi-lo)
    for p in sorted(set(d['pop'][ids])):
        q=ids[d['pop'][ids]==p]
        jobs.extend([(p+'|pooled','pred','y',q),(p+'|mapped','pred','mapped',q),(p+'|unmapped','unmapped','y',q)])
        for s in sorted(set(d['component'][q])):jobs.append((p+'|source|'+s,'pred','y',q[d['component'][q]==s]))
        for f in sorted(set(d['family'][q])):jobs.append((p+'|family|'+f,'pred','y',q[d['family'][q]==f]))
    rows=panel_batch_indexed(dict(pred=pred,y=y,mapped=mapped,unmapped=unmapped),jobs,stats='srocc',raw_errors=True)
    if any(v['n_dropped'] for v in rows):raise ValueError('unexpected dropped row')
    for row,job in zip(rows,jobs):
        q=job[3]
        if len(q)<2 or np.ptp(pred[q])==0 or np.ptp(y[q])==0:
            row['srocc_signed']=float('nan');row['correlation_status']='undefined_constant_or_singleton'
        else:row['correlation_status']='measured'
    for p in sorted(set(d['pop'][ids])):
        source=[v['srocc_signed'] for v in rows if v['label'].startswith(p+'|source|')]
        valid=[v for v in source if np.isfinite(v)]
        pooled=next(v for v in rows if v['label']==p+'|pooled')['srocc_signed']
        summary[p]=dict(pooled=pooled,within=float(np.mean(valid)) if valid else None,defined_sources=len(valid),sources=len(source),rows=int(np.sum(d['pop'][ids]==p)),mapped_mae=next(v for v in rows if v['label']==p+'|mapped')['mae_raw'],unmapped_mae=next(v for v in rows if v['label']==p+'|unmapped')['mae_raw'])
    human=[(v['pooled']+v['within'])/2 for p,v in summary.items() if p!='cid22_ssim2' and v['within'] is not None and np.isfinite(v['pooled'])]
    return clean(dict(summary=summary,selection_score=float(np.mean(human)) if len(human)==4 else None,panels=rows))


def permuted(y,d,ids,mode,seed):
    out=np.array(y);rng=np.random.default_rng(seed)
    for p in sorted(set(d['pop'][ids])):
        pp=ids[d['pop'][ids]==p]
        groups=[pp] if mode=='population' else [pp[d['family'][pp]==f] for f in sorted(set(d['family'][pp]))]
        for group in groups:out[group]=out[rng.permutation(group)]
    return out


def fit_task(rootstr,task):
    root=Path(rootstr);plan,d=load(root);os.environ['ZEN_PANEL_BIN']=str(root/'bin/panel')
    dest=root/'fits'/task['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
    ids=np.asarray(task['train'],dtype=np.int64);test=np.asarray(task['test'],dtype=np.int64)
    y=permuted(d['y'],d,ids,task['permutation'],task['seed']) if task.get('permutation') else np.array(d['y'])
    w,target,maps=weights_and_targets(y[ids],d['pop'][ids],d['component'][ids],'range',1.)
    spec=fit_spec(d['x'],ids,w,task['family'],plan['declaration']);basis=apply_basis(d['x'],spec)
    b=basis[ids].astype(np.float64);damage=1-target
    if task.get('pairs'):
        a=[];z=[];pair_w=[]
        rng=np.random.default_rng(20261106)
        for p in sorted(set(d['pop'][ids])):
            for s in sorted(set(d['component'][ids][d['pop'][ids]==p])):
                q=np.flatnonzero((d['pop'][ids]==p)&(d['component'][ids]==s))
                # Fixed cyclic offsets after a label-independent shuffle, <=8 partners per row.
                q=rng.permutation(q)
                for step in range(1,min(9,len(q))):
                    a.extend(q.tolist());z.extend(np.roll(q,step).tolist());pair_w.extend([1/(len(q)*min(8,len(q)-1)*len(set(d['component'][ids][d['pop'][ids]==p])))]*len(q))
        a=np.asarray(a);z=np.asarray(z);w=np.asarray(pair_w);b=b[a]-b[z];damage=damage[a]-damage[z]
        np.savez_compressed(dest/'pairs.npz',left=ids[a],right=ids[z],weight=w)
    bw=b*np.sqrt(w/w.sum())[:,None];dw=damage*np.sqrt(w/w.sum());g=bw.T@bw;g=(g+g.T)/2;c=bw.T@dw;y2=float(dw@dw)
    np.savez_compressed(dest/'gram.npz',g=g,c=c,y2=y2)
    if task.get('calibrate'):spec.update(calibration_ids=ids.tolist(),calibration_targets=(target*100).tolist())
    write(dest/'SPEC.json',spec);write(dest/'TASK.json',task)
    write(dest/'INPUTS.json',dict(plan_sha256=sha(root/'PLAN.json'),spec_sha256=sha(dest/'SPEC.json'),gram_sha256=sha(dest/'gram.npz'),maps=maps,train_rows=len(ids),train_components=len(set(d['component'][ids])),test_rows=len(test),permutation=task.get('permutation'),target_sha256=hashlib.sha256(y[ids].tobytes()).hexdigest()))
    expected=hashlib.sha256(basis.tobytes()).hexdigest();results=[]
    for lam in task['lambdas']:
        out=dest/f'lambda-{lam:g}'
        if not (out/'RESULT.json').exists():
            command([HEAVY,'--mem','4G','--jobs','1','--',root/'bin/bake_dial_refit','linear-study','--inputs',root/'data/inputs.npz','--gram',dest/'gram.npz','--spec',dest/'SPEC.json','--lambda',str(lam),'--out',out]+(['--positive'] if task['family'].startswith('positive') else []),dest/f'lambda-{lam:g}.log')
        result=json.loads((out/'RESULT.json').read_text())
        if result['basis_sha256']!=expected:raise ValueError('independent/native basis mismatch: '+task['id'])
        pred=np.load(out/'predictions.npz');n=len(d['y']);native=pred['native'];optimum=pred['optimum']
        reference=100*(1-basis.astype(np.float64)@np.asarray(result['coefficients']))
        check=float(np.max(np.abs(optimum[:n]-reference[:n])))
        if check>1e-6 or not np.allclose(optimum[n:],reference[n:],rtol=1e-12,atol=1e-6):raise ValueError('independent f64 prediction mismatch')
        probe=native[n:];delta=probe[128:256]-probe[:128]
        probes=dict(identity=float(probe[256]),max_coordinate_increase=float(max(0,delta.max())),violations_over_1e_5=int(np.sum(delta>1e-5)),finite=bool(np.isfinite(probe).all()),ray=probe[256:].tolist())
        if not probes['finite'] or probes['identity']!=100:raise ValueError('finite/identity probe failed')
        if task['family'].startswith('positive') and probes['violations_over_1e_5']:raise ValueError('coordinate gate failed')
        report=dict(task=task['id'],family=task['family'],lambda_=lam,certificate=result['certificate'],calibration=result['calibration'],basis_sha256=expected,bake_sha256=sha(out/'model.bin'),probes=probes,
                    maximum_f32_scientific_difference=float(np.max(np.abs(native[:n]-optimum[:n]))),f64_independent_max_difference=check,
                    train=metrics(native[:n],y,ids,d,maps),test=metrics(native[:n],d['y'],test,d,maps),train_actual=metrics(native[:n],d['y'],ids,d,maps) if task.get('permutation') else None)
        if (out/'calibrated.npz').exists():
            cp=np.load(out/'calibrated.npz')['native'][:n]
            report['calibrated']=dict(train=metrics(cp,y,ids,d,maps),test=metrics(cp,d['y'],test,d,maps))
        write(out/'REPORT.json',clean(report));results.append(dict(lambda_=lam,path=str(out),test=report['test']['selection_score']))
    complete=dict(id=task['id'],fits=len(results),results=results)
    write(dest/'COMPLETE.json',complete);return complete


def task(name,family,train,test,lambdas=(1e-4,),**kwargs):
    return dict(id=name,family=family,train=np.asarray(train).tolist(),test=np.asarray(test).tolist(),lambdas=list(lambdas),**kwargs)


def run_tasks(root,tasks,stage,workers):
    matrix=root/f'{stage}.json'
    if matrix.exists():
        if json.loads(matrix.read_text())!=tasks:raise ValueError('registered matrix changed: '+stage)
    else:write(matrix,tasks)
    status=dict(stage=stage,coordinator_pid=os.getpid(),required=sum(len(t['lambdas']) for t in tasks),completed=0,failures=[],last_check=datetime.now(timezone.utc).isoformat())
    write(root/'PROGRESS.json',status)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(fit_task,str(root),t):t for t in tasks}
        for f in as_completed(pending):
            t=pending[f]
            try:r=f.result();status['completed']+=r['fits']
            except Exception as e:status['failures'].append(dict(id=t['id'],error=str(e)))
            status['last_check']=datetime.now(timezone.utc).isoformat();write(root/'PROGRESS.json',status)
            print(json.dumps(status),flush=True)
    write(root/f'{stage}-STATUS.json',status)
    if status['failures']:raise RuntimeError(f'{stage}: {len(status["failures"])} failed tasks; preserved')


def s1(root,workers):
    _,d=load(root);fit=np.flatnonzero(~d['dev']);dev=np.flatnonzero(d['dev']);tasks=[]
    for family in FAMILIES:
        for o in range(3):
            train=fit[d['fold'][fit]!=o];test=fit[d['fold'][fit]==o]
            tasks.append(task(f's1-fullselect-{family}-o{o}',family,train,test,LAMBDAS))
            for i in range(2):
                tr=train[d['inner'][o,train]!=i];va=train[d['inner'][o,train]==i]
                tasks.append(task(f's1-inner-{family}-o{o}-i{i}',family,tr,va,LAMBDAS))
    run_tasks(root,tasks,'S1-selection',workers)
    selected=[];refits=[]
    for family in FAMILIES:
        for o in range(4):
            prefix=[f's1-inner-{family}-o{o}-i{i}' for i in range(2)] if o<3 else [f's1-fullselect-{family}-o{k}' for k in range(3)]
            scores={lam:[] for lam in LAMBDAS}
            for p in prefix:
                r=json.loads((root/'fits'/p/'COMPLETE.json').read_text())
                for cell in r['results']:
                    if cell['test'] is None:raise ValueError('undefined selection panel')
                    scores[cell['lambda_']].append(cell['test'])
            means={lam:float(np.mean(v)) for lam,v in scores.items()};chosen=max(LAMBDAS,key=lambda lam:(means[lam],lam))
            name=f's1-outer-{family}-o{o}' if o<3 else f's1-final-{family}'
            tr=fit[d['fold'][fit]!=o] if o<3 else fit;te=fit[d['fold'][fit]==o] if o<3 else dev
            selected.append(dict(family=family,outer=o if o<3 else None,scores=scores,means=means,lambda_=chosen))
            refits.append(task(name,family,tr,te,(chosen,),calibrate=True))
    write(root/'S1-SELECTED.json',selected);run_tasks(root,refits,'S1-refits',workers)


def select_sources(d,ids,fraction,seed):
    selected=set()
    # Shared components retain all population rows; selection is by hashes, not quality.
    for p in sorted(set(d['pop'][ids])):
        names=sorted(set(d['component'][ids][d['pop'][ids]==p]),key=lambda s:digest([seed,p,str(s)]))
        selected.update(names[:max(1,int(np.ceil(len(names)*fraction)))])
    return ids[np.isin(d['component'][ids],list(selected))]


def diagnostics(root,workers):
    _,d=load(root);fit=np.flatnonzero(~d['dev']);dev=np.flatnonzero(d['dev'])
    groups={}
    pair=[];learning=[];permutation=[];exclusions=[];context=[]
    for o in range(3):
        tr=fit[d['fold'][fit]!=o];te=fit[d['fold'][fit]==o]
        for family in ('free_products','positive_products'):pair.append(task(f's2-pair-{family}-o{o}',family,tr,te,pairs=True))
        for family in POSITIVE:
            learning.append(task(f's3-source-{family}-o{o}-full',family,tr,te))
            for frac in (.25,.5,.75):
                retained=[]
                for seed in SEEDS:
                    ss=select_sources(d,tr,frac,seed);retained.append(ss)
                    learning.append(task(f's3-source-{family}-o{o}-f{frac}-s{seed}',family,ss,te))
                # Match the first source-subset's row count per population, keeping each source.
                rng=np.random.default_rng(20261108);rows=[]
                for p in sorted(set(d['pop'][tr])):
                    q=tr[d['pop'][tr]==p];budget=int(np.sum(d['pop'][retained[0]]==p));sources=sorted(set(d['component'][q]))
                    anchors=[int(rng.choice(q[d['component'][q]==s])) for s in sources]
                    if budget<len(anchors):raise ValueError('row budget cannot retain all sources')
                    rest=np.setdiff1d(q,anchors);rows.extend(anchors);rows.extend(rng.choice(rest,budget-len(anchors),replace=False).tolist())
                learning.append(task(f's3-rows-{family}-o{o}-f{frac}',family,sorted(rows),te))
            for exclusion in ('avif','jpeg','jxl','webp','kadid_01','kadid_20','tid_01','jitter'):
                codecs={'avif':['aom','cld_avif','vis_avif'],'jpeg':['mozjpeg'],'jxl':['libjxl'],'webp':['cld_webp']}.get(exclusion,[exclusion])
                mask=np.isin(d['family'][tr],codecs)
                if not mask.any():raise ValueError('empty registered exclusion: '+exclusion)
                exclusions.append(task(f's5-family-{family}-o{o}-{exclusion}',family,tr[~mask],te,excluded=exclusion))
        for family in FAMILIES:
            for mode in ('population','conditional'):
                for seed in SEEDS:permutation.append(task(f's4-perm-{family}-o{o}-{mode}-s{seed}',family,tr,te,permutation=mode,seed=seed))
        context.append(task(f's5-context-o{o}','context',tr,te))
    for family in ('free_products','positive_products'):pair.append(task(f's2-pair-{family}-full',family,fit,dev,pairs=True))
    groups.update(S2_pairs=pair,S3=learning,S4_linear=permutation,S5_family=exclusions,S5_context=context)
    for stage,tasks in groups.items():run_tasks(root,tasks,stage,workers)


def influence(root,workers):
    _,d=load(root);fit=np.flatnonzero(~d['dev']);dev=np.flatnonzero(d['dev']);tasks=[];selected=[]
    for family in POSITIVE:
        result=json.loads((root/'fits'/f's1-final-{family}'/'COMPLETE.json').read_text())['results'][0]
        pred=np.load(Path(result['path'])/'predictions.npz')['native'][:len(d['y'])]/100
        w,target,_=weights_and_targets(d['y'][fit],d['pop'][fit],d['component'][fit],'range',1.)
        names=set()
        for p in sorted(set(d['pop'][fit])):
            components=sorted(set(d['component'][fit][d['pop'][fit]==p]));mass={}
            for s in components:
                mask=(d['pop'][fit]==p)&(d['component'][fit]==s)
                mass[s]=float(np.sum(w[mask]*(pred[fit][mask]-target[mask])**2))
            chosen=sorted(components,key=lambda s:(-mass[s],s))[:2];names.update(chosen)
            selected.append(dict(family=family,population=p,selected=chosen,residual_mass=mass))
        for i,s in enumerate(sorted(names)):tasks.append(task(f's5-influence-{family}-{i}',family,fit[d['component'][fit]!=s],dev,omitted_component=s))
    write(root/'S5-INFLUENCE-SELECTION.json',selected);run_tasks(root,tasks,'S5_influence',workers)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','gate','s1','diagnostics','influence']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--binary',type=Path,default=Path.home()/'tmp/shaped-linear-target/release/bake_dial_refit');ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args(argv)
    if not 1<=a.workers<=4:raise ValueError('registered maximum four concurrent workers')
    if a.stage=='prepare':prepare(a.root,a.binary)
    elif a.stage=='gate':numeric_gate(a.root)
    else:
        gate=json.loads((a.root/'checks/NUMERIC_GATE.json').read_text())
        if gate['status']!='PASS':raise ValueError('independent numeric admission absent')
        p=json.loads((a.root/'PLAN.json').read_text())
        for path,pin in {**p['binary_sha256'],**p['data_sha256']}.items():
            if sha(path)!=pin:raise ValueError('frozen execution input changed: '+path)
        {'s1':s1,'diagnostics':diagnostics,'influence':influence}[a.stage](a.root,a.workers)


if __name__=='__main__':main()
