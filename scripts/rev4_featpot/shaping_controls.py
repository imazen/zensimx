#!/usr/bin/env python3
"""Registered inner controls, using the existing quadratic and tree fit owners."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['TMPDIR']=str(Path.home()/'tmp')
import numpy as np
from shaping_completion import REPO,SEEDS,encoded,sha
from shaping_execution import load_registration
from shaping_evaluation import probe_values


DRIVER_SHA256=sha(__file__)

def write(path,value):path.write_bytes(encoded(value))


def run(args):
    from shaped_linear_study import OLD,metrics,weights_and_targets,fit_spec,apply_basis,HEAVY
    from tree_study import RAW_IDS,estimator,native_gate
    plan,cells,partitions=load_registration(args.study)
    if not args.partition.startswith('inner-'):raise ValueError('outer selection gate not yet admitted')
    membership=partitions[args.partition]
    family='extra_trees' if args.tree else 'shaped_additive'
    chosen=[c for c in cells if c['stage']=='inner' and c['family']==family and c['partition']==args.partition]
    if not chosen:raise ValueError('no registered cells')
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('binary drift')
    for path,pin in plan['input_pins'].items():
        if sha(path)!=pin:raise ValueError('frozen input drift: '+path)
    os.environ['ZEN_PANEL_BIN']=str(args.binary_dir/'panel')
    wanted=np.array(sorted(set(membership['train'])|set(membership['test'])),dtype=np.int64)
    pos={int(v):i for i,v in enumerate(wanted)}
    tr=np.array([pos[v] for v in membership['train']]);te=np.array([pos[v] for v in membership['test']])
    raw=np.array(np.load(OLD/'data/raw.npy',mmap_mode='r')[wanted])
    d={k:np.load(OLD/'data'/f'{k}.npy',mmap_mode='r')[wanted] for k in ('y','pop')}
    original=Path(plan['original_plan']).parent
    d.update({k:np.load(original/'data'/f'{k}.npy',mmap_mode='r')[wanted] for k in ('component','family')})
    weight,target,maps=weights_and_targets(d['y'][tr],d['pop'][tr],d['component'][tr],'range',1.)
    declaration=json.loads((REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json').read_text())
    cache=args.study/'control_inputs'/args.partition
    cache.mkdir(parents=True,exist_ok=True)
    linear=None
    if not args.tree:
        x=np.concatenate((raw,probe_values(raw[tr],RAW_IDS,declaration)))
        spec=fit_spec(x,tr,weight,'positive_shaped_products',declaration)
        basis=apply_basis(x,spec)
        b=basis[tr].astype(np.float64);damage=1-target
        bw=b*np.sqrt(weight/weight.sum())[:,None];dw=damage*np.sqrt(weight/weight.sum())
        gram=bw.T@bw;gram=(gram+gram.T)/2;rhs=bw.T@dw
        linear=cache/'linear'
        if linear.exists():raise ValueError('linear input cache exists; preserve/inspect before retry')
        linear.mkdir()
        np.savez_compressed(linear/'inputs.npz',x=x)
        np.savez_compressed(linear/'gram.npz',g=gram,c=rhs,y2=float(dw@dw))
        write(linear/'SPEC.json',spec)
        write(linear/'INPUTS.json',dict(training_indices=membership['train'],validation_indices=membership['test'],
              plan_sha256=sha(args.study/'PLAN.json'),spec_sha256=sha(linear/'SPEC.json'),
              gram_sha256=sha(linear/'gram.npz'),basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),maps=maps))
        del b,bw,gram,rhs
    for cell in chosen:
        dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
        if dest.exists():raise ValueError('control attempt exists: '+str(dest))
        dest.mkdir(parents=True)
        running=dict(cell=cell,pid=os.getpid(),process_role='control_worker',stage='running',
                     started_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=DRIVER_SHA256,
                     plan_sha256=sha(args.study/'PLAN.json'),build_sha256=sha(args.binary_dir/'BUILD.json'))
        write(dest/'RUN.json',running);print(json.dumps({'cell':cell['id'],'pid':os.getpid(),'stage':'running'}),flush=True)
        try:
            if args.tree:
                leaf,fraction=((2,.33),(8,.33),(16,.33),(8,1.))[cell['setting']]
                config=dict(family='extra',features='raw',params=dict(min_samples_leaf=leaf,max_features=fraction))
                model=estimator(config,SEEDS[cell['seed']][0],raw.shape[1])
                model.fit(raw[tr],target,sample_weight=weight)
                prediction,gate=native_gate(model,raw,dest/'model',args.binary_dir/'corrhead_parity',
                                            dict(plan_sha256=sha(args.study/'PLAN.json'),cell=cell['id']))
                pred=prediction*100
                qualification=dict(native_tree_gate=gate,config=config,objective='reference-balanced mapped-target MSE')
            else:
                lam=(1e-7,1e-6,1e-4,1e-2)[cell['setting']]
                output=dest/'native';cmd=[str(HEAVY),'--mem','4G','--jobs','1','--',str(args.binary_dir/'bake_dial_refit'),
                     'linear-study','--inputs',str(linear/'inputs.npz'),'--gram',str(linear/'gram.npz'),
                     '--spec',str(linear/'SPEC.json'),'--lambda',str(lam),'--out',str(output),'--positive']
                with (dest/'train.log').open('w') as log:
                    log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                result=json.loads((output/'RESULT.json').read_text())
                if result['basis_sha256']!=hashlib.sha256(basis.tobytes()).hexdigest():raise ValueError('native basis differs')
                predictions=np.load(output/'predictions.npz');native=predictions['native'];optimum=predictions['optimum']
                reference=100*(1-basis.astype(np.float64)@np.asarray(result['coefficients']))
                if not np.allclose(optimum,reference,rtol=1e-12,atol=1e-6):raise ValueError('independent f64 optimum differs')
                probes=native[len(wanted):];delta=probes[128:256]-probes[:128]
                if not np.isfinite(native).all() or probes[256]!=100 or np.any(delta>1e-5):raise ValueError('native shape feature gate failed')
                pred=native[:len(wanted)]
                qualification=dict(certificate=result['certificate'],lambda_=lam,model_sha256=sha(output/'model.bin'),
                    basis_sha256=result['basis_sha256'],maximum_f32_difference=float(np.max(np.abs(native-optimum))),
                    identity=float(probes[256]),damage_increase_violations=int(np.sum(delta>1e-5)),
                    objective='reference-balanced mapped-target regularized quadratic MSE')
            if not np.isfinite(pred).all():raise ValueError('nonfinite control predictions')
            evaluation=dest/'evaluation';evaluation.mkdir()
            report=dict(cell=cell,qualification=qualification,maps=maps,
                        train=metrics(pred,d['y'],tr,d,maps),inner_validation=metrics(pred,d['y'],te,d,maps),
                        outer_results_read=False,plan_sha256=sha(args.study/'PLAN.json'))
            np.savez_compressed(evaluation/'predictions.npz',original_indices=wanted,native=pred)
            write(evaluation/'REPORT.json',report)
            write(evaluation/'COMPLETE.json',dict(cell=cell['id'],report_sha256=sha(evaluation/'REPORT.json')))
            write(dest/'RESULT.json',{**running,'stage':'complete','exit_code':0,'finished_utc':datetime.now(timezone.utc).isoformat()})
            print(json.dumps({'cell':cell['id'],'stage':'complete','inner_selection_score':report['inner_validation']['selection_score']}),flush=True)
        except Exception as error:
            write(dest/'RESULT.json',{**running,'stage':'failed','exit_code':1,'error':str(error),'finished_utc':datetime.now(timezone.utc).isoformat()})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True);p.add_argument('--attempt',type=int,default=1)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--tree',action='store_true');mode.add_argument('--linear',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
