#!/usr/bin/env python3
"""Registered inner and admitted outer controls using existing fit owners."""
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
from shaping_evaluation import probe_values,apply_report_scope


DRIVER_SHA256=sha(__file__)

def write(path,value):path.write_bytes(encoded(value))


def selected_cells(root,partition,family,cells):
    if partition.startswith('inner-'):
        return [c for c in cells if c['stage']=='inner' and c['family']==family and c['partition']==partition],None
    if partition not in ('outer-0','outer-1','outer-2'):raise ValueError('unregistered control partition')
    from shaping_outer_admission import resolve
    admission,path=resolve(root,int(partition.split('-')[1]))
    if not admission['outer_evaluation_admitted']:raise ValueError('outer controls are not admitted')
    schedule=admission['schedules'][family]
    if schedule['phase_epochs'] or schedule['optimizer_choice']!='base' or schedule['setting'] not in range(4):
        raise ValueError('nonneural control schedule differs')
    chosen=[c for c in cells if c['stage']=='outer' and c['family']==family and c['partition']==partition]
    expected={0,1,2} if family=='extra_trees' else {None}
    if len(chosen)!=len(expected) or {c['seed'] for c in chosen}!=expected:
        raise ValueError('outer control seed coverage differs')
    return chosen,dict(admission_sha256=sha(path),selected_setting=schedule['setting'])


def full_fit_cells(root,partition,family,cells):
    from shaping_full_fit_admission import resolve
    admission,path=resolve(root)
    if partition not in ('outer-0','outer-1','outer-2') or family not in ('extra_trees','shaped_additive'):
        raise ValueError('registered full-FIT control partition/family required')
    if admission['finalist_roles'][family]!=family:raise ValueError('full-FIT control role differs')
    chosen=[c for c in admission['selection_cells'] if c['family']==family and c['partition']==partition]
    seeds=(0,1,2) if family=='extra_trees' else (None,)
    expected={(setting,seed) for setting in range(4) for seed in seeds}
    if (len(chosen)!=len(expected) or {(c['setting'],c['seed']) for c in chosen}!=expected
        or any(c not in cells or c['stage']!='full_fit_selection' for c in chosen)):
        raise ValueError('full-FIT control search coverage differs')
    schedule=admission['schedules'][family]
    if schedule['optimizer_choice']!='base' or schedule['phase_epochs']:raise ValueError('full-FIT control schedule differs')
    return chosen,dict(full_fit=True,admission_sha256=sha(path),selection_scope='full_fit_only',
                       development_results_read=False,resolved_family=family)


def final_refit_cells(root,partition,family,cells):
    from shaping_full_fit_execution import refit_cell
    if partition!='full-fit' or family not in ('extra_trees','shaped_additive'):raise ValueError('registered final control required')
    chosen=[c for c in cells if c['stage']=='full_fit' and c['family']==family]
    seeds={0,1,2} if family=='extra_trees' else {None}
    if len(chosen)!=len(seeds) or {c['seed'] for c in chosen}!=seeds:raise ValueError('final control seed coverage differs')
    _,_,_,admission,path,_,resolved,schedule=refit_cell(root,chosen[0]['id'])
    if resolved!=family or schedule['phase_epochs'] or schedule['optimizer_choice']!='base':raise ValueError('final control schedule differs')
    return chosen,dict(full_fit=True,descriptive=True,admission_sha256=sha(path),selected_setting=schedule['setting'],
        selection_sha256=admission['full_fit_selection_sha256'],selection_scope='full_fit_refit',
        development_results_read=False,resolved_family=family)


def control_cache_key(partition,context):
    variant=context.get('variant') if context else None
    if variant:return Path(variant['groups_path']).parent.name
    return ('full-fit-refit-' if context and context.get('descriptive') else 'full-fit-selection-' if context and context.get('full_fit') else '')+partition


def run(args):
    plan,cells,partitions=load_registration(args.study)
    family='extra_trees' if args.tree else 'shaped_additive'
    variant=None
    full_fit=getattr(args,'full_fit_selection',False)
    final_refit=getattr(args,'full_fit_refit',False)
    if final_refit:
        if full_fit or getattr(args,'cell',None):raise ValueError('final refit cannot share tuning or variant mode')
        chosen,outer=final_refit_cells(args.study,args.partition,family,cells)
        member=partitions['full-fit'];partitions={**partitions,'full-fit':dict(member,test=member['descriptive_development'])}
    elif full_fit:
        if getattr(args,'cell',None):raise ValueError('full-FIT search and variant cell are mutually exclusive')
        chosen,outer=full_fit_cells(args.study,args.partition,family,cells)
    elif getattr(args,'cell',None):
        from shaping_variant_execution import resolve as resolve_variant
        from shaping_variant_inputs import load_variant
        variant=resolve_variant(args.study,args.cell)
        if variant['resolved_family']!=family or variant['cell']['partition']!=args.partition:
            raise ValueError('variant control family or partition differs')
        _,_,_,membership=load_variant(args.study,args.cell)
        partitions={**partitions,args.partition:membership};chosen=[variant['cell']]
        outer=dict(admission_sha256=variant['admission_sha256'],selected_setting=variant['schedule']['setting'],variant=variant)
    else:chosen,outer=selected_cells(args.study,args.partition,family,cells)
    if not chosen:raise ValueError('no registered cells')
    if outer is None:return fit(args,plan,partitions,family,chosen,outer)
    from shaping_storage import reservation
    key=control_cache_key(args.partition,outer)
    dest=args.study/'control_inputs'/key/f'{family}-attempt-{args.attempt:03}'
    with reservation(args.study,dest,plan['minimum_ssd_free_bytes']):
        # Admission may have waited for space; revalidate before reading labels.
        current=resolve_variant(args.study,args.cell) if variant else (final_refit_cells if final_refit else full_fit_cells if full_fit else selected_cells)(args.study,args.partition,family,cells)
        if current!=(variant if variant else (chosen,outer)):
            raise ValueError('outer control admission changed while waiting')
        return fit(args,plan,partitions,family,chosen,outer)


def fit(args,plan,partitions,family,chosen,outer):
    from shaped_linear_study import OLD,metrics,weights_and_targets,fit_spec,apply_basis,HEAVY
    from tree_study import RAW_IDS,estimator,native_gate
    membership=partitions[args.partition]
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
    variant=outer.get('variant') if outer else None
    if variant:
        from shaped_linear_study import PREP
        from shaping_evaluation import training_targets
        record=json.loads(Path(variant['groups_path']).read_text())
        groups={g[0]:g[1] for g in record['groups']};selected=set(membership['train']);offset=0
        for group in json.loads(PREP.read_text())['groups']:
            ids=[v for v in range(offset,offset+group['rows']) if v in selected];offset+=group['rows']
            if ids:training_targets(d['y'],pos,ids,groups[group['name']])
    weight,target,maps=weights_and_targets(d['y'][tr],d['pop'][tr],d['component'][tr],'range',1.)
    declaration=json.loads((REPO/'scripts/rev4_featpot/color_noise_fx2_declaration.json').read_text())
    cache=args.study/'control_inputs'/control_cache_key(args.partition,outer)
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
        setting=outer['selected_setting'] if outer is not None and (not outer.get('full_fit') or outer.get('descriptive')) else cell['setting']
        dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
        if dest.exists():raise ValueError('control attempt exists: '+str(dest))
        dest.mkdir(parents=True)
        running=dict(cell=cell,pid=os.getpid(),process_role='control_worker',stage='running',
                     started_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=DRIVER_SHA256,
                     plan_sha256=sha(args.study/'PLAN.json'),build_sha256=sha(args.binary_dir/'BUILD.json'))
        if outer is not None:running.update(outer)
        write(dest/'RUN.json',running);print(json.dumps({'cell':cell['id'],'pid':os.getpid(),'stage':'running'}),flush=True)
        try:
            if args.tree:
                leaf,fraction=((2,.33),(8,.33),(16,.33),(8,1.))[setting]
                config=dict(family='extra',features='raw',params=dict(min_samples_leaf=leaf,max_features=fraction))
                model=estimator(config,SEEDS[cell['seed']][0],raw.shape[1])
                model.fit(raw[tr],target,sample_weight=weight)
                prediction,gate=native_gate(model,raw,dest/'model',args.binary_dir/'corrhead_parity',
                                            dict(plan_sha256=sha(args.study/'PLAN.json'),cell=cell['id']))
                pred=prediction*100
                qualification=dict(native_tree_gate=gate,config=config,objective='reference-balanced mapped-target MSE')
            else:
                lam=(1e-7,1e-6,1e-4,1e-2)[setting]
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
                        train=metrics(pred,d['y'],tr,d,maps,allow_absent_maps=bool(variant)),
                        inner_validation=metrics(pred,d['y'],te,d,maps,allow_absent_maps=bool(variant)),
                        outer_results_read=False,plan_sha256=sha(args.study/'PLAN.json'))
            if outer is not None:
                report.update(outer)
                apply_report_scope(report,outer,family)
            if variant:
                report.update(training_target_policy='registered permutation' if cell['stage']=='shuffle' else 'actual targets',
                              outer_target_policy='actual targets',resolved_family=family)
            np.savez_compressed(evaluation/'predictions.npz',original_indices=wanted,native=pred)
            write(evaluation/'REPORT.json',report)
            write(evaluation/'COMPLETE.json',dict(cell=cell['id'],report_sha256=sha(evaluation/'REPORT.json')))
            write(dest/'RESULT.json',{**running,'stage':'complete','exit_code':0,'finished_utc':datetime.now(timezone.utc).isoformat()})
            role='exposed_development' if outer and outer.get('descriptive') else 'full_fit_validation' if outer and outer.get('full_fit') else 'outer_test' if outer is not None else 'inner_validation'
            key='exposed_development_score' if role=='exposed_development' else 'full_fit_selection_score' if role=='full_fit_validation' else 'outer_test_score' if outer is not None else 'inner_selection_score'
            print(json.dumps({'cell':cell['id'],'stage':'complete',key:report[role]['selection_score']}),flush=True)
        except Exception as error:
            write(dest/'RESULT.json',{**running,'stage':'failed','exit_code':1,'error':str(error),'finished_utc':datetime.now(timezone.utc).isoformat()})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True);p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--cell',help='registered generalization control cell; retains inner-selected settings')
    p.add_argument('--full-fit-refit',action='store_true',help='final all-FIT control with descriptive development evaluation')
    p.add_argument('--full-fit-selection',action='store_true',help='registered four-setting full-FIT control search')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--tree',action='store_true');mode.add_argument('--linear',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
