"""Refit registered generalization variants with inner-selected native models."""
from __future__ import annotations
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_execution import load_registration
from shaping_outer_admission import resolve as resolve_outer
from shaping_variant_inputs import load_variant,verify_prepared
from shaping_extension_execution import command as phase_command
from shaping_rescue import controls,admitted_families
from shaping_rescue_execution import primary_source,phase
from shaping_pretraining import structural_terms
from shaping_storage import reservation

STAGES=('source_curve','row_curve','shuffle','human_only')
POPULATIONS={'human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22'}


def command(original,binaries,dest,epochs,groups,absent):
    names=[g[0] for g in groups]
    if set(names)|set(absent)!=POPULATIONS or set(names)&set(absent) or len(names)!=len(set(names)):
        raise ValueError('variant populations must match their explicit absence declaration')
    if not names or any(w<=0 or v!=0 or m!=('withinref,both' if n=='cid22' else 'withinref,rank') for n,_,w,v,m in groups):
        raise ValueError('variant objective requires admitted TRAIN-only groups')
    if epochs not in (120,240,480):raise ValueError('unregistered variant phase exposure')
    cmd=phase_command(original,binaries,dest,240 if epochs==120 else epochs)
    cmd[cmd.index('--epochs')+1]=str(epochs)
    if cmd.count('--mse-weight')!=1:raise ValueError('one native MSE weight required')
    cmd[cmd.index('--mse-weight')+1]='1' if 'cid22' in names else '0'
    result=[];i=0
    while i<len(cmd):
        if cmd[i]=='--group':i+=2;continue
        result.append(cmd[i]);i+=1
    for n,p,w,v,m in groups:result.extend(['--group',f'{n}:{p}:{w}:{v}:{m}'])
    return result


def optimizer(choice,analytic,groups):
    result=controls('init_half' if choice=='base' else choice,analytic,groups=groups,
                    human_only=not any(g['name']=='cid22' for g in groups))
    if choice=='base':
        result['init_scale']=1.
        if analytic:result['bias_scale']=1.
    return result


def resolve(root,cell_id):
    plan,cell,identifier,view=load_variant(root,cell_id)
    if cell['stage'] not in STAGES:raise ValueError('registered generalization variant required')
    outer=int(cell['partition'].split('-')[1]);admission,path=resolve_outer(root,outer)
    if not admission['outer_evaluation_admitted']:raise ValueError('variant finalists are not admitted')
    family=admission['finalist_roles'][cell['family']];schedule=admission['schedules'][family]
    key=f"variant-{identifier[:20]}-{cell.get('permutation','true')}-{cell.get('permutation_seed',0)}"
    group_path=root/'comparators'/key/'GROUPS.json'
    record=verify_prepared(group_path,root,identifier,view,cell)
    audit_path=root/'VARIANT_INPUT_AUDIT.json';audit=json.loads(audit_path.read_text())
    if (audit['plan_sha256']!=sha(root/'PLAN.json') or audit['progress_sha256']!=sha(root/'VARIANT_INPUT_PROGRESS.json')
        or not audit['feature_bits_and_keys_unchanged'] or not audit['training_labels_match_registration'] or audit['outer_responses_used']):
        raise ValueError('variant input audit differs')
    matches=[v for v in audit['views'] if v['path']==str(group_path)]
    if len(matches)!=1 or matches[0]['receipt_sha256']!=sha(group_path):raise ValueError('variant view is not independently audited')
    shape_path=root/'VARIANT_SHAPES.json';shapes=json.loads(shape_path.read_text())
    if (shapes['plan_sha256']!=sha(root/'PLAN.json') or shapes['audit_sha256']!=sha(audit_path)
        or shapes['membership_sha256']!=sha(root/'VARIANT_MEMBERSHIPS.json') or shapes['cells'][cell_id]!=identifier
        or not shapes['label_independent_initializations_only'] or shapes['learned_shapes_reused'] or shapes['outer_responses_used']):
        raise ValueError('variant shape preparation differs')
    pins={str(p):sha(p) for p in (path,group_path,audit_path,shape_path,root/'VARIANT_MEMBERSHIPS.json')}
    pins.update(record['file_sha256']);initial=None
    if family.startswith(('product_','separate_')):
        entry=shapes['views'][identifier]['manifests'][family.split('_')[0]];initial=Path(entry['path'])
        if sha(initial)!=entry['sha256'] or sha(initial.with_name('RECEIPT.json'))!=entry['receipt_sha256']:
            raise ValueError('variant initial shape drift')
        pins[str(initial)]=sha(initial);pins[str(initial.with_name('RECEIPT.json'))]=sha(initial.with_name('RECEIPT.json'))
    if cell['stage']=='human_only' and (view['teacher_training_rows']!=0 or any(g[0]=='cid22' for g in record['groups'])):
        raise ValueError('teacher entered human-only variant')
    return dict(cell=cell,resolved_family=family,schedule=schedule,view_identifier=identifier,
        groups_path=str(group_path),initial_shape=str(initial) if initial else None,
        admission_sha256=sha(path),file_sha256=pins,outer_results_read=False)


def execute(args):
    plan,cells,_=load_registration(args.study);resolution=resolve(args.study,args.cell)
    cell=resolution['cell'];family=resolution['resolved_family'];schedule=resolution['schedule']
    if family not in NEURAL:raise ValueError('variant neural runner requires a neural finalist')
    build_path=args.binary_dir/'BUILD.json';build=json.loads(build_path.read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('variant native binary drift')
    analytic=family in ('softplus','silu_monotone','silu_diagnostic')
    record=json.loads(Path(resolution['groups_path']).read_text());groups=admitted_families(args.study,record)
    qualification_pins={}
    if analytic:
        path=args.neutral_qualification/'RESULT.json';q=json.loads(path.read_text());qualification_pins[str(path)]=sha(path)
        if not q['match'] or q['epochs']!=240 or q['build_sha256']!=sha(build_path):raise ValueError('full native neutral replay required')
        if not any(g['name']=='cid22' for g in groups):
            path=args.human_qualification/family/'RESULT.json';q=json.loads(path.read_text());qualification_pins[str(path)]=sha(path)
            if not q['match'] or q['family']!=family or q['teacher_rows']!=0 or q['build_sha256']!=sha(build_path):
                raise ValueError('human-only native qualification required')
    outer=int(cell['partition'].split('-')[1]);admission,_=resolve_outer(args.study,outer)
    template=dict(cell,family=family,partition=f'inner-{outer}-0')
    source,original=primary_source(args.study,cells,template,schedule['setting'])
    if admission['decision_sha256'].get(str(source/'RUN.json'))!=sha(source/'RUN.json'):raise ValueError('unpinned variant command template')
    builder=lambda original,binaries,dest,epochs:command(original,binaries,dest,epochs,record['groups'],record['explicit_absent_fit_groups'])
    dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
    if args.attempt<1 or dest.exists():raise ValueError('preserve variant fit attempts')
    with reservation(args.study,dest,plan['minimum_ssd_free_bytes']):
        if resolve(args.study,args.cell)!=resolution:raise ValueError('variant admission changed while waiting')
        for name,pin in qualification_pins.items():
            if sha(name)!=pin:raise ValueError('variant native qualification changed while waiting')
        dest.mkdir(parents=True)
        run=dict(cell=cell,resolution=resolution,plan_sha256=sha(args.study/'PLAN.json'),
            build_sha256=sha(build_path),driver_sha256=sha(__file__),pid=os.getpid(),
            process_role='variant_controller',stage='running',fit_directory='fit',epochs=schedule['phase_epochs'][-1],
            pairs_per_epoch=50000,training_exposure_pair_draws=schedule['total_pair_draws'],outer_results_read=False)
        run['qualification_sha256']=qualification_pins
        (dest/'RUN.json').write_bytes(encoded(run));phases=[]
        shaping=json.loads(Path(resolution['initial_shape']).read_text()) if resolution['initial_shape'] else None
        try:
            choice=schedule['optimizer_choice']
            if '_pretrained_' in family:
                parent_family=family.split('_pretrained_')[0]+'_joint_identity'
                parent,parent_run=primary_source(args.study,cells,template,schedule['setting'],parent_family)
                if admission['decision_sha256'].get(str(parent/'RUN.json'))!=sha(parent/'RUN.json'):raise ValueError('unpinned variant pretraining template')
                epochs=schedule['phase_epochs'][0]
                phases.append(phase(args.study,args.binary_dir,parent,parent_run,dest/'pretraining',epochs,choice,groups,shaping,
                    optimizer=optimizer(choice,False,groups),command_builder=builder))
                prefix=dest/'pretraining/checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
                learned=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'read-meta','--in',str(prefix),'--key','zentrain.input_shaping'],text=True))
                if structural_terms(learned)!=structural_terms(shaping):raise ValueError('variant pretrained shape layout differs')
                shaping=learned;shaping['trainable']=family.endswith('_joint')
            phases.append(phase(args.study,args.binary_dir,source,original,dest/'fit',run['epochs'],choice,groups,shaping,
                optimizer=optimizer(choice,analytic,groups),command_builder=builder))
            (dest/'RESULT.json').write_bytes(encoded({**run,'phases':phases,'stage':'trained','exit_code':0,
                'bake_sha256':phases[-1]['bake_sha256'],'finished_utc':datetime.now(timezone.utc).isoformat()}))
        except Exception as error:
            (dest/'RESULT.json').write_bytes(encoded({**run,'stage':'failed','exit_code':1,'error':str(error),
                'finished_utc':datetime.now(timezone.utc).isoformat()}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','neutral-qualification','human-qualification'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cell',required=True);p.add_argument('--attempt',type=int,default=1)
    execute(p.parse_args())
