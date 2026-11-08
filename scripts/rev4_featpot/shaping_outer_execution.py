"""Native outer neural refits after complete inner-only admission."""
from __future__ import annotations
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_execution import load_registration,prepare_partition
from shaping_outer_admission import resolve
from shaping_inputs import prepare as prepare_shapes
from shaping_pretraining import structural_terms
from shaping_rescue import admitted_families
from shaping_rescue_execution import primary_source,phase
from shaping_extension_execution import command as phase_command,optimizer,fresh_command
from shaping_storage import reservation


def command(original,binaries,dest,epochs,groups):
    if epochs not in (120,240,480):raise ValueError('unregistered outer phase exposure')
    if {g[0] for g in groups}!={'human_kadid','human_tid2013','human_konfig','human_cid22_a25','cid22'} or len(groups)!=5:
        raise ValueError('exact admitted primary populations required')
    if any(weight<=0 or validation!=0 for _,_,weight,validation,_ in groups):raise ValueError('outer refit must use TRAIN only')
    if any(mode!=('withinref,both' if name=='cid22' else 'withinref,rank') for name,_,_,_,mode in groups):
        raise ValueError('registered human/teacher objectives required')
    cmd=phase_command(original,binaries,dest,240 if epochs==120 else epochs)
    cmd[cmd.index('--epochs')+1]=str(epochs)
    result=[];i=0
    while i<len(cmd):
        if cmd[i]=='--group':i+=2;continue
        result.append(cmd[i]);i+=1
    for name,path,weight,validation,mode in groups:
        result.extend(['--group',f'{name}:{path}:{weight}:{validation}:{mode}'])
    return result


def prepare(root,outer):
    plan,_,partitions=load_registration(root);admission,admission_path=resolve(root,outer)
    partition=f'outer-{outer}';dest=root/'outer_inputs'/partition
    if dest.exists():return prepared(root,outer,admission_path)
    with reservation(root,dest,plan['minimum_ssd_free_bytes']):
        record=prepare_partition(root,partition,partitions[partition],plan)
        shapes={kind:prepare_shapes(root,partition,partitions[partition],plan,kind=='separate') for kind in ('product','separate')}
        pins={str(root/'comparators'/partition/'GROUPS.json'):sha(root/'comparators'/partition/'GROUPS.json'),**record['file_sha256']}
        for path in shapes.values():
            pins[str(path)]=sha(path);pins[str(path.parent/'RECEIPT.json')]=sha(path.parent/'RECEIPT.json')
        dest.mkdir(parents=True,exist_ok=False)
        receipt=dict(partition=partition,admission_sha256=sha(admission_path),plan_sha256=sha(root/'PLAN.json'),
                     file_sha256=pins,driver_sha256=sha(__file__),outer_results_read=False,models_fitted=0)
        (dest/'PREPARED.json').write_bytes(encoded(receipt));return receipt


def prepared(root,outer,admission_path):
    receipt=json.loads((root/'outer_inputs'/f'outer-{outer}'/'PREPARED.json').read_text())
    if receipt['admission_sha256']!=sha(admission_path) or receipt['plan_sha256']!=sha(root/'PLAN.json') or receipt['outer_results_read']:
        raise ValueError('outer input preparation identity differs')
    for name,pin in receipt['file_sha256'].items():
        if sha(name)!=pin:raise ValueError('outer TRAIN input drift')
    return receipt


def resolve_cell(cell,admission):
    if cell['partition']!=f"outer-{admission['outer']}" or not admission['outer_evaluation_admitted']:
        raise ValueError('outer fold not admitted')
    if cell['stage']=='outer' and cell['family'] in NEURAL:
        schedule=admission['schedules'][cell['family']]
        return dict(schedule,fresh=False)
    if cell['stage'] in ('fresh_control_outer','fresh_control_outer_matched'):
        matches=[v for v in admission['fresh_controls'] if v['cell']==cell]
        if len(matches)!=1:raise ValueError('fresh control is not exactly admitted')
        value=matches[0]
        return dict(setting=value['setting'],optimizer_choice=value['optimizer_choice'],
                    phase_epochs=[value['epochs']],total_pair_draws=value['epochs']*50000,fresh=True)
    raise ValueError('registered outer neural or fresh-control cell required')


def admit(args,*,host=None):
    """Reproduce outer admission and resolve immutable inputs without fitting."""
    plan,cells,_=load_registration(args.study);admission,admission_path=resolve(args.study,args.outer)
    cells+=admission['additional_cells'];matches=[c for c in cells if c['id']==args.cell]
    if len(matches)!=1:raise ValueError('one registered outer cell required')
    cell=matches[0];schedule=resolve_cell(cell,admission);prepared(args.study,args.outer,admission_path)
    from shaping_reduced_scope import admit_cell
    admit_cell(args.study,cell,host=host)
    build=json.loads((args.binary_dir/'BUILD.json').read_text())
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('outer native binary drift')
    analytic=cell['family'] in ('softplus','silu_monotone','silu_diagnostic')
    if analytic:
        q=json.loads((args.neutral_qualification/'RESULT.json').read_text())
        if not q['match'] or q['epochs']!=240 or q['build_sha256']!=sha(args.binary_dir/'BUILD.json'):
            raise ValueError('full analytic neutral replay required')
    record=json.loads((args.study/'comparators'/cell['partition']/'GROUPS.json').read_text())
    groups=admitted_families(args.study,record)
    template=dict(cell,partition=f'inner-{args.outer}-0')
    source,original=primary_source(args.study,cells,template,schedule['setting'])
    if admission['decision_sha256'].get(str(source/'RUN.json'))!=sha(source/'RUN.json'):
        raise ValueError('outer command template is not pinned by admission')
    if schedule['fresh']:original=fresh_command(original,cell['seed'])
    templates=[(source,original)]
    if '_pretrained_' in cell['family']:
        family=cell['family'].split('_pretrained_')[0]+'_joint_identity'
        parent,parent_run=primary_source(args.study,cells,template,schedule['setting'],family)
        if admission['decision_sha256'].get(str(parent/'RUN.json'))!=sha(parent/'RUN.json'):
            raise ValueError('outer pretraining template is not pinned by admission')
        templates.append((parent,parent_run))
    return dict(plan=plan,cells=cells,admission=admission,admission_path=admission_path,
                cell=cell,schedule=schedule,record=record,groups=groups,source=source,
                original=original,analytic=analytic,templates=templates)


def execute(args):
    context=admit(args)
    plan,cells,admission,admission_path,cell,schedule,record,groups,source,original,analytic=(
        context[name] for name in ('plan','cells','admission','admission_path','cell','schedule',
                                  'record','groups','source','original','analytic'))
    builder=lambda original,binaries,dest,epochs:command(original,binaries,dest,epochs,record['groups'])
    dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
    if args.attempt<1 or dest.exists():raise ValueError('preserve outer fit attempts')
    with reservation(args.study,dest,plan['minimum_ssd_free_bytes']):
        dest.mkdir(parents=True)
        run=dict(cell=cell,schedule=schedule,admission_sha256=sha(admission_path),
                 execution_host=__import__('socket').gethostname(),
                 plan_sha256=sha(args.study/'PLAN.json'),build_sha256=sha(args.binary_dir/'BUILD.json'),
                 driver_sha256=sha(__file__),pid=os.getpid(),process_role='outer_controller',stage='running',
                 fit_directory='fit',epochs=schedule['phase_epochs'][-1],pairs_per_epoch=50000,
                 training_exposure_pair_draws=schedule['total_pair_draws'],outer_results_read=False)
        (dest/'RUN.json').write_bytes(encoded(run));phases=[];shaping=None
        try:
            if cell['family'].startswith(('product_','separate_')):
                kind=cell['family'].split('_')[0]
                shaping=json.loads((args.study/'shapes'/cell['partition']/kind/'INITIAL.json').read_text())
            choice=schedule['optimizer_choice']
            if '_pretrained_' in cell['family']:
                family=cell['family'].split('_pretrained_')[0]+'_joint_identity'
                parent,parent_run=primary_source(args.study,cells,template,schedule['setting'],family)
                if admission['decision_sha256'].get(str(parent/'RUN.json'))!=sha(parent/'RUN.json'):
                    raise ValueError('outer pretraining template is not pinned by admission')
                epochs=schedule['phase_epochs'][0]
                phases.append(phase(args.study,args.binary_dir,parent,parent_run,dest/'pretraining',epochs,choice,groups,shaping,
                    optimizer=optimizer(choice,False,groups),command_builder=builder))
                prefix=dest/'pretraining/checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
                learned=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'read-meta','--in',str(prefix),'--key','zentrain.input_shaping'],text=True))
                if structural_terms(learned)!=structural_terms(shaping):raise ValueError('outer pretrained shape layout differs')
                shaping=learned;shaping['trainable']=cell['family'].endswith('_joint')
            phases.append(phase(args.study,args.binary_dir,source,original,dest/'fit',run['epochs'],choice,groups,shaping,
                optimizer=optimizer(choice,analytic,groups),command_builder=builder))
            (dest/'RESULT.json').write_bytes(encoded({**run,'phases':phases,'stage':'trained','exit_code':0,
                'bake_sha256':phases[-1]['bake_sha256'],'finished_utc':datetime.now(timezone.utc).isoformat()}))
        except Exception as error:
            (dest/'RESULT.json').write_bytes(encoded({**run,'stage':'failed','exit_code':1,'error':str(error),
                'finished_utc':datetime.now(timezone.utc).isoformat()}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--binary-dir',type=Path);p.add_argument('--neutral-qualification',type=Path)
    p.add_argument('--cell');p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.study,a.outer)),flush=True)
    else:
        if not (a.binary_dir and a.neutral_qualification and a.cell):p.error('fit requires binary, qualification and cell')
        execute(a)
