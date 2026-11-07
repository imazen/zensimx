"""Native full-FIT setting search, isolated from earlier nested-study decisions."""
from __future__ import annotations
import argparse,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,encoded,sha
from shaping_execution import load_registration,prepare_partition
from shaping_full_fit_admission import resolve as resolve_admission
from shaping_inputs import prepare as prepare_shapes
from shaping_outer_execution import command
from shaping_extension_execution import optimizer
from shaping_pretraining import structural_terms
from shaping_rescue import admitted_families
from shaping_rescue_execution import primary_source,phase
from shaping_storage import reservation


def search_cell(root,identifier):
    plan,cells,partitions=load_registration(root);admission,path=resolve_admission(root)
    matches=[c for c in admission['selection_cells'] if c['id']==identifier]
    if len(matches)!=1 or matches[0] not in cells:raise ValueError('registered full-FIT selection cell required')
    cell=matches[0]
    if cell['stage']!='full_fit_selection' or cell['partition'] not in ('outer-0','outer-1','outer-2') or cell['setting'] not in range(4):
        raise ValueError('full-FIT selection scope differs')
    family=admission['finalist_roles'][cell['family']]
    schedule=dict(admission['schedules'][family],setting=cell['setting'])
    return plan,cells,partitions,admission,path,cell,family,schedule


def prepare(root,identifier):
    plan,_,partitions,_,path,cell,_,_=search_cell(root,identifier)
    partition=cell['partition'];dest=root/'full_fit/inputs'/partition
    if dest.exists():return prepared(root,identifier)
    with reservation(root,dest,plan['minimum_ssd_free_bytes']):
        # Reproduce admission again after a possible storage wait.
        search_cell(root,identifier)
        record=prepare_partition(root,partition,partitions[partition],plan)
        shapes=[prepare_shapes(root,partition,partitions[partition],plan,separate) for separate in (False,True)]
        pins={str(root/'comparators'/partition/'GROUPS.json'):sha(root/'comparators'/partition/'GROUPS.json'),**record['file_sha256']}
        for shape in shapes:
            pins[str(shape)]=sha(shape);pins[str(shape.with_name('RECEIPT.json'))]=sha(shape.with_name('RECEIPT.json'))
        receipt=dict(partition=partition,admission_sha256=sha(path),plan_sha256=sha(root/'PLAN.json'),
                     file_sha256=pins,selection_scope='full_fit_only',development_results_read=False,
                     models_fitted=0,driver_sha256=sha(__file__))
        dest.mkdir(parents=True,exist_ok=False)
        (dest/'PREPARED.json').write_bytes(encoded(receipt))
        return receipt


def prepared(root,identifier):
    _,_,partitions,_,path,cell,_,_=search_cell(root,identifier)
    receipt=json.loads((root/'full_fit/inputs'/cell['partition']/'PREPARED.json').read_text())
    if (receipt['partition']!=cell['partition'] or receipt['admission_sha256']!=sha(path)
        or receipt['plan_sha256']!=sha(root/'PLAN.json') or receipt['selection_scope']!='full_fit_only'
        or receipt['development_results_read']):raise ValueError('full-FIT input scope or admission differs')
    for name,pin in receipt['file_sha256'].items():
        if sha(name)!=pin:raise ValueError('full-FIT TRAIN input drift')
    group_path=root/'comparators'/cell['partition']/'GROUPS.json'
    if receipt['file_sha256'].get(str(group_path))!=sha(group_path):raise ValueError('unbound full-FIT groups')
    record=json.loads(group_path.read_text())
    if record['membership_sha256']!=__import__('hashlib').sha256(encoded(partitions[cell['partition']])).hexdigest() or record['exposed_development_passed_to_trainer']:
        raise ValueError('full-FIT source membership or development exposure differs')
    return receipt


def execute(args):
    plan,cells,_,admission,path,cell,family,schedule=search_cell(args.study,args.cell)
    if family not in NEURAL:raise ValueError('native full-FIT runner requires a neural finalist')
    inputs=prepared(args.study,args.cell)
    build_path=args.binary_dir/'BUILD.json';build=json.loads(build_path.read_text())
    pins={str(path):sha(path),str(build_path):sha(build_path),**inputs['file_sha256']}
    for name,pin in build['binary_sha256'].items():
        if sha(args.binary_dir/name)!=pin:raise ValueError('full-FIT native binary drift')
        pins[str(args.binary_dir/name)]=pin
    analytic=family in ('softplus','silu_monotone','silu_diagnostic')
    if analytic:
        qpath=args.neutral_qualification/'RESULT.json';q=json.loads(qpath.read_text())
        if not q['match'] or q['epochs']!=240 or q['build_sha256']!=sha(build_path):raise ValueError('full native neutral replay required')
        pins[str(qpath)]=sha(qpath)
    group_record=json.loads((args.study/'comparators'/cell['partition']/'GROUPS.json').read_text())
    groups=admitted_families(args.study,group_record)
    # Commands encode the fixed four settings and paired seeds. Reuse only their
    # configuration; scales, weights and supervised shaping start afresh below.
    template=dict(cell,family=family,partition='inner-0-0')
    source,original=primary_source(args.study,cells,template,cell['setting'])
    if admission['evidence_sha256'].get(str(source/'RUN.json'))!=sha(source/'RUN.json'):raise ValueError('unadmitted full-FIT command template')
    builder=lambda original,binaries,dest,epochs:command(original,binaries,dest,epochs,group_record['groups'])
    dest=args.study/'runs'/cell['id']/f'attempt-{args.attempt:03}'
    if args.attempt<1 or dest.exists():raise ValueError('preserve full-FIT attempts')
    with reservation(args.study,dest,plan['minimum_ssd_free_bytes']):
        if search_cell(args.study,args.cell)[3]!=admission or prepared(args.study,args.cell)!=inputs:raise ValueError('full-FIT admission changed while waiting')
        for name,pin in pins.items():
            if sha(name)!=pin:raise ValueError('full-FIT input changed while waiting')
        dest.mkdir(parents=True)
        run=dict(cell=cell,resolved_family=family,schedule=schedule,admission_sha256=sha(path),
                 plan_sha256=sha(args.study/'PLAN.json'),build_sha256=sha(build_path),file_sha256=pins,
                 driver_sha256=sha(__file__),pid=os.getpid(),process_role='full_fit_controller',stage='running',
                 fit_directory='fit',epochs=schedule['phase_epochs'][-1],pairs_per_epoch=50000,
                 training_exposure_pair_draws=schedule['total_pair_draws'],selection_scope='full_fit_only',
                 development_results_read=False,outer_results_read=False)
        (dest/'RUN.json').write_bytes(encoded(run));phases=[];shaping=None
        try:
            if family.startswith(('product_','separate_')):
                shape_path=args.study/'shapes'/cell['partition']/family.split('_')[0]/'INITIAL.json'
                if pins.get(str(shape_path))!=sha(shape_path):raise ValueError('unbound full-FIT initial shape')
                shaping=json.loads(shape_path.read_text())
            choice=schedule['optimizer_choice']
            if '_pretrained_' in family:
                parent_family=family.split('_pretrained_')[0]+'_joint_identity'
                parent,parent_run=primary_source(args.study,cells,template,cell['setting'],parent_family)
                if admission['evidence_sha256'].get(str(parent/'RUN.json'))!=sha(parent/'RUN.json'):raise ValueError('unadmitted full-FIT pretraining template')
                epochs=schedule['phase_epochs'][0]
                phases.append(phase(args.study,args.binary_dir,parent,parent_run,dest/'pretraining',epochs,choice,groups,shaping,
                    optimizer=optimizer(choice,False,groups),command_builder=builder))
                prefix=dest/'pretraining/checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
                learned=json.loads(subprocess.check_output([str(args.binary_dir/'bake_dial_refit'),'read-meta','--in',str(prefix),'--key','zentrain.input_shaping'],text=True))
                if structural_terms(learned)!=structural_terms(shaping):raise ValueError('full-FIT pretrained shape layout differs')
                shaping=learned;shaping['trainable']=family.endswith('_joint')
            phases.append(phase(args.study,args.binary_dir,source,original,dest/'fit',run['epochs'],choice,groups,shaping,
                optimizer=optimizer(choice,analytic,groups),command_builder=builder))
            (dest/'RESULT.json').write_bytes(encoded({**run,'phases':phases,'stage':'trained','exit_code':0,
                'bake_sha256':phases[-1]['bake_sha256'],'finished_utc':datetime.now(timezone.utc).isoformat()}))
        except Exception as error:
            (dest/'RESULT.json').write_bytes(encoded({**run,'stage':'failed','exit_code':1,'error':str(error),
                'finished_utc':datetime.now(timezone.utc).isoformat()}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--cell',required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--binary-dir',type=Path);p.add_argument('--neutral-qualification',type=Path)
    p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.study,a.cell)),flush=True)
    else:
        if not (a.binary_dir and a.neutral_qualification):p.error('fit requires binaries and neutral qualification')
        execute(a)
