"""Bounded native outer refits, admitted only by the complete inner-evidence gate."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,REPO,encoded,sha,matrix
from shaping_execution import load_registration
from shaping_outer_admission import resolve
from shaping_outer_execution import prepare,resolve_cell


def selected_cells(cells,admission):
    partition=f"outer-{admission['outer']}"
    primary=[c for c in cells if c['stage']=='outer' and c['partition']==partition and c['family'] in NEURAL]
    if len(primary)!=30 or {(c['family'],c['seed']) for c in primary}!={(f,s) for f in NEURAL for s in range(3)}:
        raise ValueError('complete thirty-cell paired outer neural matrix required')
    fresh=[v['cell'] for v in admission['fresh_controls']]
    registered=cells+admission['additional_cells']
    if any(c not in registered for c in fresh):raise ValueError('unregistered fresh outer control')
    selected=primary+fresh
    if len({c['id'] for c in selected})!=len(selected):raise ValueError('duplicate outer cell')
    for cell in selected:resolve_cell(cell,admission)
    return sorted(selected,key=lambda c:(c['seed'],c['family'],c['id']))


def existing(root,cell,admission,admission_pin,build_pin):
    directory=root/'runs'/cell['id']
    if not directory.exists():return None
    attempts=list(directory.glob('attempt-*'))
    if len(attempts)!=1:raise ValueError('outer attempts require explicit reconciliation')
    run=attempts[0];identity=json.loads((run/'RUN.json').read_text())
    schedule=resolve_cell(cell,admission)
    expected=dict(cell=cell,schedule=schedule,admission_sha256=admission_pin,
                  build_sha256=build_pin,plan_sha256=sha(root/'PLAN.json'),
                  epochs=schedule['phase_epochs'][-1],process_role='outer_controller')
    if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('existing outer identity differs')
    result=run/'RESULT.json'
    if result.exists():
        record=json.loads(result.read_text())
        if any(record.get(k)!=v for k,v in expected.items()) or record['stage']!='trained' or record['exit_code']!=0:
            raise ValueError('existing outer fit failed or differs')
        checkpoint=run/'fit/checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin"
        if sha(checkpoint)!=record['bake_sha256']:raise ValueError('existing outer checkpoint drift')
        return 'trained'
    pid=identity['pid'];command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    arguments=[v.decode() for v in command if v]
    if not any(Path(v).name=='shaping_outer_execution.py' for v in arguments):raise ValueError('outer controller identity missing')
    for flag,value in [('--study',str(root)),('--cell',cell['id']),('--outer',str(admission['outer']))]:
        if arguments.count(flag)!=1 or arguments[arguments.index(flag)+1]!=value:
            raise ValueError('outer controller command differs')
    return 'running'


def variant_cells(cells,admission,controls=False):
    if not admission['outer_evaluation_admitted']:raise ValueError('variants are not admitted')
    stages=('source_curve','row_curve','shuffle','human_only')
    partition=f"outer-{admission['outer']}"
    roles=('extra_trees','shaped_additive') if controls else ('relu','best_analytic','best_product','best_separate')
    chosen=[c for c in cells if c['stage'] in stages and c['partition']==partition and c['family'] in roles]
    expected=[c for c in matrix() if c['stage'] in stages and c['partition']==partition and c['family'] in roles]
    if sorted(chosen,key=lambda c:c['id'])!=sorted(expected,key=lambda c:c['id']):
        raise ValueError('complete registered variant matrix required')
    allowed=('extra_trees','shaped_additive') if controls else NEURAL
    if any(admission['finalist_roles'][role] not in allowed for role in roles):
        raise ValueError('variant role resolved to wrong neural/nonneural family')
    return sorted(chosen,key=lambda c:(-1 if c['seed'] is None else c['seed'],c['stage'],c['family'],c['id']))


def existing_variant(root,cell,admission,admission_pin,build_pin,controls=False):
    directory=root/'runs'/cell['id']
    if not directory.exists():return None
    attempts=list(directory.glob('attempt-*'))
    if len(attempts)!=1:raise ValueError('variant attempts require explicit reconciliation')
    from shaping_variant_execution import resolve as resolve_variant
    resolution=resolve_variant(root,cell['id'])
    if resolution['admission_sha256']!=admission_pin:raise ValueError('variant admission differs')
    run=attempts[0];identity=json.loads((run/'RUN.json').read_text())
    expected=dict(cell=cell,build_sha256=build_pin,plan_sha256=sha(root/'PLAN.json'))
    if controls:
        expected.update(variant=resolution,admission_sha256=admission_pin,
            selected_setting=resolution['schedule']['setting'],process_role='control_worker')
    else:
        expected.update(resolution=resolution,epochs=resolution['schedule']['phase_epochs'][-1],
            process_role='variant_controller')
    if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('existing variant identity differs')
    result=run/'RESULT.json'
    if result.exists():
        record=json.loads(result.read_text())
        if any(record.get(k)!=v for k,v in expected.items()) or record['stage']!=('complete' if controls else 'trained') or record['exit_code']!=0:
            raise ValueError('existing variant fit failed or differs')
        if controls:
            evaluation=run/'evaluation';report=json.loads((evaluation/'REPORT.json').read_text())
            marker=json.loads((evaluation/'COMPLETE.json').read_text())
            if (marker!={'cell':cell['id'],'report_sha256':sha(evaluation/'REPORT.json')}
                or any(report.get(k)!=expected[k] for k in ('cell','plan_sha256','variant','admission_sha256','selected_setting'))
                or report.get('predictions_sha256')!=sha(evaluation/'predictions.npz')
                or report.get('outer_results_read') is not True or 'outer_test' not in report):
                raise ValueError('variant control completed evidence differs')
            return 'trained'
        checkpoint=run/'fit/checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin"
        if sha(checkpoint)!=record['bake_sha256']:raise ValueError('existing variant checkpoint drift')
        return 'trained'
    arguments=[v.decode() for v in Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b'\0') if v]
    driver='shaping_controls.py' if controls else 'shaping_variant_execution.py'
    if not any(Path(v).name==driver for v in arguments):raise ValueError('variant controller identity missing')
    for flag,value in [('--study',str(root)),('--cell',cell['id'])]:
        if arguments.count(flag)!=1 or arguments[arguments.index(flag)+1]!=value:
            raise ValueError('variant controller command differs')
    return 'running'


def dispatch(args):
    if not 1<=args.workers<=4:raise ValueError('explicit one-to-four worker budget required')
    if not args.session.replace('-','').isalnum():raise ValueError('invalid dispatch session')
    controls=getattr(args,'variant_controls',False)
    variants=getattr(args,'variants',False) or controls
    if variants and not controls and not getattr(args,'human_qualification',None):raise ValueError('variant human qualification required')
    directory=args.study/'dispatch'/('variants-controls' if controls else 'variants-neural' if variants else 'outer')/f'outer-{args.outer}';directory.mkdir(parents=True,exist_ok=True)
    # Ownership of this one queue, never a global training/compute lock.
    with (directory/'OWNER.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=directory/args.session;dest.mkdir(exist_ok=False)
        status=dict(pid=os.getpid(),driver_sha256=sha(__file__),outer=args.outer,workers=args.workers)
        def save(stage,**values):
            status.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**values)
            (dest/'STATUS.json').write_bytes(encoded(status))
        admission_path=args.study/'selection'/f'outer-{args.outer}'/'OUTER_ADMISSION.json'
        while not admission_path.exists():
            if not args.watch_admission:raise ValueError('outer evidence admission is not yet complete')
            save('waiting_for_verified_outer_admission');time.sleep(30)
        admission,path=resolve(args.study,args.outer)
        _,cells,_=load_registration(args.study);selected=variant_cells(cells,admission,controls) if variants else selected_cells(cells,admission)
        pins=dict(admission_sha256=sha(path),build_sha256=sha(args.binary_dir/'BUILD.json'),
                  plan_sha256=sha(args.study/'PLAN.json'),required_cells=selected)
        (dest/'RUN.json').write_bytes(encoded(dict(status,**pins)))
        save('validating_prepared_variants' if variants else 'preparing_training_inputs',**pins)
        if variants:
            from shaping_variant_execution import resolve as resolve_variant
            for cell in selected:resolve_variant(args.study,cell['id'])
        else:prepare(args.study,args.outer)
        active={};completed=set();failure=None
        while len(completed)<len(selected):
            for cell_id,(process,log) in list(active.items()):
                if process.poll() is None:continue
                log.close();del active[cell_id]
                if process.returncode:failure='outer child failed: '+cell_id
            external=set()
            try:
                for cell in selected:
                    if cell['id'] in active or cell['id'] in completed:continue
                    state=(existing_variant(args.study,cell,admission,pins['admission_sha256'],pins['build_sha256'],controls) if variants
                           else existing(args.study,cell,admission,pins['admission_sha256'],pins['build_sha256']))
                    if state=='trained':completed.add(cell['id'])
                    elif state=='running':external.add(cell['id'])
            except (ValueError,OSError,KeyError) as error:failure=str(error)
            for cell in selected:
                if failure or len(active)+len(external)>=args.workers:break
                if cell['id'] in completed or cell['id'] in active or cell['id'] in external:continue
                if sha(path)!=pins['admission_sha256'] or sha(args.binary_dir/'BUILD.json')!=pins['build_sha256']:
                    failure='admitted dispatch inputs changed';break
                command=['python3','-B','scripts/rev4_featpot/shaping_outer_execution.py','--study',str(args.study),
                         '--outer',str(args.outer),'--binary-dir',str(args.binary_dir),
                         '--neutral-qualification',str(args.neutral_qualification),'--cell',cell['id']]
                if variants and not controls:
                    command=['python3','-B','scripts/rev4_featpot/shaping_variant_execution.py','--study',str(args.study),
                        '--binary-dir',str(args.binary_dir),'--neutral-qualification',str(args.neutral_qualification),
                        '--human-qualification',str(args.human_qualification),'--cell',cell['id']]
                if controls:
                    command=['python3','-B','scripts/rev4_featpot/shaping_controls.py','--study',str(args.study),
                        '--binary-dir',str(args.binary_dir),'--partition',cell['partition'],'--cell',cell['id'],
                        '--tree' if cell['family']=='extra_trees' else '--linear']
                log=(dest/(cell['id']+'.log')).open('x')
                active[cell['id']]=(subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT),log)
            save('failed' if failure and not active else 'training_finished' if len(completed)==len(selected) else 'supervising',
                 completed=sorted(completed),active={k:v[0].pid for k,v in active.items()},
                 externally_supervised=sorted(external),failure=failure)
            if failure and not active:raise RuntimeError(failure)
            if len(completed)<len(selected):time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','neutral-qualification'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True)
    p.add_argument('--workers',type=int,required=True);p.add_argument('--session',default='initial')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--variants',action='store_true')
    mode.add_argument('--variant-controls',action='store_true');p.add_argument('--human-qualification',type=Path)
    p.add_argument('--watch-admission',action='store_true');dispatch(p.parse_args())
