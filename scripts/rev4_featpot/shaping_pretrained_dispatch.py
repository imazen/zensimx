"""Release registered pretrained trials only after matching parents finish."""
from __future__ import annotations
import argparse,json,os,subprocess,time,fcntl
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_pretraining import matching_parent,completed_parent
from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker,verify_inner_fit


DRIVER_SHA256=sha(__file__)


def parent_state(root,child,cells,build_pin):
    parent=matching_parent(child,cells)
    paths=list((root/'runs'/parent['id']).glob('attempt-*/RESULT.json'))
    if not paths:return False,None
    records=[json.loads(p.read_text()) for p in paths]
    if any(r['stage']=='trained' and r['exit_code']==0 for r in records):
        completed_parent(root,child,cells)
        return True,None
    # An explicit preserved interruption is a dependency wait, never a usable
    # partial prefix and never permission to replay its parent or child.
    pins={}
    for path,result in zip(paths,records):
        if (result.get('stage')!='interrupted' or result.get('exit_code') is not None
                or result.get('restart_approval_pending') is not True
                or len(result.get('preserved_archive_receipt_sha256',''))!=64):
            raise ValueError('parent did not complete; explicit reconciliation required')
        expected=dict(cell=parent,plan_sha256=sha(root/'PLAN.json'),build_sha256=build_pin,
                      epochs=240,pairs_per_epoch=50000)
        identity=json.loads(path.with_name('RUN.json').read_text())
        if any(row.get(k)!=v for row in (identity,result) for k,v in expected.items()):
            raise ValueError('interrupted parent identity differs')
        pins[str(path)]=sha(path);pins[str(path.with_name('RUN.json'))]=sha(path.with_name('RUN.json'))
    return False,dict(parent_cell=parent['id'],reason='preserved interruption awaits owner clarification',evidence_sha256=pins)


def existing_trial(root,cell,plan_pin,build_pin):
    directory=root/'runs'/cell['id']
    if not directory.exists():return None
    trials=[]
    for attempt in directory.glob('attempt-*'):
        if (attempt/'ORCHESTRATION_FAILURE.json').exists():
            if (attempt/'RUN.json').exists():raise ValueError('contradictory pre-training failure')
            continue
        run=json.loads((attempt/'RUN.json').read_text())
        if run['cell']!=cell or run['plan_sha256']!=plan_pin or run['build_sha256']!=build_pin or run['epochs']!=120:
            raise ValueError('existing pretrained identity differs')
        result=attempt/'RESULT.json'
        if result.exists():
            value=json.loads(result.read_text())
            if value['stage']!='trained' or value['exit_code']!=0:raise ValueError('existing training failed')
            if sha(attempt/'checkpoints/ckpt_epoch119.bin')!=value['bake_sha256']:raise ValueError('existing checkpoint drift')
            trials.append(('trained',None))
        else:
            command=Path(f"/proc/{run['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
            if str(attempt) not in command or 'run-heavy' not in command:raise ValueError('existing child is not live')
            trials.append(('running',run['pid']))
    if len(trials)!=1:raise ValueError('existing trials require explicit reconciliation')
    return trials[0]


def dispatch(args):
    if not 1<=args.workers<=4:raise ValueError('explicit 1–4 worker budget required')
    _,cells,_=load_registration(args.study)
    selected=[c for c in cells if c['stage']=='inner' and c['partition']==args.partition and '_pretrained_' in c['family']]
    if not selected:raise ValueError('no registered pretrained cells')
    selected.sort(key=lambda c:(c['setting'],c['seed'],c['family']))
    dest=args.study/'dispatch'/'pretrained'/args.partition
    if args.resume_name:
        if not args.resume_name.replace('-','').isalnum():raise ValueError('invalid resume name')
        dest=dest/args.resume_name
    dest.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),driver_sha256=DRIVER_SHA256,plan_sha256=sha(args.study/'PLAN.json'),
                  build_sha256=sha(args.binary_dir/'BUILD.json'),required_cells=[c['id'] for c in selected])
    (dest/'RUN.json').write_bytes(encoded(identity))
    active={};completed=set();failure=None
    while len(completed)<len(selected):
        for cell_id,(process,log) in list(active.items()):
            if process.poll() is None:continue
            log.close()
            del active[cell_id]
            try:
                if process.returncode:raise RuntimeError('pretrained child failed; inspect '+str(dest/(cell_id+'.log')))
                verify_inner_fit(args.study,args.binary_dir,next(c for c in selected if c['id']==cell_id))
                completed.add(cell_id)
            except Exception as error:failure=str(error)
            finally:release_worker(args.study,cell_id)
        external={}
        for cell in ([] if failure else selected):
            if cell['id'] in active or cell['id'] in completed:continue
            try:state=existing_trial(args.study,cell,identity['plan_sha256'],identity['build_sha256'])
            except (ValueError,OSError,KeyError) as error:
                failure=str(error);break
            if state is not None:
                if state[0]=='trained':completed.add(cell['id'])
                else:external[cell['id']]=state[1]
        blocked={};admission=None
        for cell in selected:
            if failure is not None:break
            if len(active)+len(external)>=args.workers:break
            if cell['id'] in active or cell['id'] in completed or cell['id'] in external:continue
            if (args.study/'runs'/cell['id']).exists():raise ValueError('existing child requires explicit reconciliation')
            try:ready,dependency=parent_state(args.study,cell,cells,identity['build_sha256'])
            except ValueError as error:
                failure=str(error);break
            if dependency:blocked[cell['id']]=dependency
            if not ready:continue
            admission=reserve_worker(args.study,cell['id'])
            if not admission['can_start']:break
            cmd=['python3','-B','scripts/rev4_featpot/shaping_execution.py','--study',str(args.study),
                 '--binary-dir',str(args.binary_dir),'--partition',args.partition,'--inner-family',
                 '--family',cell['family'],'--cell',cell['id'],'--workers','1']
            log=None;process=None
            try:
                log=(dest/(cell['id']+'.log')).open('x')
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                active[cell['id']]=(process,log)
                bind_worker(args.study,cell['id'],process.pid)
            except Exception as error:
                if process is None:
                    if log is not None:log.close()
                    release_worker(args.study,cell['id'])
                failure=str(error);break
        status=dict(**identity,checked_utc=datetime.now(timezone.utc).isoformat(),completed=sorted(completed),
                    active={key:value[0].pid for key,value in active.items()},
                    externally_supervised=external,
                    interrupted_parent_dependencies=blocked,resource_admission=admission,
                    failure=failure,stage=('failed' if failure and not active else
                        'training_finished' if len(completed)==len(selected) else 'supervising'))
        (dest/'STATUS.json').write_bytes(encoded(status))
        if failure and not active:raise RuntimeError(failure)
        if len(completed)<len(selected):time.sleep(30)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume-name',help='new audit directory when resuming an existing dispatch')
    args=p.parse_args()
    directory=args.study/'dispatch'/'pretrained'/args.partition;directory.mkdir(parents=True,exist_ok=True)
    with (directory/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(args)


if __name__=='__main__':main()
