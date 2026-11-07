"""Release registered pretrained trials only after matching parents finish."""
from __future__ import annotations
import argparse,json,os,subprocess,time,fcntl,signal,socket
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_pretraining import matching_parent,completed_parent
from shaping_rescue_dispatch import reserve_worker,bind_worker,release_worker,verify_inner_fit
from shaping_host_assignment import assigned_placement


DRIVER_SHA256=sha(__file__)


def handoff_controller(args,proc=Path('/proc')):
    """Replace only a verified occupied queue writer; retain every fit process."""
    from shaping_host_assignment import process_identity
    qualification=json.loads(args.handoff_qualification.read_text())
    wrapper=Path.home()/'work/claudehints/scripts/run-heavy'
    if (qualification.get('schema')!='shaping-controller-scope-survival-v1'
            or qualification.get('host')!=socket.gethostname()
            or qualification.get('child_completed_after_parent_signal') is not True
            or qualification.get('parent_only_signals')!=[int(v) for v in (signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT)]
            or qualification.get('run_heavy_sha256')!=sha(wrapper)):
        raise ValueError('matching controller scope-survival qualification required')
    if (not args.resume_name or not args.resume_name.replace('-','').isalnum()
            or not 1<=args.workers<=4 or args.handoff_receipt.exists()
            or (args.study/'dispatch/pretrained'/args.partition/args.resume_name).exists()):
        raise ValueError('new valid resume name, worker budget and handoff receipt required')
    original=json.loads(args.handoff_status.read_text());pid=original['pid']
    command=(proc/str(pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
    def argument(argv,flag):
        if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):raise ValueError('unique queue argument required: '+flag)
        return argv[argv.index(flag)+1]
    if (Path(command[0]).name.startswith('python') is False or '-c' in command or '-m' in command
            or not any(Path(v).name=='shaping_pretrained_dispatch.py' for v in command[1:])
            or argument(command,'--study')!=str(args.study)
            or argument(command,'--binary-dir')!=str(args.binary_dir)
            or argument(command,'--partition')!=args.partition):
        raise ValueError('matching pretrained queue process required')
    _,cells,_=load_registration(args.study)
    expected={c['id'] for c in cells if c['stage']=='inner' and c['partition']==args.partition and '_pretrained_' in c['family']}
    if (set(original['required_cells'])!=expected or original['stage']!='supervising' or original['failure'] is not None
            or original['plan_sha256']!=sha(args.study/'PLAN.json')
            or original['build_sha256']!=sha(args.binary_dir/'BUILD.json')
            or original.get('externally_supervised')):raise ValueError('matching occupied queue evidence required')
    workers=int(argument(command,'--workers'))
    state,start=process_identity(proc,pid)
    if state in ('T','Z','X'):raise ValueError('queue already stopped or terminal')
    paused=False;terminated=False;record=None
    try:
        os.kill(pid,signal.SIGSTOP);paused=True;deadline=time.monotonic()+5
        while process_identity(proc,pid)[0]!='T':
            if time.monotonic()>deadline:raise ValueError('queue did not stop admissions')
            time.sleep(.02)
        current=json.loads(args.handoff_status.read_text());active=current['active']
        children=set(map(int,(proc/str(pid)/'task'/str(pid)/'children').read_text().split()))
        if (current['pid']!=pid or current['required_cells']!=original['required_cells']
                or current['failure'] is not None or len(active)!=workers
                or set(active.values())!=children):raise ValueError('all queue slots must have recorded live fit controllers')
        retained={};pins={}
        for cell_id,child_pid in active.items():
            argv=(proc/str(child_pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
            if (not Path(argv[0]).name.startswith('python') or '-c' in argv or '-m' in argv
                    or not any(Path(v).name=='shaping_execution.py' for v in argv[1:])
                    or argument(argv,'--study')!=str(args.study) or argument(argv,'--cell')!=cell_id
                    or argument(argv,'--partition')!=args.partition):raise ValueError('recorded fit controller differs')
            runs=[p for p in (args.study/'runs'/cell_id).glob('attempt-*/RUN.json') if not p.with_name('RESULT.json').exists()]
            if len(runs)!=1:raise ValueError('one active native attempt required')
            path=runs[0];identity=json.loads(path.read_text());wrapper_pid=identity['pid']
            expected_cell=next(c for c in cells if c['id']==cell_id)
            if any(identity.get(k)!=v for k,v in dict(cell=expected_cell,plan_sha256=original['plan_sha256'],
                    build_sha256=original['build_sha256'],epochs=120,pairs_per_epoch=50000).items()):
                raise ValueError('active pretrained scientific identity differs')
            native_command=(proc/str(wrapper_pid)/'cmdline').read_bytes().decode()
            if str(path.parent) not in native_command or 'run-heavy' not in native_command:
                raise ValueError('active native wrapper is not verified live')
            retained[cell_id]=dict(controller_pid=child_pid,controller_start_ticks=process_identity(proc,child_pid)[1],
                wrapper_pid=wrapper_pid,wrapper_start_ticks=process_identity(proc,wrapper_pid)[1])
            pins[str(path)]=sha(path)
        record=dict(stage='verified_before_controller_signal',checked_utc=datetime.now(timezone.utc).isoformat(),
            previous_pid=pid,previous_start_ticks=start,previous_command=command,previous_status=current,
            qualification_sha256=sha(args.handoff_qualification),retained_active_fits=retained,
            input_sha256=pins,driver_sha256=DRIVER_SHA256,training_replayed=False)
        with args.handoff_receipt.open('xb') as stream:stream.write(encoded(record))
        if process_identity(proc,pid)!=('T',start):raise ValueError('queue identity changed while stopped')
        # Signal exactly the queue writer; never its scope, process group or fits.
        os.kill(pid,signal.SIGTERM)
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
        paused=False
        deadline=time.monotonic()+5
        while (proc/str(pid)/'stat').exists() and process_identity(proc,pid)[0] not in ('Z','X'):
            if time.monotonic()>deadline:raise ValueError('old queue did not exit; replacement refused')
            time.sleep(.02)
        terminated=True
        for cell_id,fit in retained.items():
            vanished=False
            for label in ('controller','wrapper'):
                try:state,ticks=process_identity(proc,fit[label+'_pid'])
                except FileNotFoundError:vanished=True;continue
                if ticks!=fit[label+'_start_ticks'] or state in ('Z','X'):vanished=True
            if vanished:
                # A completed fit may exit naturally during the signal/readback.
                cell=next(c for c in cells if c['id']==cell_id)
                if existing_trial(args.study,cell,original['plan_sha256'],original['build_sha256'])!=('trained',None):
                    raise ValueError('active fit did not survive controller-only handoff')
                fit['completed_during_handoff']=True
        record.update(stage='controller_exited_fits_preserved',verified_utc=datetime.now(timezone.utc).isoformat())
        args.handoff_receipt.write_bytes(encoded(record))
        return record
    finally:
        if paused and not terminated:
            try:same=process_identity(proc,pid)[1]==start
            except FileNotFoundError:same=False
            if same:os.kill(pid,signal.SIGCONT)


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
        expected=dict(cell=cell,plan_sha256=plan_pin,build_sha256=build_pin,epochs=120,pairs_per_epoch=50000)
        if any(run.get(k)!=v for k,v in expected.items()):
            raise ValueError('existing pretrained identity differs')
        result=attempt/'RESULT.json'
        if result.exists():
            value=json.loads(result.read_text())
            if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('existing pretrained result identity differs')
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
        placed={}
        for cell in ([] if failure else selected):
            if cell['id'] in completed or cell['id'] in active or cell['id'] in external:continue
            try:placement=assigned_placement(args.study,cell,identity['build_sha256'])
            except ValueError as error:failure=str(error);break
            if placement and placement['host']!=__import__('socket').gethostname():placed[cell['id']]=placement
        blocked={};admission=None
        for cell in selected:
            if failure is not None:break
            if len(active)+len(external)>=args.workers:break
            if cell['id'] in active or cell['id'] in completed or cell['id'] in external or cell['id'] in placed:continue
            if (args.study/'runs'/cell['id']).exists():raise ValueError('existing child requires explicit reconciliation')
            try:ready,dependency=parent_state(args.study,cell,cells,identity['build_sha256'])
            except ValueError as error:
                failure=str(error);break
            if dependency:blocked[cell['id']]=dependency
            if not ready:continue
            admission=reserve_worker(args.study,cell['id'])
            if not admission['can_start']:break
            try:placement=assigned_placement(args.study,cell,identity['build_sha256'])
            except Exception as error:
                release_worker(args.study,cell['id']);failure=str(error);break
            if placement and placement['host']!=__import__('socket').gethostname():
                release_worker(args.study,cell['id']);placed[cell['id']]=placement;continue
            cmd=[os.sys.executable,'-B','scripts/rev4_featpot/shaping_execution.py','--study',str(args.study),
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
                    externally_assigned_required=placed,
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
    p.add_argument('--handoff-status',type=Path,help='replace this verified occupied old queue writer only')
    p.add_argument('--handoff-receipt',type=Path)
    p.add_argument('--handoff-qualification',type=Path)
    args=p.parse_args()
    if any((args.handoff_status,args.handoff_receipt,args.handoff_qualification)):
        if not all((args.handoff_status,args.handoff_receipt,args.handoff_qualification,args.resume_name)):
            p.error('handoff requires status, receipt, scope qualification and resume-name')
        handoff_controller(args)
    # Queue-writer ownership only; independent fits still run concurrently.
    directory=args.study/'dispatch'/'pretrained'/'locks';directory.mkdir(parents=True,exist_ok=True)
    with (directory/(args.partition+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch(args)


if __name__=='__main__':main()
