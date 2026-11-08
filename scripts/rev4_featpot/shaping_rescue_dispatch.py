"""Release registered rescue fits only after complete paired base selection."""
from __future__ import annotations
import argparse,json,os,subprocess,time,fcntl,uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import MAX_PAIR_FAMILIES,NEURAL,REPO,encoded,sha
from shaping_execution import load_registration,registered_subset
from shaping_host_assignment import check
from shaping_rescue import selected_base,resolve
from shaping_selection import select


def base_ready(root,cells,family,outer):
    expected=[c for c in cells if c['family']==family and c['stage']=='inner' and
              c['partition'] in (f'inner-{outer}-0',f'inner-{outer}-1')]
    seeds=(None,) if family=='shaped_additive' else (0,1,2)
    coverage={(setting,f'inner-{outer}-{fold}',seed) for setting in range(4)
              for fold in range(2) for seed in seeds}
    if (family not in (*NEURAL,'extra_trees','shaped_additive') or len(expected)!=len(coverage)
        or {(c['setting'],c['partition'],c['seed']) for c in expected}!=coverage):
        raise ValueError('complete registered base matrix required')
    repaired=family in MAX_PAIR_FAMILIES
    for cell in expected:
        directory=root/('reexport_max_pairs' if repaired else 'runs')/cell['id']
        markers=list(directory.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
        if len(markers)>1:raise ValueError('ambiguous base report')
        if not markers:return False
        parent=markers[0].parent.parent
        if family in NEURAL and not (parent/'CONVERGENCE.json').exists() and not (parent/'CONVERGENCE_FAILURE.json').exists():return False
    return True


def completed_runs(root,selected,build_pin):
    """Resume only terminal successful fits with the original selected inputs."""
    completed=set();pins={}
    for cell in selected:
        directory=root/'runs'/cell['id']
        if not directory.exists():continue
        attempts=list(directory.glob('attempt-*'))
        if len(attempts)!=1:raise ValueError('rescue attempts require explicit reconciliation')
        run=attempts[0];identity=json.loads((run/'RUN.json').read_text())
        result=json.loads((run/'RESULT.json').read_text())
        expected=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256=build_pin,
            resolution=resolve(root,cell['id']),process_role='rescue_controller',
            epochs=120 if '_pretrained_' in cell['family'] else 240,
            pairs_per_epoch=50000,training_exposure_pair_draws=12000000,fit_directory='fit')
        if any(identity.get(k)!=v or result.get(k)!=v for k,v in expected.items()):
            raise ValueError('resumed rescue identity or exposure differs')
        if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('failed rescue cannot be resumed as complete')
        checkpoint=run/'fit/checkpoints'/f"ckpt_epoch{expected['epochs']-1:03}.bin"
        if sha(checkpoint)!=result['bake_sha256']:raise ValueError('resumed rescue checkpoint drift')
        for name in ('RUN.json','RESULT.json'):pins[str(run/name)]=sha(run/name)
        pins[str(checkpoint)]=sha(checkpoint);completed.add(cell['id'])
    return completed,pins


def resource_controls(args):
    workers=args.workers;pin=None
    path=getattr(args,'worker_budget_file',None)
    if path is not None:
        content=path.read_bytes();record=json.loads(content)
        if set(record)!={'workers'}:raise ValueError('worker budget contains non-resource fields')
        workers=record['workers'];pin=__import__('hashlib').sha256(content).hexdigest()
    capacity=getattr(args,'worker_capacity',4)
    if type(workers) is not int or not 1<=workers<=capacity:raise ValueError('worker budget exceeds declared capacity')
    drain=getattr(args,'drain_file',None)
    return workers,drain is not None and drain.exists(),pin


def worker_slot(active_cells,proc=Path('/proc'),cpu_count=None):
    cpu_count=len(os.sched_getaffinity(0)) if cpu_count is None else cpu_count
    native=0;own=0
    for path in proc.glob('[0-9]*/comm'):
        try:
            if path.read_text().strip()!='zensim_mlp_trai':continue
            native+=1
            command=(path.parent/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if any(cell in command for cell in active_cells):own+=1
        except (FileNotFoundError,ProcessLookupError):continue
    pending=max(0,len(active_cells)-own)
    available=int(next(line.split()[1] for line in (proc/'meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    budget=max(1,(cpu_count or 1)-4);minimum=(6+2*pending)*1024**3
    return dict(can_start=native+pending<budget and available>=minimum,native_threads=native,
                pending_dispatcher_threads=pending,cpu_budget=budget,available_memory_bytes=available,
                minimum_available_memory_bytes=minimum)


OWNER_TOKEN=f'{os.getpid()}:{uuid.uuid4().hex}'


@contextmanager
def worker_claims(root):
    directory=root/'cpu-admission';directory.mkdir(exist_ok=True)
    with (directory/'transaction.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=directory/'CLAIMS.json';claims=json.loads(path.read_text()) if path.exists() else {}
        yield claims
        path.write_bytes(encoded(claims))


def pending_scientific_cells(root):
    import re
    cells={p.parent.parent.name for p in root.glob('runs/*/attempt-*/RUN.json')
           if not p.with_name('RESULT.json').exists()}
    directory=root/'storage-admission'
    if directory.exists():
        with (directory/'transaction.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=directory/'RESERVATIONS.json'
            for path in json.loads(ledger.read_text()) if ledger.exists() else {}:
                match=re.search(r'/runs/([a-f0-9]{20})/attempt-',path)
                if match:cells.add(match[1])
    return cells


def reserve_worker(root,cell,proc=Path('/proc'),cpu_count=None):
    # The lock spans this short admission transaction, never training.
    # Unexpected dead-owner claims remain reserved for explicit reconciliation.
    with worker_claims(root) as claims:
        legacy=pending_scientific_cells(root)
        observation=worker_slot(set(claims)|legacy,proc,cpu_count)
        observation['shared_claims']=len(claims)
        observation['existing_scientific_reservations']=len(legacy)
        if cell in claims:
            observation.update(can_start=False,reason='cell already claimed')
        if observation['can_start']:
            claims[cell]=dict(owner=OWNER_TOKEN,dispatcher_pid=os.getpid(),child_pid=None,
                admitted_utc=datetime.now(timezone.utc).isoformat())
        return observation


def bind_worker(root,cell,pid):
    with worker_claims(root) as claims:
        if claims[cell]['owner']!=OWNER_TOKEN:raise ValueError('worker claim owner differs')
        claims[cell]['child_pid']=pid


def release_worker(root,cell):
    with worker_claims(root) as claims:
        if claims[cell]['owner']!=OWNER_TOKEN:raise ValueError('worker claim owner differs')
        del claims[cell]


def dispatch(args):
    capacity=getattr(args,'worker_capacity',4)
    host_budget=max(1,len(os.sched_getaffinity(0))-4)
    if type(capacity) is not int or not 1<=capacity<=max(4,host_budget):raise ValueError('worker capacity exceeds host budget')
    if not 1<=args.workers<=capacity:raise ValueError('workers exceed declared capacity')
    _,cells,_=load_registration(args.study)
    trigger=args.study/f'RESCUE_TRIGGER_OUTER_{args.outer}.json'
    if not trigger.exists():raise ValueError('registered rescue trigger required')
    selected=[c for c in cells if c['stage']=='rescue' and c['partition'].startswith(f'inner-{args.outer}-')]
    if len(selected)!=240:raise ValueError('complete registered rescue wave required')
    if set(json.loads(trigger.read_text())['required_rescue_cells'])!={c['id'] for c in selected}:
        raise ValueError('trigger coverage differs')
    from shaping_reduced_scope import coverage,shortlist
    reduced=coverage(args.study)
    scope_pin=None
    if reduced is not None:
        scope,scope_path=shortlist(args.study,args.outer)
        required={c['id'] for c in scope['cells'] if c['stage']=='rescue'}
        selected=[c for c in selected if c['id'] in required]
        if len(selected)!=48 or {c['id'] for c in selected}!=required:
            raise ValueError('complete owner-approved reduced rescue wave required')
        scope_pin=sha(scope_path)
        from shaping_reduced_scope import admit_rescue
        admit_rescue(args.study,selected[0])
    resume=getattr(args,'resume_completed',False)
    completed,pins=completed_runs(args.study,selected,sha(args.binary_dir/'BUILD.json')) if resume else (set(),{})
    if not resume and any((args.study/'runs'/c['id']).exists() for c in selected):
        raise ValueError('existing rescue attempts require explicit reconciliation')
    directory=args.study/'dispatch/rescue'/f'outer-{args.outer}'
    if reduced is not None:directory=directory/'reduced-v1'
    if resume and directory.exists():
        previous=json.loads((directory/'STATUS.json').read_text())
        if Path(f"/proc/{previous['pid']}").exists():raise ValueError('previous rescue dispatcher still live')
        if any(Path(f'/proc/{pid}').exists() for pid in previous['active'].values()):
            raise ValueError('previous rescue children still live')
        directory=directory/('resume-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    elif resume and reduced is None:
        raise ValueError('previous rescue dispatcher status required')
    directory.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),driver_sha256=sha(__file__),plan_sha256=sha(args.study/'PLAN.json'),
                  build_sha256=sha(args.binary_dir/'BUILD.json'),trigger_sha256=sha(trigger),
                  workers=args.workers,worker_capacity=capacity,required_cells=[c['id'] for c in selected],
                  worker_budget_file=str(args.worker_budget_file) if getattr(args,'worker_budget_file',None) else None,
                  drain_file=str(args.drain_file) if getattr(args,'drain_file',None) else None)
    if reduced is not None:
        identity.update(reduced_scope_sha256=reduced['registration_sha256'],shortlist_sha256=scope_pin)
    if resume:identity.update(resumed_completed=sorted(completed),resume_evidence_sha256=pins)
    (directory/'RUN.json').write_bytes(encoded(identity))
    active={};ready=set();failure=None;worker_limit=args.workers;draining=False;budget_pin=None;admission=None
    selected.sort(key=lambda c:(NEURAL.index(c['family']),c['rescue'],c['seed'],c['partition']))
    while len(completed)<len(selected):
        for cell_id,(process,log) in list(active.items()):
            if process.poll() is None:continue
            log.close();del active[cell_id]
            try:
                release_worker(args.study,cell_id)
                if process.returncode:raise RuntimeError('rescue child failed: '+cell_id)
                result=json.loads((args.study/'runs'/cell_id/'attempt-001/RESULT.json').read_text())
                if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('rescue child lacks successful result')
                completed.add(cell_id)
            except Exception as error:failure=str(error)
        try:
            worker_limit,draining,budget_pin=resource_controls(args)
            if not failure and not draining:
                for family in NEURAL:
                    if family in ready:continue
                    path=args.study/'selection'/f'outer-{args.outer}'/(family+'-base.json')
                    if not path.exists():
                        if not base_ready(args.study,cells,family,args.outer):continue
                        select(SimpleNamespace(study=args.study,family=family,outer=args.outer,reexport_max_pairs=family in MAX_PAIR_FAMILIES))
                    selected_base(args.study,family,args.outer,cells)
                    # Recheck the frozen trigger and selected evidence before
                    # considering the family ready; each child repeats this gate.
                    resolve(args.study,next(c['id'] for c in selected if c['family']==family))
                    ready.add(family)
                for cell in selected:
                    if len(active)>=worker_limit:break
                    if cell['family'] not in ready or cell['id'] in active or cell['id'] in completed:continue
                    if (args.study/'runs'/cell['id']).exists():raise ValueError('unexpected existing rescue attempt')
                    admission=reserve_worker(args.study,cell['id'])
                    if not admission['can_start']:break
                    command=['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_rescue_execution.py'),
                        '--study',str(args.study),'--binary-dir',str(args.binary_dir),
                        '--neutral-qualification',str(args.neutral_qualification),'--cell',cell['id']]
                    log=None;process=None
                    try:
                        log=(directory/(cell['id']+'.log')).open('x')
                        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
                        active[cell['id']]=(process,log)
                        bind_worker(args.study,cell['id'],process.pid)
                    except BaseException:
                        if process is None:
                            if log is not None:log.close()
                            release_worker(args.study,cell['id'])
                        raise
        except Exception as error:failure=str(error)
        status=dict(**identity,checked_utc=datetime.now(timezone.utc).isoformat(),ready_families=sorted(ready),
                    waiting_families=sorted(set(NEURAL)-ready),completed=sorted(completed),
                    active={key:value[0].pid for key,value in active.items()},failure=failure,
                    worker_limit=worker_limit,worker_budget_sha256=budget_pin,drain_requested=draining,resource_admission=admission,
                    stage='failed' if failure and not active else 'drained' if draining and not active else 'training_finished' if len(completed)==len(selected) else 'supervising')
        (directory/'STATUS.json').write_bytes(encoded(status))
        if failure and not active:raise RuntimeError(failure)
        if draining and not active:return
        if len(completed)<len(selected):time.sleep(30)


def selected_inner_cells(root,cells,requested,partition,family):
    families={'joint_shapes':('product_joint_identity','separate_joint_identity'),
              'analytic':('softplus','silu_monotone','silu_diagnostic'),
              'relu':('relu',),
              'pretrained_shapes':('product_pretrained_frozen','product_pretrained_joint',
                                   'separate_pretrained_frozen','separate_pretrained_joint')}[family]
    selected=registered_subset([c for c in cells if c['stage']=='inner' and
        c['partition']==partition and c['family'] in families],requested)
    if not selected:raise ValueError('explicit registered subset required')
    if any((root/'runs'/c['id']).exists() for c in selected):
        raise ValueError('existing cell attempts require explicit reconciliation; no automatic replay')
    return sorted(selected,key=lambda c:(c['setting'],c['seed'],families.index(c['family'])))


def verify_inner_fit(root,binary_dir,cell):
    run=root/'runs'/cell['id']/'attempt-001'
    identity=json.loads((run/'RUN.json').read_text())
    result=json.loads((run/'RESULT.json').read_text())
    epochs=120 if '_pretrained_' in cell['family'] else 240
    expected=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),
                  build_sha256=sha(binary_dir/'BUILD.json'),epochs=epochs,pairs_per_epoch=50000)
    if any(record.get(k)!=v for record in (identity,result) for k,v in expected.items()):
        raise ValueError('completed inner fit identity differs')
    if result.get('stage')!='trained' or result.get('exit_code')!=0:
        raise ValueError('child lacks successful native result')
    checkpoint=run/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin'
    if sha(checkpoint)!=result['bake_sha256']:raise ValueError('final native checkpoint drift')
    return {str(p):sha(p) for p in (run/'RUN.json',run/'RESULT.json',checkpoint)}


def dispatch_inner(args):
    host_budget=max(1,len(os.sched_getaffinity(0))-4)
    if not 1<=args.worker_capacity<=host_budget:raise ValueError('capacity exceeds host CPU budget')
    resource_controls(args)
    _,cells,_=load_registration(args.study)
    selected=selected_inner_cells(args.study,cells,args.cell,args.partition,args.family)
    check(args.study,selected,sha(args.binary_dir/'BUILD.json'))
    args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        driver_sha256=sha(__file__),execution_sha256=sha(REPO/'scripts/rev4_featpot/shaping_execution.py'),
        plan_sha256=sha(args.study/'PLAN.json'),build_sha256=sha(args.binary_dir/'BUILD.json'),
        required_cells=[c['id'] for c in selected],partition=args.partition,family=args.family,
        worker_capacity=args.worker_capacity,worker_budget_file=str(args.worker_budget_file),
        drain_file=str(args.drain_file) if args.drain_file else None,no_existing_attempts=True)
    (args.output/'RUN.json').write_bytes(encoded(identity))
    active={};completed={};failure=None
    while True:
        for cell_id,(process,log,cell) in list(active.items()):
            if process.poll() is None:continue
            log.close();del active[cell_id]
            try:
                if process.returncode:raise RuntimeError(f'inner child failed: {cell_id}, exit {process.returncode}')
                completed[cell_id]=verify_inner_fit(args.study,args.binary_dir,cell)
            except Exception as error:failure=str(error)
            finally:release_worker(args.study,cell_id)
        admission=None;budget_pin=None;limit=0;draining=False
        try:limit,draining,budget_pin=resource_controls(args)
        except Exception as error:failure=str(error)
        if not failure and not draining:
            for cell in selected:
                cell_id=cell['id']
                if len(active)>=limit:break
                if cell_id in completed or cell_id in active:continue
                admission=reserve_worker(args.study,cell_id)
                if not admission['can_start']:break
                command=[os.sys.executable,'-B',str(REPO/'scripts/rev4_featpot/shaping_execution.py'),
                    '--study',str(args.study),'--binary-dir',str(args.binary_dir),'--inner-family',
                    '--family',args.family,'--partition',args.partition,'--workers','1','--cell',cell_id]
                log=None
                try:
                    log=(args.output/(cell_id+'.log')).open('x')
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
                except Exception as error:
                    if log is not None:log.close()
                    release_worker(args.study,cell_id);failure=str(error);break
                # Register the live child before binding, so any bind error retains its identity.
                active[cell_id]=(process,log,cell)
                try:bind_worker(args.study,cell_id,process.pid)
                except Exception as error:failure=str(error);break
        done=len(completed)==len(selected)
        stage='complete' if done else ('failed' if failure else 'drained') if not active and (failure or draining) else 'supervising'
        status={**identity,'checked_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,
            'active':{cell:p.pid for cell,(p,_,_) in active.items()},'completed':completed,
            'failure':failure,'worker_limit':limit,'draining':draining,
            'worker_budget_sha256':budget_pin,'last_admission':admission}
        temporary=args.output/'STATUS.json.writing';temporary.write_bytes(encoded(status))
        temporary.replace(args.output/'STATUS.json')
        if done or (not active and (failure or draining)):break
        time.sleep(args.interval)
    if failure:raise RuntimeError(failure)
    return status



def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--neutral-qualification',type=Path)
    p.add_argument('--outer',type=int,choices=range(3));p.add_argument('--workers',type=int,default=2)
    p.add_argument('--worker-capacity',type=int,default=4)
    p.add_argument('--worker-budget-file',type=Path);p.add_argument('--drain-file',type=Path)
    p.add_argument('--resume-completed',action='store_true')
    p.add_argument('--inner-subset',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--partition')
    p.add_argument('--family',choices=('joint_shapes','analytic','relu','pretrained_shapes'))
    p.add_argument('--cell',action='append');p.add_argument('--interval',type=float,default=10)
    args=p.parse_args()
    if args.inner_subset:
        if not all((args.output,args.partition,args.family,args.cell,args.worker_budget_file)):
            p.error('inner subset requires output, partition, family, cell and worker-budget-file')
        if args.outer is not None or args.neutral_qualification or args.resume_completed:
            p.error('rescue options cannot change an inner subset')
        if args.interval<=0:p.error('positive polling interval required')
        dispatch_inner(args)
    else:
        if args.outer is None or args.neutral_qualification is None:
            p.error('rescue requires outer and neutral-qualification')
        if any((args.output,args.partition,args.family,args.cell)):
            p.error('inner subset options require --inner-subset')
        dispatch(args)


if __name__=='__main__':main()
