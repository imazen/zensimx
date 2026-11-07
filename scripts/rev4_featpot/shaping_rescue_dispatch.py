"""Release registered rescue fits only after complete paired base selection."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import MAX_PAIR_FAMILIES,NEURAL,REPO,encoded,sha
from shaping_execution import load_registration
from shaping_rescue import selected_base,resolve
from shaping_selection import select


def base_ready(root,cells,family,outer):
    expected=[c for c in cells if c['family']==family and c['stage']=='inner' and
              c['partition'] in (f'inner-{outer}-0',f'inner-{outer}-1')]
    if len(expected)!=24:raise ValueError('complete registered base matrix required')
    repaired=family in MAX_PAIR_FAMILIES
    for cell in expected:
        directory=root/('reexport_max_pairs' if repaired else 'runs')/cell['id']
        markers=list(directory.glob('attempt-*/final/COMPLETE.json' if repaired else 'attempt-*/evaluation/COMPLETE.json'))
        if len(markers)>1:raise ValueError('ambiguous base report')
        if not markers:return False
        parent=markers[0].parent.parent
        if not (parent/'CONVERGENCE.json').exists() and not (parent/'CONVERGENCE_FAILURE.json').exists():return False
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
    if type(workers) is not int or not 1<=workers<=4:raise ValueError('explicit one-to-four worker budget required')
    drain=getattr(args,'drain_file',None)
    return workers,drain is not None and drain.exists(),pin


def worker_slot(active_cells,proc=Path('/proc'),cpu_count=None):
    cpu_count=os.cpu_count() if cpu_count is None else cpu_count
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
    budget=max(1,(cpu_count or 1)-4);minimum=6*1024**3
    return dict(can_start=native+pending<budget and available>=minimum,native_threads=native,
                pending_dispatcher_threads=pending,cpu_budget=budget,available_memory_bytes=available,
                minimum_available_memory_bytes=minimum)


def dispatch(args):
    if not 1<=args.workers<=4:raise ValueError('explicit one-to-four worker budget required')
    _,cells,_=load_registration(args.study)
    trigger=args.study/f'RESCUE_TRIGGER_OUTER_{args.outer}.json'
    if not trigger.exists():raise ValueError('registered rescue trigger required')
    selected=[c for c in cells if c['stage']=='rescue' and c['partition'].startswith(f'inner-{args.outer}-')]
    if len(selected)!=240:raise ValueError('complete registered rescue wave required')
    if set(json.loads(trigger.read_text())['required_rescue_cells'])!={c['id'] for c in selected}:
        raise ValueError('trigger coverage differs')
    resume=getattr(args,'resume_completed',False)
    completed,pins=completed_runs(args.study,selected,sha(args.binary_dir/'BUILD.json')) if resume else (set(),{})
    if not resume and any((args.study/'runs'/c['id']).exists() for c in selected):
        raise ValueError('existing rescue attempts require explicit reconciliation')
    directory=args.study/'dispatch/rescue'/f'outer-{args.outer}'
    if resume:
        previous=json.loads((directory/'STATUS.json').read_text())
        if Path(f"/proc/{previous['pid']}").exists():raise ValueError('previous rescue dispatcher still live')
        if any(Path(f'/proc/{pid}').exists() for pid in previous['active'].values()):
            raise ValueError('previous rescue children still live')
        directory=directory/('resume-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),driver_sha256=sha(__file__),plan_sha256=sha(args.study/'PLAN.json'),
                  build_sha256=sha(args.binary_dir/'BUILD.json'),trigger_sha256=sha(trigger),
                  workers=args.workers,required_cells=[c['id'] for c in selected],
                  worker_budget_file=str(args.worker_budget_file) if getattr(args,'worker_budget_file',None) else None,
                  drain_file=str(args.drain_file) if getattr(args,'drain_file',None) else None)
    if resume:identity.update(resumed_completed=sorted(completed),resume_evidence_sha256=pins)
    (directory/'RUN.json').write_bytes(encoded(identity))
    active={};ready=set();failure=None;worker_limit=args.workers;draining=False;budget_pin=None;admission=None
    selected.sort(key=lambda c:(NEURAL.index(c['family']),c['rescue'],c['seed'],c['partition']))
    while len(completed)<len(selected):
        for cell_id,(process,log) in list(active.items()):
            if process.poll() is None:continue
            log.close();del active[cell_id]
            try:
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
                    admission=worker_slot(active)
                    if not admission['can_start']:break
                    command=['python3','-B',str(REPO/'scripts/rev4_featpot/shaping_rescue_execution.py'),
                        '--study',str(args.study),'--binary-dir',str(args.binary_dir),
                        '--neutral-qualification',str(args.neutral_qualification),'--cell',cell['id']]
                    log=(directory/(cell['id']+'.log')).open('x')
                    active[cell['id']]=(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO),log)
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','neutral-qualification'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--worker-budget-file',type=Path);p.add_argument('--drain-file',type=Path)
    p.add_argument('--resume-completed',action='store_true')
    dispatch(p.parse_args())
