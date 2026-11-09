"""Execute registered doubled-exposure trials without changing their schedules."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import NEURAL,REPO,encoded,sha
from shaping_execution import load_registration
from shaping_extension import matrix,verify_decision


def admitted(root,outer):
    _,cells,_=load_registration(root);decisions={};pins={}
    from shaping_reduced_scope import neural_families
    required=neural_families(root,outer)
    for family in required:
        decisions[family],path=verify_decision(root,family,outer,cells);pins[str(path)]=sha(path)
    path=root/'extensions'/f'outer-{outer}'/'REGISTRATION.json';record=json.loads(path.read_text())
    expected=matrix(decisions,outer,candidate_families=required)
    if any(record[k]!=v for k,v in expected.items()) or record['decision_sha256']!=pins:
        raise ValueError('extension matrix or paired decisions differ')
    if record['plan_sha256']!=sha(root/'PLAN.json') or record['policy_sha256']!=sha(REPO/'docs/SHAPING_EXPOSURE_EXTENSION.md'):
        raise ValueError('extension plan or policy differs')
    return record,path


def existing(root,cell,registration,build):
    directory=root/'runs'/cell['id']
    if not directory.exists():return None
    attempts=list(directory.glob('attempt-*'))
    if len(attempts)!=1:raise ValueError('extension attempts require explicit reconciliation')
    run=attempts[0];identity=json.loads((run/'RUN.json').read_text())
    expected=dict(cell=cell,plan_sha256=sha(root/'PLAN.json'),build_sha256=build,
                  process_role='extension_controller',epochs=cell['phase_epochs'][-1],
                  training_exposure_pair_draws=cell['total_pair_draws'])
    if any(identity.get(k)!=v for k,v in expected.items()) or identity['resolution']['registration_sha256']!=registration or identity['resolution']['cell']!=cell:
        raise ValueError('existing extension identity differs')
    result=run/'RESULT.json'
    if result.exists():
        record=json.loads(result.read_text())
        if any(record.get(k)!=v for k,v in expected.items()) or record['resolution']!=identity['resolution'] or record['stage']!='trained' or record['exit_code']!=0:
            raise ValueError('existing extension fit failed or differs')
        checkpoint=run/'fit/checkpoints'/f"ckpt_epoch{identity['epochs']-1:03}.bin"
        if sha(checkpoint)!=record['bake_sha256']:raise ValueError('extension final checkpoint drift')
        return 'trained'
    command=Path(f"/proc/{identity['pid']}/cmdline").read_bytes().split(b'\0');args=[v.decode() for v in command if v]
    if not any(Path(v).name=='shaping_extension_execution.py' for v in args):raise ValueError('extension controller not live')
    for flag,value in [('--study',str(root)),('--cell',cell['id'])]:
        if args.count(flag)!=1 or args[args.index(flag)+1]!=value:raise ValueError('extension process command differs')
    return 'running'


def dispatch(args):
    if not 1<=args.workers<=4:raise ValueError('explicit one-to-four worker budget required')
    if not args.session.replace('-','').isalnum():raise ValueError('invalid session name')
    root=args.study/'dispatch/extensions'/f'outer-{args.outer}';root.mkdir(parents=True,exist_ok=True)
    with (root/'OWNER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=root/args.session;dest.mkdir(exist_ok=False)
        status=dict(pid=os.getpid(),outer=args.outer,workers=args.workers,driver_sha256=sha(__file__))
        def save(stage,**values):
            status.update(stage=stage,checked_utc=datetime.now(timezone.utc).isoformat(),**values)
            (dest/'STATUS.json').write_bytes(encoded(status))
        path=args.study/'extensions'/f'outer-{args.outer}'/'REGISTRATION.json'
        while not path.exists():
            if not args.watch_registration:raise ValueError('verified extension registration required')
            save('waiting_for_extension_registration');time.sleep(60)
        record,path=admitted(args.study,args.outer);selected=record['cells']
        pin=sha(path);build=sha(args.binary_dir/'BUILD.json')
        status.update(registration_sha256=pin,build_sha256=build,required_cells=selected)
        (dest/'RUN.json').write_bytes(encoded(status));active={};completed=set();failure=None
        while len(completed)<len(selected):
            for cell_id,(process,log) in list(active.items()):
                if process.poll() is None:continue
                log.close();del active[cell_id]
                if process.returncode:failure='extension child failed: '+cell_id
            external=set()
            try:
                for cell in selected:
                    if cell['id'] in active or cell['id'] in completed:continue
                    state=existing(args.study,cell,pin,build)
                    if state=='trained':completed.add(cell['id'])
                    elif state=='running':external.add(cell['id'])
            except (ValueError,OSError,KeyError) as error:failure=str(error)
            for cell in selected:
                if failure or len(active)+len(external)>=args.workers:break
                if cell['id'] in completed or cell['id'] in active or cell['id'] in external:continue
                if sha(path)!=pin or sha(args.binary_dir/'BUILD.json')!=build:
                    failure='extension dispatch pins changed';break
                command=['python3','-B','scripts/rev4_featpot/shaping_extension_execution.py','--study',str(args.study),
                         '--binary-dir',str(args.binary_dir),'--neutral-qualification',str(args.neutral_qualification),'--cell',cell['id']]
                log=(dest/(cell['id']+'.log')).open('x')
                active[cell['id']]=(subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT),log)
            save('failed' if failure and not active else 'supervising',completed=sorted(completed),
                 active={k:v[0].pid for k,v in active.items()},externally_supervised=sorted(external),failure=failure)
            if failure and not active:raise RuntimeError(failure)
            if len(completed)<len(selected):time.sleep(30)
        save('training_finished',completed=sorted(completed),active={},failure=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','neutral-qualification'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--outer',type=int,choices=range(3),required=True);p.add_argument('--workers',type=int,required=True)
    p.add_argument('--session',default='initial');p.add_argument('--watch-registration',action='store_true')
    dispatch(p.parse_args())
