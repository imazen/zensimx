"""Dispatch an explicit never-started inner subset without replaying existing fits."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha
from shaping_execution import load_registration,registered_subset
from shaping_host_assignment import check
from shaping_rescue_dispatch import resource_controls,reserve_worker,bind_worker,release_worker


def selected_cells(root,cells,requested,partition,family):
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


def verify_fit(root,binary_dir,cell):
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


def dispatch(args):
    host_budget=max(1,len(os.sched_getaffinity(0))-4)
    if not 1<=args.worker_capacity<=host_budget:raise ValueError('capacity exceeds host CPU budget')
    resource_controls(args)
    _,cells,_=load_registration(args.study)
    selected=selected_cells(args.study,cells,args.cell,args.partition,args.family)
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
                completed[cell_id]=verify_fit(args.study,args.binary_dir,cell)
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
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--partition',required=True)
    p.add_argument('--family',choices=('joint_shapes','analytic','relu','pretrained_shapes'),required=True)
    p.add_argument('--cell',action='append',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--worker-capacity',type=int,default=4)
    p.add_argument('--worker-budget-file',type=Path,required=True);p.add_argument('--drain-file',type=Path)
    p.add_argument('--interval',type=float,default=10)
    args=p.parse_args()
    if args.interval<=0:p.error('positive polling interval required')
    dispatch(args)


if __name__=='__main__':main()
