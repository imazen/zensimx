"""Release registered pretrained trials only after matching parents finish."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_pretraining import matching_parent,completed_parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--partition',required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.workers<=4:raise ValueError('explicit 1–4 worker budget required')
    _,cells,_=load_registration(args.study)
    selected=[c for c in cells if c['stage']=='inner' and c['partition']==args.partition and '_pretrained_' in c['family']]
    if not selected:raise ValueError('no registered pretrained cells')
    selected.sort(key=lambda c:(c['setting'],c['seed'],c['family']))
    dest=args.study/'dispatch'/'pretrained'/args.partition
    dest.mkdir(parents=True,exist_ok=False)
    identity=dict(pid=os.getpid(),driver_sha256=sha(__file__),plan_sha256=sha(args.study/'PLAN.json'),
                  build_sha256=sha(args.binary_dir/'BUILD.json'),required_cells=[c['id'] for c in selected])
    (dest/'RUN.json').write_bytes(encoded(identity))
    active={};completed=set();failure=None
    while len(completed)<len(selected):
        for cell_id,(process,log) in list(active.items()):
            if process.poll() is None:continue
            log.close()
            if process.returncode:
                failure='pretrained child failed; inspect '+str(dest/(cell_id+'.log'))
                del active[cell_id];continue
            result=json.loads((args.study/'runs'/cell_id/'attempt-001/RESULT.json').read_text())
            if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('child did not finish training')
            completed.add(cell_id);del active[cell_id]
        for cell in selected:
            if failure is not None:break
            if len(active)>=args.workers:break
            if cell['id'] in active or cell['id'] in completed:continue
            if (args.study/'runs'/cell['id']).exists():raise ValueError('existing child requires explicit reconciliation')
            parent=matching_parent(cell,cells)
            if not list((args.study/'runs'/parent['id']).glob('attempt-*/RESULT.json')):continue
            try:completed_parent(args.study,cell,cells)
            except ValueError as error:
                failure=str(error);break
            cmd=['python3','-B','scripts/rev4_featpot/shaping_execution.py','--study',str(args.study),
                 '--binary-dir',str(args.binary_dir),'--partition',args.partition,'--inner-family',
                 '--family',cell['family'],'--cell',cell['id'],'--workers','1']
            log=(dest/(cell['id']+'.log')).open('w')
            active[cell['id']]=(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log)
        status=dict(**identity,checked_utc=datetime.now(timezone.utc).isoformat(),completed=sorted(completed),
                    active={key:value[0].pid for key,value in active.items()},
                    failure=failure,stage=('failed' if failure and not active else
                        'training_finished' if len(completed)==len(selected) else 'supervising'))
        (dest/'STATUS.json').write_bytes(encoded(status))
        if failure and not active:raise RuntimeError(failure)
        if len(completed)<len(selected):time.sleep(30)


if __name__=='__main__':main()
