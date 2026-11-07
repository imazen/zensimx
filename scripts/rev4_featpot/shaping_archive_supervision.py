"""Archive immutable completed inner cells; never reclaim or copy active attempts."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import REPO,encoded,sha
from shaping_execution import load_registration


def eligible(directory,cells,plan_pin):
    attempts=list(directory.glob('attempt-*'))
    # Multiple attempts need explicit reconciliation of all retained failures.
    if len(attempts)!=1:return False
    run=attempts[0]
    if not all((run/name).is_file() for name in ('RUN.json','RESULT.json','CONVERGENCE.json','evaluation/COMPLETE.json')):return False
    identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
    cell=identity['cell']
    if cell not in cells or cell['id']!=directory.name:raise ValueError('archive cell identity differs')
    if cell['stage'] not in ('inner','fresh_control_inner') or result['stage']!='trained':return False
    if identity['plan_sha256']!=plan_pin or result['cell']!=cell or result['exit_code']!=0:raise ValueError('archive fit differs')
    c=json.loads((run/'CONVERGENCE.json').read_text())
    if c['cell']!=cell or c['outer_results_read']:raise ValueError('archive convergence identity differs')
    for name,pin in c['report_sha256'].items():
        if sha(name)!=pin:raise ValueError('archive curve drift')
    final=run/'evaluation/REPORT.json'
    if json.loads((run/'evaluation/COMPLETE.json').read_text())['report_sha256']!=sha(final):raise ValueError('archive final report drift')
    if (run/'EVALUATION_FAILURES.json').exists():return False
    return True


def sweep(args):
    _,cells,_=load_registration(args.study);completed=0
    logs=args.study/'archive_supervision';logs.mkdir(exist_ok=True)
    for directory in sorted((args.study/'runs').iterdir()):
        if not directory.is_dir():continue
        if (directory/'MIRROR_VERIFIED.json').exists():continue
        if not eligible(directory,cells,sha(args.study/'PLAN.json')):continue
        dest=args.destination/('inner-'+directory.name)
        if dest.exists() or (directory/'SHA256SUMS').exists():
            raise ValueError('unfinished archive requires explicit reconciliation: '+str(directory))
        log=logs/(directory.name+'.log')
        command=['python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),
                 '--root',str(directory),'--nfs','--cold-readback','--destination',str(dest)]
        print(json.dumps(dict(cell=directory.name,stage='archiving',destination=str(dest))),flush=True)
        with log.open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        receipt=json.loads((directory/'MIRROR_VERIFIED.json').read_text())
        if not receipt['all_artifact_sha256_verified'] or receipt['destination']!=str(dest):raise ValueError('archive readback incomplete')
        if sha(directory/'SHA256SUMS')!=sha(dest/'SHA256SUMS'):raise ValueError('archive manifest differs')
        completed+=1
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','destination'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        completed=sweep(args)
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
                    newly_archived=completed,stage='watching' if args.watch else 'sweep_finished',driver_sha256=sha(__file__))
        (args.study/'ARCHIVE_SUPERVISOR_STATUS.json').write_bytes(encoded(status));print(json.dumps(status),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
