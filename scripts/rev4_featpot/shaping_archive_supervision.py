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
    if cell['stage'] not in ('inner','fresh_control_inner'):return False
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



def supplemental_sweep(args):
    """Mirror only complete, uniformly pinned supplemental checkpoint panels."""
    from shaping_evaluation import diagnostic_epochs
    _,cells,_=load_registration(args.study);completed=0
    logs=args.study/'archive_supervision';logs.mkdir(exist_ok=True)
    for cpath in sorted((args.study/'reexport_max_pairs').glob('*/attempt-*/CONVERGENCE.json')):
        root=cpath.parent;convergence=json.loads(cpath.read_text());cell=convergence['cell']
        if cell['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension'):continue
        if cell['stage']=='exposure_extension':
            from shaping_extension import resolve
            registered=resolve(args.study,cell['id'])['cell']
            if cell!=registered:raise ValueError('supplemental extension identity differs')
            if cell not in cells:cells.append(cell)
        if cell not in cells:raise ValueError('supplemental archive cell not registered')
        run=args.study/'runs'/cell['id']/root.name
        identity=json.loads((run/'RUN.json').read_text());result=json.loads((run/'RESULT.json').read_text())
        if identity['cell']!=cell or result['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0:
            raise ValueError('supplemental archive training identity differs')
        if identity['plan_sha256']!=sha(args.study/'PLAN.json') or convergence['outer_results_read']:
            raise ValueError('supplemental archive scope differs')
        if convergence['original_training_run']!=str(run):raise ValueError('supplemental original run differs')
        expected={str(root/f'epoch-{e:03}'/'REPORT.json') for e in diagnostic_epochs(result['epochs'])}
        if set(convergence['report_sha256'])!=expected:raise ValueError('incomplete supplemental archive checkpoints')
        for name,pin in convergence['report_sha256'].items():
            if sha(name)!=pin:raise ValueError('supplemental report drift')
        for epoch in (None,*diagnostic_epochs(result['epochs'])):
            panel=root/('final' if epoch is None else f'epoch-{epoch:03}')
            path=panel/'REPORT.json';marker=panel/('COMPLETE.json' if epoch is None else 'DIAGNOSTIC_COMPLETE.json')
            if json.loads(marker.read_text())['report_sha256']!=sha(path):raise ValueError('supplemental marker drift')
            row=json.loads(path.read_text());actual=result['epochs']-1 if epoch is None else epoch
            if row['cell']!=cell or row['checkpoint_epoch']!=actual or row['outer_results_read']:
                raise ValueError('supplemental panel identity differs')
            receipt=Path(row['reexport_receipt'])
            if sha(receipt)!=row['reexport_receipt_sha256']:raise ValueError('supplemental receipt drift')
            conversion=json.loads(receipt.read_text())
            original=run/identity.get('fit_directory','.')/'checkpoints'/f'ckpt_epoch{actual:03}.bin'
            if conversion['original_sha256']!=sha(original) or conversion['training_build_sha256']!=identity['build_sha256']:
                raise ValueError('supplemental bake source drift')
            if conversion['evaluation_build_sha256']!=convergence['evaluation_build_sha256'] or row['evaluation_build_sha256']!=convergence['evaluation_build_sha256']:
                raise ValueError('supplemental runtime drift')
            for name,pin in conversion['file_sha256'].items():
                if sha(name)!=pin:raise ValueError('supplemental artifact drift')
            if row['checkpoint_sha256']!=sha(conversion['bake']):raise ValueError('supplemental scored bake drift')
        for kind,directory in [('panels',root.parent),('bakes',args.study/'reexports'/cell['id'])]:
            if (directory/'MIRROR_VERIFIED.json').exists():continue
            dest=args.destination/(f'supplemental-{kind}-'+cell['id'])
            if dest.exists() or (directory/'SHA256SUMS').exists():raise ValueError('unfinished supplemental archive needs reconciliation')
            log=logs/(f'supplemental-{kind}-'+cell['id']+'.log')
            command=['python3','-B',str(REPO/'scripts/byconstruction/dataset_mirror.py'),'--root',str(directory),'--nfs','--cold-readback','--destination',str(dest)]
            with log.open('x') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            receipt=json.loads((directory/'MIRROR_VERIFIED.json').read_text())
            if not receipt['all_artifact_sha256_verified'] or sha(directory/'SHA256SUMS')!=sha(dest/'SHA256SUMS'):
                raise ValueError('supplemental archive incomplete')
            completed+=1
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','destination'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--supplemental',action='store_true')
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        completed=supplemental_sweep(args) if args.supplemental else sweep(args)
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
                    newly_archived=completed,stage='watching' if args.watch else 'sweep_finished',driver_sha256=sha(__file__))
        (args.study/('SUPPLEMENTAL_ARCHIVE_STATUS.json' if args.supplemental else 'ARCHIVE_SUPERVISOR_STATUS.json')).write_bytes(encoded(status));print(json.dumps(status),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
