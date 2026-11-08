"""Uniform supplemental family evaluation, including future completed inner fits."""
from __future__ import annotations
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import max_pair_families,encoded,sha
from shaping_execution import load_registration
from shaping_reexport import reexport
from shaping_evaluation import evaluate,NativeFeatureGateError,diagnostic_epochs
from shaping_supervision import complete_marker,preserve_failure
from shaping_convergence import report,UndefinedInnerComposite


# Preserve the loaded driver identity across later source edits.
DRIVER_SHA256=sha(__file__)

def sweep(args):
    family=getattr(args,'family','relu')
    if family not in max_pair_families(args.study):raise ValueError('unsupported supplemental family')
    _,cells,_=load_registration(args.study);completed=0
    for path in sorted((args.study/'runs').glob('*/attempt-*/RESULT.json')):
        result=json.loads(path.read_text());cell=result['cell'];run=path.parent
        if getattr(args,'cells',None) is not None and cell['id'] not in args.cells:continue
        if cell['family']!=family or cell['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension') or result['stage']!='trained':continue
        if cell['stage']=='exposure_extension':
            from shaping_extension import resolve
            registered=resolve(args.study,cell['id'])['cell']
            if cell!=registered:raise ValueError('extension cell differs')
            if cell not in cells:cells.append(cell)
        if cell not in cells:raise ValueError('unregistered result')
        root=args.study/'reexport_max_pairs'/cell['id']/run.name
        failures=[]
        for epoch in (None,*diagnostic_epochs(result['epochs'])):
            if epoch is not None and epoch>=result['epochs']:continue
            name='final' if epoch is None else f'epoch-{epoch:03}'
            dest=root/name;marker='COMPLETE.json' if epoch is None else 'DIAGNOSTIC_COMPLETE.json'
            failure=dest/'FAILED_GATE.json'
            if not failure.exists() and not complete_marker(dest,marker):
                print(json.dumps(dict(cell=cell['id'],checkpoint=epoch,stage='supplemental_evaluation')),flush=True)
                values=SimpleNamespace(study=args.study,run=run,binary_dir=args.binary_dir,
                                       inspector=args.inspector,checkpoint=epoch,analytic_diagnostic=family=='softplus')
                existing=args.study/'reexports'/cell['id']/run.name/name/'RECEIPT.json'
                try:
                    if existing.exists():
                        # Resume only an already completed native conversion;
                        # evaluator revalidates its source and artifact hashes.
                        values.reexport_receipt=existing;evaluate(values)
                    else:reexport(values)
                except NativeFeatureGateError:
                    if not failure.exists():raise
            if failure.exists():failures.append(dict(path=str(failure),sha256=sha(failure)))
        if failures:
            preserve_failure(root/'EVALUATION_FAILURES.json',dict(cell=cell,failures=failures,scientifically_complete=False))
            continue
        if not (root/'CONVERGENCE.json').exists():
            try:
                if family=='softplus':report(run,root,analytic_diagnostic=True)
                else:report(run,root)
            except UndefinedInnerComposite as error:
                evidence=[run/'RUN.json',run/'RESULT.json',*sorted(root.glob('epoch-*/REPORT.json'))]
                preserve_failure(root/'CONVERGENCE_FAILURE.json',dict(cell=cell,reason=str(error),
                    scientifically_complete=False,requires_investigation=True,outer_results_read=False,
                    evidence_sha256={str(p):sha(p) for p in evidence}))
                continue
        completed+=1
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','binary-dir','inspector'):p.add_argument('--'+key,type=Path,required=True)
    from shaping_stable_pairs import FAMILIES as qualified_families
    p.add_argument('--family',choices=('relu','softplus',*qualified_families),default='relu')
    p.add_argument('--cell',action='append',dest='cells',help='explicit registered wave; superseded historical cells stay preserved')
    p.add_argument('--status',type=Path,help='fresh per-wave status; refuses reuse')
    p.add_argument('--reserve-slot',action='store_true',help='reserve one shared CPU admission slot for this evaluation wave')
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    default=('REEXPORT_EVALUATOR_STATUS.json' if args.family=='relu' else 'SOFTPLUS_REEXPORT_EVALUATOR_STATUS.json'
             if args.family=='softplus' else 'STABLE_EXPORT_'+args.family.upper()+'_STATUS.json')
    status_path=args.status or args.study/default
    if args.status:
        status_path.parent.mkdir(parents=True,exist_ok=True)
        with status_path.open('x') as stream:stream.write(encoded(dict(stage='starting',pid=os.getpid())))
    if args.cells:
        _,registered,_=load_registration(args.study)
        wanted={c['id'] for c in registered if c['family']==args.family and c['stage'] in ('inner','fresh_control_inner','rescue','exposure_extension')}
        if len(args.cells)!=len(set(args.cells)) or not set(args.cells)<=wanted:raise ValueError('distinct registered family cells required')
    claimed=False;key='reexport-'+args.family+'-'+__import__('hashlib').sha256(encoded(args.cells or ['all'])).hexdigest()[:16]
    try:
        if args.reserve_slot:
            from shaping_rescue_dispatch import reserve_worker,release_worker
            while not reserve_worker(args.study,key)['can_start']:time.sleep(10)
            claimed=True
        while True:
            count=sweep(args)
            status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
                        completed_supplemental_cells=count,stage='watching' if args.watch else 'sweep_finished',
                        driver_sha256=DRIVER_SHA256,shared_slot=key if claimed else None)
            status_path.write_bytes(encoded(status));print(json.dumps(status),flush=True)
            if not args.watch:return
            time.sleep(30)
    finally:
        if claimed:release_worker(args.study,key)


if __name__=='__main__':main()
