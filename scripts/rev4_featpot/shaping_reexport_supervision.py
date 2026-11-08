"""Uniform supplemental family evaluation, including future completed inner fits."""
from __future__ import annotations
import argparse,json,os,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from shaping_completion import encoded,sha
from shaping_execution import load_registration
from shaping_reexport import reexport
from shaping_evaluation import evaluate,NativeFeatureGateError,diagnostic_epochs
from shaping_supervision import complete_marker,preserve_failure
from shaping_convergence import report,UndefinedInnerComposite


# Preserve the loaded driver identity across later source edits.
DRIVER_SHA256=sha(__file__)

def sweep(args):
    family=getattr(args,'family','relu')
    if family not in ('relu','softplus'):raise ValueError('unsupported supplemental family')
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
    p.add_argument('--family',choices=('relu','softplus'),default='relu')
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        count=sweep(args)
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),
                    completed_supplemental_cells=count,stage='watching' if args.watch else 'sweep_finished',
                    driver_sha256=DRIVER_SHA256)
        (args.study/('REEXPORT_EVALUATOR_STATUS.json' if args.family=='relu' else 'SOFTPLUS_REEXPORT_EVALUATOR_STATUS.json')).write_bytes(encoded(status))
        print(json.dumps(status),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
