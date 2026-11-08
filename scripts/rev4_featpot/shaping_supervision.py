"""Evaluate completed registered inner neural fits serially, preserving failures."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from shaping_completion import encoded,sha
from shaping_evaluation import evaluate,NativeFeatureGateError,diagnostic_epochs
from shaping_convergence import report,UndefinedInnerComposite
from shaping_execution import load_registration


DRIVER_SHA256=sha(__file__)

def complete_marker(path,marker):
    if not path.exists():return False
    receipt=json.loads((path/marker).read_text())
    if receipt['report_sha256']!=sha(path/'REPORT.json'):raise ValueError('completed evaluation report drift')
    return True


def preserve_failure(path, record):
    """Completed failure evidence is immutable, including during archive reads."""
    payload=encoded(record)
    if path.exists():
        if path.read_bytes()!=payload:
            raise ValueError('existing failure evidence differs: '+str(path))
        return
    with path.open('xb') as stream:stream.write(payload)


def sweep(args):
    _,cells,_=load_registration(args.study)
    completed=0
    for result_path in sorted((args.study/'runs').glob('*/attempt-*/RESULT.json')):
        result=json.loads(result_path.read_text());cell=result['cell'];run=result_path.parent
        if getattr(args,'cells',None) is not None and cell['id'] not in args.cells:continue
        if cell['stage'] not in ('inner','fresh_control_inner','rescue','exposure_extension') or result['stage']!='trained':continue
        if cell['stage']=='exposure_extension':
            from shaping_extension import resolve
            registered=resolve(args.study,cell['id'])['cell']
            if cell!=registered:raise ValueError('extension cell differs')
            if cell not in cells:cells.append(cell)
        if cell not in cells:raise ValueError('unregistered training result')
        binaries=args.binary_dir
        if cell['stage'] in ('rescue','exposure_extension'):
            binaries=getattr(args,'rescue_binary_dir',None)
            if binaries is None:raise ValueError('rescue/extension evaluation requires its explicit pinned binary directory')
        values=dict(study=args.study,binary_dir=binaries,run=run)
        print(json.dumps(dict(cell=cell['id'],stage='evaluating',utc=datetime.now(timezone.utc).isoformat())),flush=True)
        failures=[]
        for epoch in (None,*diagnostic_epochs(result['epochs'])):
            if epoch is not None and epoch>=result['epochs']:continue
            dest=run/'evaluation' if epoch is None else run/'curve_evaluation'/f'epoch-{epoch:03}'
            marker='COMPLETE.json' if epoch is None else 'DIAGNOSTIC_COMPLETE.json'
            failed=dest/'FAILED_GATE.json'
            if not failed.exists():
                try:
                    if not complete_marker(dest,marker):evaluate(SimpleNamespace(**values,checkpoint=epoch))
                except NativeFeatureGateError:
                    if not failed.exists():raise
            if failed.exists():
                failures.append(dict(checkpoint=epoch,gate=str(failed),gate_sha256=sha(failed),
                                     finding=json.loads(failed.read_text())))
        if failures:
            # Failed checkpoints remain failed. No convergence/selection completion
            # receipt is emitted, but independent cells can still be evaluated.
            preserve_failure(run/'EVALUATION_FAILURES.json',dict(cell=cell,failures=failures,
                scientifically_complete=False,requires_investigation=True,outer_results_read=False))
            continue
        if not (run/'CONVERGENCE.json').exists():
            try:report(run)
            except UndefinedInnerComposite as error:
                evidence=[run/'RUN.json',run/'RESULT.json',*sorted((run/'curve_evaluation').glob('epoch-*/REPORT.json'))]
                preserve_failure(run/'CONVERGENCE_FAILURE.json',dict(cell=cell,reason=str(error),
                    scientifically_complete=False,requires_investigation=True,outer_results_read=False,
                    evidence_sha256={str(p):sha(p) for p in evidence}))
                continue
        completed+=1
    return completed


LATER_STAGES=('outer','fresh_control_outer','fresh_control_outer_matched','source_curve','row_curve','shuffle','human_only','full_fit_selection','full_fit')


def later_sweep(args):
    from shaping_evaluation import outer_context
    from shaping_reexport import reexport
    from shaping_convergence import training_report,save_training_report
    _,cells,_=load_registration(args.study);completed=0
    for path in sorted((args.study/'runs').glob('*/attempt-*/RESULT.json')):
        result=json.loads(path.read_text());cell=result['cell'];run_dir=path.parent
        if getattr(args,'cells',None) is not None and cell['id'] not in args.cells:continue
        if cell['stage'] not in LATER_STAGES or result['stage']!='trained':continue
        run=json.loads((run_dir/'RUN.json').read_text())
        if run['cell']!=cell:raise ValueError('later-stage run/result identity differs')
        context=outer_context(args.study,run,cells)
        if context is None:raise ValueError('later-stage evaluation lacks admission')
        family=context.get('resolved_family',context.get('variant',{}).get('resolved_family',cell['family']))
        identity=context['admission']['evaluation_representations'][family]
        repaired=identity['representation']=='research-relu-max-pairs-v1'
        values=dict(study=args.study,run=run_dir,checkpoint=None)
        if repaired:
            binaries=getattr(args,'repair_binary_dir',None);inspector=getattr(args,'repair_inspector',None)
            if binaries is None or inspector is None:raise ValueError('later ReLU requires explicit repaired runtime and inspector')
            dest=args.study/'reexport_max_pairs'/cell['id']/run_dir.name/'final'
            existing=args.study/'reexports'/cell['id']/run_dir.name/'final/RECEIPT.json'
            values.update(binary_dir=binaries,inspector=inspector)
        else:
            candidates=[args.binary_dir,getattr(args,'rescue_binary_dir',None),*getattr(args,'candidate_binary_dir',[])]
            matches={p for p in candidates if p is not None and sha(p/'BUILD.json')==run['build_sha256']}
            if len(matches)!=1:raise ValueError('one explicit matching later-stage binary directory required')
            values['binary_dir']=next(iter(matches));dest=run_dir/'evaluation'
        failure=dest/'FAILED_GATE.json'
        if not failure.exists() and not complete_marker(dest,'COMPLETE.json'):
            print(json.dumps(dict(cell=cell['id'],stage='later_final_evaluation',checkpoint=None)),flush=True)
            try:
                if repaired and not existing.exists():reexport(SimpleNamespace(**values))
                else:
                    if repaired:values['reexport_receipt']=existing
                    evaluate(SimpleNamespace(**values))
            except NativeFeatureGateError:
                if not failure.exists():raise
        if failure.exists():
            preserve_failure(dest.parent/'LATER_EVALUATION_FAILURE.json',dict(cell=cell,gate=str(failure),gate_sha256=sha(failure),scientifically_complete=False))
            continue
        if cell['stage'] in ('full_fit_selection','full_fit'):
            diagnostic=run_dir/'TRAIN_CONVERGENCE.json'
            if not diagnostic.exists():save_training_report(run_dir,dest/'REPORT.json')
            else:
                saved=json.loads(diagnostic.read_text());actual=training_report(run_dir,dest/'REPORT.json')
                if {k:v for k,v in saved.items() if k!='driver_sha256'}!={k:v for k,v in actual.items() if k!='driver_sha256'}:
                    raise ValueError('later-stage training diagnostic drift')
        completed+=1
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--rescue-binary-dir',type=Path)
    p.add_argument('--later-only',action='store_true')
    p.add_argument('--candidate-binary-dir',type=Path,action='append',default=[])
    p.add_argument('--repair-binary-dir',type=Path);p.add_argument('--repair-inspector',type=Path)
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        n=(later_sweep if args.later_only else sweep)(args)
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),evaluated_inner_neural=n,
                    stage='watching' if args.watch else 'sweep_finished',driver_sha256=DRIVER_SHA256)
        if args.later_only:
            status['evaluated_later_neural']=status.pop('evaluated_inner_neural')
        (args.study/('LATER_EVALUATOR_STATUS.json' if args.later_only else 'EVALUATOR_STATUS.json')).write_bytes(encoded(status));print(json.dumps(status),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
