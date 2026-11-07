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
from shaping_evaluation import evaluate,NativeFeatureGateError
from shaping_convergence import report
from shaping_execution import load_registration


DRIVER_SHA256=sha(__file__)

def complete_marker(path,marker):
    if not path.exists():return False
    receipt=json.loads((path/marker).read_text())
    if receipt['report_sha256']!=sha(path/'REPORT.json'):raise ValueError('completed evaluation report drift')
    return True


def sweep(args):
    _,cells,_=load_registration(args.study)
    completed=0
    for result_path in sorted((args.study/'runs').glob('*/attempt-*/RESULT.json')):
        result=json.loads(result_path.read_text());cell=result['cell'];run=result_path.parent
        if cell not in cells:raise ValueError('unregistered training result')
        if cell['stage'] not in ('inner','fresh_control_inner') or result['stage']!='trained':continue
        values=dict(study=args.study,binary_dir=args.binary_dir,run=run)
        print(json.dumps(dict(cell=cell['id'],stage='evaluating',utc=datetime.now(timezone.utc).isoformat())),flush=True)
        failures=[]
        for epoch in (None,49,99,119,149,199,219,239):
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
            (run/'EVALUATION_FAILURES.json').write_bytes(encoded(dict(cell=cell,failures=failures,
                scientifically_complete=False,requires_investigation=True,outer_results_read=False)))
            continue
        if not (run/'CONVERGENCE.json').exists():report(run)
        completed+=1
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--binary-dir',type=Path,required=True)
    p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        n=sweep(args)
        status=dict(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),evaluated_inner_neural=n,
                    stage='watching' if args.watch else 'sweep_finished',driver_sha256=DRIVER_SHA256)
        (args.study/'EVALUATOR_STATUS.json').write_bytes(encoded(status));print(json.dumps(status),flush=True)
        if not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
