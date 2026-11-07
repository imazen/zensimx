"""Reconcile the retained zero-step knot-admission failure and successful retry."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_execution import load_registration


def reconcile(root,cell_id):
    _,cells,_=load_registration(root)
    cell=next(c for c in cells if c['id']==cell_id)
    if cell['stage']!='inner' or '_pretrained_' not in cell['family']:raise ValueError('registered pretrained inner cell required')
    directory=root/'runs'/cell_id
    if {p.name for p in directory.glob('attempt-*')}!={'attempt-001','attempt-002'}:raise ValueError('expected one preserved failure and one retry')
    failed=directory/'attempt-001';done=directory/'attempt-002'
    if {p.name for p in failed.iterdir()}!={'keep.txt','checkpoints','ORCHESTRATION_FAILURE.json'}:
        raise ValueError('failed attempt has additional artifacts; inspect them')
    if list((failed/'checkpoints').iterdir()):raise ValueError('failed attempt contains checkpoints')
    failure=json.loads((failed/'ORCHESTRATION_FAILURE.json').read_text())
    if failure['cell']!=cell_id or failure['stage']!='failed_before_training' or failure['optimizer_steps']!=0:
        raise ValueError('failure was not the zero-step admission rejection')
    if failure['error']!='prefix shape layout or knots differ' or failure['repair_commit']!='822318632b57':
        raise ValueError('this reconciliation only covers the verified f32 knot repair')
    result=json.loads((done/'RESULT.json').read_text());run=json.loads((done/'RUN.json').read_text())
    if result['cell']!=cell or run['cell']!=cell or result['stage']!='trained' or result['exit_code']!=0:
        raise ValueError('retry did not finish the registered fit')
    if run['plan_sha256']!=sha(root/'PLAN.json') or result['epochs']!=120:raise ValueError('retry protocol differs')
    if sha(done/'checkpoints/ckpt_epoch119.bin')!=result['bake_sha256']:raise ValueError('retry final model drift')
    proc=Path('/proc')/str(result['pid'])/'cmdline'
    if proc.exists() and str(done).encode() in proc.read_bytes():raise ValueError('retry process still live')
    c=json.loads((done/'CONVERGENCE.json').read_text())
    if c['cell']!=cell or c['outer_results_read']:raise ValueError('retry diagnostics differ')
    for path,pin in c['report_sha256'].items():
        if sha(path)!=pin:raise ValueError('retry diagnostic drift')
    marker=done/'evaluation/COMPLETE.json'
    if json.loads(marker.read_text())['report_sha256']!=sha(done/'evaluation/REPORT.json'):raise ValueError('retry report drift')
    record=dict(successful_attempt='attempt-002',failed_attempt='attempt-001',failed_optimizer_steps=0,
                failed_attempt_files=['keep.txt','ORCHESTRATION_FAILURE.json'],failed_checkpoints=[],
                completed_run_pid=result['pid'],completed_run_not_live=True,all_attempts_preserved=True,
                training_receipt_sha256=sha(done/'RESULT.json'),failed_receipt_sha256=sha(failed/'ORCHESTRATION_FAILURE.json'),
                driver_sha256=sha(__file__))
    with (directory/'ARCHIVE_RECONCILIATION.json').open('xb') as out:out.write(encoded(record))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--cell',required=True);a=p.parse_args();reconcile(a.study,a.cell)
