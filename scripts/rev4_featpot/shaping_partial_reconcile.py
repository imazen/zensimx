"""Repair checkpoint comparison supervision without restarting either trainer."""
import argparse,json,os,socket,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_partial_restart import authorize,compare_available,CELLS
from shaping_external_recovery import verify_attempt


def reconcile(study,cell):
    if socket.gethostname()!='i134':raise ValueError('approved worker required')
    row,pin=authorize(study,cell,'attempt-002');old=study/'runs'/cell/'attempt-001';dest=old.parent/'attempt-002'
    run=json.loads((dest/'RUN.json').read_text());verify_attempt(study,run,'attempt-002')
    if sha(old/'RESULT.json')!=row['old_result_sha256'] or sha(old/'RUN.json')!=row['old_run_sha256']:raise ValueError('old evidence drift')
    binary=study.parent/'study-bin/bake_dial_refit'
    while True:
        record=compare_available(old,dest,row,binary)
        result=dest/'RESULT.json'
        if result.exists():break
        process=Path('/proc')/str(run['pid'])/'cmdline'
        if process.exists() and str(dest).encode() not in process.read_bytes():raise ValueError('trainer PID identity differs')
        time.sleep(5)
    previous=json.loads(result.read_text())
    if (previous['exit_code']!=65 or previous['stage']!='failed'
        or 'run-heavy: done rc=0 ' not in (dest/'train.log').read_text()
        or len(record['compared'])!=row['completed_epochs_replayed']
        or not all(v['equal'] for v in record['compared'].values())
        or {p.name for p in (dest/'checkpoints').glob('ckpt_epoch*.bin')}!={f'ckpt_epoch{e:03}.bin' for e in range(240)}):
        raise ValueError('preserved attempt is not a successful native fit with fully matched prefix')
    with (dest/'RESULT.before-comparison-reconciliation.json').open('xb') as stream:stream.write(result.read_bytes())
    updated=dict(previous,stage='trained',exit_code=0,bake_sha256=sha(dest/'checkpoints/ckpt_epoch239.bin'),
        prefix_comparison_sha256=sha(dest/'PREFIX_COMPARISON_V2.json'),comparison_reconciliation_driver_sha256=sha(__file__),
        previous_result_sha256=sha(dest/'RESULT.before-comparison-reconciliation.json'),
        reconciliation='Native trainer succeeded; comparison helper had required absent final provenance metadata. All old prefixes now match.')
    temporary=dest/'RESULT.reconciled.writing';temporary.write_bytes(encoded(updated));temporary.replace(result)
    return updated


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--cell',choices=CELLS,required=True)
    args=p.parse_args();print(json.dumps(reconcile(args.study,args.cell)),flush=True)
