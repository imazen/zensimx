"""R14 supervision from live process identities, lock holders and artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

X=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(X/'scripts/rev4_featpot'))
from run_processes import process_tree, compute_lock


def main():
    root=Path('/var/tmp/byc-transforms-20261104')
    now=datetime.now(timezone.utc).isoformat()
    lock=X/'.workongoing'
    if lock.exists() and 'codex-zensim-transforms' not in lock.read_text():
        raise ValueError('workspace lock belongs to another lane')
    lock.write_text(f'{now} codex-zensim-transforms supervising registered R14\n')
    active=json.loads((root/'ACTIVE_JOB.json').read_text()) if (root/'ACTIVE_JOB.json').exists() else {}
    tree=process_tree(active.get('pid',0))
    # A recycled PID cannot establish that this recorded job is still running.
    identity=any(p['pid']==active.get('pid') and p['argv']==active.get('command') for p in tree)
    live=[{k:p[k] for k in ('pid','parent_pid','state','start_ticks','cpu_ticks')} |
          dict(executable=p['argv'][0] if p['argv'] else None)
          for p in tree if identity and p['state'] not in ('Z','X')]
    completed=[]
    for p in sorted((root/'fits').glob('*/FIT.json')):
        d=json.loads(p.read_text())
        completed.append(dict(name=p.parent.name,wall_seconds=d['wall_seconds'],capacity=d['capacity']))
    epochs={}
    for p in (root/'fits').glob('*/*/train.log'):
        matches=re.findall(r'epoch=(\d+)',p.read_text())
        epochs[str(p.relative_to(root))]=int(matches[-1]) if matches else None
    receipt=root/'logs'/f"{active.get('stage')}.receipt.json"
    result=json.loads(receipt.read_text()) if receipt.exists() else None
    lock_state=compute_lock(active.get('pid',0))
    state=('running' if lock_state['held_by_worker_ancestor'] else 'waiting-for-shared-lock') if live else 'exited; inspect receipt'
    if result:state='complete' if result['rc']==0 else 'failed'
    report=dict(checked=now,stage=active.get('stage'),state=state,live=live,lock=lock_state,
        completed_fits=len(completed),required_T1_fits=3,fits=completed,epochs=epochs,
        source_pins_present=(root/'SOURCE_PINS.json').exists(),receipt=result,
        next_action='finish registered fits, then development feasibility and frozen reads')
    (root/'SUPERVISION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
