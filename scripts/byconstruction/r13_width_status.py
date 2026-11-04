"""Live R13 supervision from process identities, lock inode and packed artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

X = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(X/'scripts/rev4_featpot'))
from run_processes import process_tree, compute_lock
import r13_width as owner


def main():
    if sys.argv[1:] not in ([], ['--compact']):
        raise ValueError('expected no arguments or --compact')
    root=Path('/var/tmp/byc-width-20261104')
    now=datetime.now(timezone.utc).isoformat()
    lock=X/'.workongoing'
    if lock.exists() and 'codex-zensim-width' not in lock.read_text():
        raise ValueError('workspace lock belongs to another lane')
    lock.write_text(f'{now} codex-zensim-width supervising registered R13\n')
    active=json.loads((root/'ACTIVE_JOB.json').read_text()) if (root/'ACTIVE_JOB.json').exists() else {}
    tree=process_tree(active.get('pid',0))
    live=[{k:p[k] for k in ('pid','parent_pid','state','start_ticks','cpu_ticks')} |
          dict(executable=p['argv'][0] if p['argv'] else None) for p in tree if p['state'] not in ('Z','X')]
    done=[]
    for path in (root/'fits').glob('*/FIT.json'):
        d=json.loads(path.read_text())
        done.append(dict(name=path.parent.name,wall_seconds=d['wall_seconds'],capacity=d['capacity']))
    epochs={}
    for p in (root/'fits').glob('*/*/train.log'):
        matches=re.findall(r'epoch=(\d+)',p.read_text())
        epochs[str(p.relative_to(root))]=int(matches[-1]) if matches else None
    state=dict(checked=now,stage=active.get('stage'),active_pid=active.get('pid'),live=live,
               lock=compute_lock(active.get('pid',0)),completed_fits=len(done),required_fits=owner.required_fits(),
               fits=done,epochs=epochs,source_pins=str(root/'SOURCE_PINS.json'),
               owner_cost_cut=(root/'OWNER_COST_CUT.json').exists(),
               next_action='finish registered fits, freeze before assessment' if len(done)<owner.required_fits() else 'complete registered assessments')
    (root/'SUPERVISION.json').write_text(json.dumps(state,indent=2)+'\n')
    if sys.argv[1:] == ['--compact']:
        state = {k:state[k] for k in ('checked','stage','completed_fits','required_fits','epochs','lock')}
        state['trainer'] = [p for p in live if p['executable'] and p['executable'].endswith('zensim_mlp_train')]
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
