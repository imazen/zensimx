"""Read live R12 processes and artifacts; refresh only this lane's lock."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os

root = Path('/var/tmp/byc-linear-20261104')
now = datetime.now(timezone.utc).isoformat()
Path('.workongoing').write_text(f'{now} codex-zensim-linear supervising R12\n')
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try:
        cmd=p.joinpath('cmdline').read_bytes().decode().split('\0')
        exe=Path(cmd[0]).name
        if (exe in ('zensim_mlp_train','prune_forward_bench') or
                (exe.startswith('python') and len(cmd)>2 and any(Path(c).name.startswith(('r12_linear_', 'r12_pruning_', 'hdr_panel')) for c in cmd[1:3]))):
            processes.append(dict(pid=int(p.name),exe=exe,script=cmd[2] if exe.startswith('python') else None,
                                  start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19]))
    except (OSError, UnicodeError,IndexError): pass
status=dict(checked=now,stage=(root/'reads/STAGE').read_text() if (root/'reads/STAGE').exists() else (root/'STAGE').read_text(),fit_and_head_stage=(root/'STAGE').read_text(),complete=(root/'PROGRESS').read_text().splitlines(),
            live=processes,models={p.relative_to(root/'fits').parts[0]: str(p) for p in (root/'fits').glob('**/refit/production-f16.bin')})
status['extra_stage']=(root/'extra/STAGE').read_text() if (root/'extra/STAGE').exists() else 'queued build/tests'
status['pruning_feature_proof']=json.loads((root/'extra/proof/FEATURE_PROOF.json').read_text()) if (root/'extra/proof/FEATURE_PROOF.json').exists() else None
status['extra_read_stage']=(root/'extra/reads/STAGE').read_text() if (root/'extra/reads/STAGE').exists() else None
status['source_pins']=str(root/'SOURCE_PINS.json')
status['completed_read_receipts']={p.stem:json.loads(p.read_text())['rc'] for p in (root/'reads').glob('*.receipt.json')}
(root/'SUPERVISION.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
