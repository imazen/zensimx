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
                (exe.startswith('python') and len(cmd)>2 and any(Path(c).name.startswith('r12_linear_') for c in cmd[1:3]))):
            processes.append(dict(pid=int(p.name),exe=exe,script=cmd[2] if exe.startswith('python') else None,
                                  start_ticks=p.joinpath('stat').read_text().split(') ',1)[1].split()[19]))
    except (OSError, UnicodeError,IndexError): pass
status=dict(checked=now,stage=(root/'STAGE').read_text(),complete=(root/'PROGRESS').read_text().splitlines(),
            live=processes,models={p.parent.parent.name: str(p) for p in (root/'fits').glob('*/refit/production-f16.bin')})
(root/'SUPERVISION.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
