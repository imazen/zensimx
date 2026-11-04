#!/usr/bin/env bash
# Trainer-only successor; frozen serving binaries remain unchanged.
set -euo pipefail
export R14_TRAINER_KIND=${1:-selected-callers}
case "$R14_TRAINER_KIND" in selected-callers|t2-bounds) ;; *) exit 2 ;; esac
export CARGO_TARGET_DIR=/var/tmp/byc-transforms-20261104/target
export TMPDIR="$HOME/tmp/zensim-transforms"
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/var/tmp/byc-transforms-20261104')
original=json.loads((root/'SOURCE_PINS.json').read_text())
for name in ('Cargo.toml','Cargo.lock','tools/zensim-viz/Cargo.lock'):
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==original['source_sha256'][name],f'frozen dependency graph changed: {name}'
assert subprocess.check_output(['rustc','--version'],text=True).strip()==original['rustc'],'frozen compiler changed'
if os.environ['R14_TRAINER_KIND']=='t2-bounds':
    selected=json.loads((root/'ACTIVE_TRAINER.json').read_text())
    predecessor=json.loads((Path(selected['trainer_dir'])/'TRAINER_PINS.json').read_text())
    changed=subprocess.check_output(['jj','diff','--from',predecessor['source_commit'],'--to','@-',
        '--name-only','zensim-validate/src'],text=True).splitlines()
    assert set(changed)=={'zensim-validate/src/mlp_train/r14_knees.rs'},f'unreviewed native trainer changes: {changed}'
PY
cargo build --release --locked -p zensim-validate --bin zensim_mlp_train
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/var/tmp/byc-transforms-20261104')
kind=os.environ['R14_TRAINER_KIND']
dest=root/f'repairs/{kind}/bin'
dest.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins=json.loads((root/'SOURCE_PINS.json').read_text())
for name,digest in pins['binary_sha256'].items():
    original=root/'bin'/name
    assert sha(original)==digest,'frozen original binary changed'
    if name=='zensim_mlp_train':shutil.copy2(root/'target/release'/name,dest/name)
    else:os.symlink(original,dest/name)
assert subprocess.check_output(['jj','log','-r','@','--no-graph','-T','empty'],text=True).strip()=='true'
source=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip()
record=dict(registration='5b5a013482db4944042883995ad6892f92244c69',source_commit=source,
    trainer_sha256=sha(dest/'zensim_mlp_train'),serving_pins_sha256=sha(root/'SOURCE_PINS.json'),
    purpose=f'R14 trainer successor: {kind}',
    source_sha256={str(p):sha(p) for p in [Path('zensim-validate/src/bin/zensim_mlp_train.rs'),
        Path('zensim-validate/src/mlp_train/mod.rs'),Path('zensim-validate/src/mlp_train/r14_knees.rs'),
        Path('scripts/rev4_featpot/r14_transforms_fit.py'),Path('scripts/rev4_featpot/r14_learn_knees_fit.py')]})
(dest/'TRAINER_PINS.json').write_text(json.dumps(record,indent=2)+'\n')
active=root/('T2_TRAINER.json' if kind=='t2-bounds' else 'ACTIVE_TRAINER.json')
assert not active.exists(),'preserve prior trainer selection'
active.write_text(json.dumps(dict(schema='r14-t2-f32-bounds-v1' if kind=='t2-bounds' else 'r14-selected-caller-loader-v1',trainer_dir=str(dest),
    trainer_pins_sha256=sha(dest/'TRAINER_PINS.json')),indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
