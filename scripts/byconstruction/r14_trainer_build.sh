#!/usr/bin/env bash
# Trainer-only successor; frozen serving binaries remain unchanged.
set -euo pipefail
export CARGO_TARGET_DIR=/var/tmp/byc-transforms-20261104/target
export TMPDIR="$HOME/tmp/zensim-transforms"
cargo build --release --locked -p zensim-validate --bin zensim_mlp_train
python3 - <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,subprocess
root=Path('/var/tmp/byc-transforms-20261104')
dest=root/'repairs/selected-callers/bin'
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
    purpose='declared subset load, materialized products before transforms; T2 support included',
    source_sha256={str(p):sha(p) for p in [Path('zensim-validate/src/bin/zensim_mlp_train.rs'),
        Path('zensim-validate/src/mlp_train/mod.rs'),Path('zensim-validate/src/mlp_train/r14_knees.rs'),
        Path('scripts/rev4_featpot/r14_transforms_fit.py'),Path('scripts/rev4_featpot/r14_learn_knees_fit.py')]})
(dest/'TRAINER_PINS.json').write_text(json.dumps(record,indent=2)+'\n')
active=root/'ACTIVE_TRAINER.json'
assert not active.exists(),'preserve prior trainer selection'
active.write_text(json.dumps(dict(schema='r14-selected-caller-loader-v1',trainer_dir=str(dest),
    trainer_pins_sha256=sha(dest/'TRAINER_PINS.json')),indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
