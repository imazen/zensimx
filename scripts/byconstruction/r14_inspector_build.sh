#!/usr/bin/env bash
# Rebuild inspection against transformed derived inputs; retain R12 timing bytes.
set -euo pipefail
export CARGO_TARGET_DIR=/var/tmp/byc-transforms-20261104/target
export TMPDIR="$HOME/tmp/zensim-transforms"
cargo build --release --locked -p zensim-validate --example prune_forward_bench
python3 - <<'PY'
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/var/tmp/byc-transforms-20261104')
dest=root/'inspection/prune_forward_bench'
dest.parent.mkdir(exist_ok=False)
shutil.copy2(root/'target/release/examples/prune_forward_bench',dest)
assert subprocess.check_output(['jj','log','-r','@','--no-graph','-T','empty'],text=True).strip()=='true'
record=dict(source_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),
    binary=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    example_sha256=hashlib.sha256(Path('zensim-validate/examples/prune_forward_bench.rs').read_bytes()).hexdigest(),
    role='native parameter inspection only; frozen R12 head timing instrument remains unchanged')
(root/'INSPECTOR_PINS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
