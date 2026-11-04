#!/usr/bin/env bash
# Final native research binaries. Caller owns heavy.lock and run-heavy scope.
set -euo pipefail
R14_ROOT=/var/tmp/byc-transforms-20261104
export CARGO_TARGET_DIR="$R14_ROOT/target"
export TMPDIR="$HOME/tmp/zensim-transforms"
mkdir -p "$R14_ROOT/bin" "$R14_ROOT/build" "$TMPDIR"
cargo build --release --locked -p zensim-validate \
  --bin zensim_mlp_train --bin bake_dial_refit --bin bake_verdict \
  --bin panel --bin byc_predict --bin predict_features_with_bake
for R14_BIN in zensim_mlp_train bake_dial_refit bake_verdict panel byc_predict predict_features_with_bake; do
  install -m755 "$CARGO_TARGET_DIR/release/$R14_BIN" "$R14_ROOT/bin/$R14_BIN"
done
cargo build --release --locked -p zensim --features custom-profiles,training,classification --example serve_custom_bake
install -m755 "$CARGO_TARGET_DIR/release/examples/serve_custom_bake" "$R14_ROOT/bin/serve_custom_bake"
cargo build --release --locked --manifest-path tools/zensim-viz/Cargo.toml
install -m755 "$CARGO_TARGET_DIR/release/zensim-viz-export" "$R14_ROOT/bin/zensim-viz-export"
RUSTFLAGS='--cfg steerfix_instrument' cargo test --release --locked -p zensim \
  --features custom-profiles,training,classification --lib --no-run --message-format=json \
  > "$R14_ROOT/build/steer-artifacts.jsonl"
python3 - <<'PY'
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/var/tmp/byc-transforms-20261104')
artifacts=[json.loads(line) for line in (root/'build/steer-artifacts.jsonl').read_text().splitlines()]
paths=[Path(a['executable']) for a in artifacts if a.get('reason')=='compiler-artifact'
       and a.get('target',{}).get('name')=='zensim' and a.get('profile',{}).get('test') and a.get('executable')]
assert len(paths)==1,paths
shutil.copy2(paths[0],root/'bin/steer_instrument')
head=Path('/var/tmp/byc-linear-20261104/extra/heads/bin/prune_forward_bench')
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(head)=='aa409c8e689323f7ed0f8ca4f07fa34565d673efa18e26806f98572677af3a33'
shutil.copy2(head,root/'bin/prune_forward_bench')
source=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip()
assert subprocess.check_output(['jj','diff','--stat'],text=True).strip()=='','final source must be clean'
files=[p for folder in ('scripts/byconstruction','scripts/rev4_featpot') for p in Path(folder).glob('*.py')]
files+=[Path('Cargo.lock'),Path('Cargo.toml'),Path('tools/zensim-viz/Cargo.lock')]
pins=dict(source_commit=source,source_sha256={str(p):sha(p) for p in files},
    binary_sha256={p.name:sha(p) for p in (root/'bin').iterdir()},
    instrument_rustflags='--cfg steerfix_instrument',
    reused_head=dict(source_commit='3c35deae',sha256=sha(head),path=str(head)),
    rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
    screen_summary_sha256=sha(root/'screen/SUMMARY.json'))
(root/'SOURCE_PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
print(json.dumps(dict(source_commit=source,binaries=pins['binary_sha256'])),flush=True)
PY
