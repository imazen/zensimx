#!/usr/bin/env bash
# Caller owns the shared heavy lock. Preserve the original five widths' binary.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TMPDIR=$HOME/tmp/zensim-width CARGO_TARGET_DIR=/var/tmp/byc-width-20261104/target
task_overlay_root=$PWD
task_source_root=/var/tmp/byc-width-20261104/w4-source
test ! -e "$task_source_root" || { echo 'preserve existing W4 source snapshot' >&2; exit 1; }
mkdir -p "$task_source_root"
task_git_directory=$(jj git root)
git --git-dir="$task_git_directory" archive ccdb5c7aa0d4b67d52f0e2874f325fbc84ddb11c | tar -x -C "$task_source_root"
patch --batch --forward -d "$task_source_root" -p1 < "$task_overlay_root/scripts/byconstruction/r13_width_w4_admission.patch"
cd "$task_source_root"
cargo fmt -p zensim-validate --check
cargo test --locked --release -p zensim-validate --bin zensim_mlp_train subjective_training_leg::tests
cargo build --locked --release -p zensim-validate --bin zensim_mlp_train
cargo clippy --locked --release -p zensim-validate --bin zensim_mlp_train
cd "$task_overlay_root"
python3 -B - <<'PY'
from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/var/tmp/byc-width-20261104');src=root/'target/release/zensim_mlp_train'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=root/'bin/zensim_mlp_train'
native_source=root/'w4-source/zensim-validate/src/subjective_training_leg.rs'
if sha(native_source)!='ed117852818cc2e2a9b1ad5eda2f410761efa27557bd048210297dbc4293c7fb':
    raise ValueError('W4 source differs from the verified original-source admission overlay')
expected=json.loads((root/'SOURCE_PINS.json').read_text())['bin_sha256'][str(original)]
if sha(original)!=expected:raise ValueError('original trainer changed')
dest=root/'w4-bin';dest.mkdir(exist_ok=True)
for p in (root/'bin').iterdir():
    if p.name!='zensim_mlp_train' and not (dest/p.name).exists():
        (dest/p.name).symlink_to(p.resolve())
trainer=dest/'zensim_mlp_train'
if trainer.exists():raise ValueError('preserve previous W4 admission binary')
shutil.copy2(src,trainer)
if sha(original)!=expected:raise ValueError('W4 build replaced the original trainer')
files=['scripts/rev4_featpot/subjective_training_leg.py',
       'scripts/rev4_featpot/r13_width_fit.py','Cargo.lock']
pins=dict(registration='397d67be879339e7014d3b29c2a99d3fe51df6b2',
          numerical_source_commit='ccdb5c7aa0d4b67d52f0e2874f325fbc84ddb11c',
          numerical_source_snapshot=str(root/'w4-source'),
          sole_rust_overlay='zensim-validate/src/subjective_training_leg.rs',
          native_source_sha256=sha(root/'w4-source/zensim-validate/src/subjective_training_leg.rs'),
          native_admission_patch_sha256=sha(Path('scripts/byconstruction/r13_width_w4_admission.patch')),
          native_binary=str(trainer),native_sha256=sha(trainer),original_trainer_sha256=expected,
          source_sha256={p:sha(Path(p)) for p in files},
          observed_working_change=subprocess.check_output(['jj','log','-r','@','--no-graph','-T','change_id'],text=True))
(root/'W4_SOURCE_PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
print(json.dumps(pins,indent=2))
PY
