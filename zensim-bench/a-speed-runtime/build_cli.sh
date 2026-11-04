#!/usr/bin/env bash
# Compile the original CLI source against the frozen instrument dependency graph.
# The caller holds the heavy lock and supplies run-heavy; no metric source changes.
set -euo pipefail
root=$(realpath "${1:?artifact root}")
runtime=$(realpath "${2:-$root/harness/a-speed-runtime}")
validation="$root/harness/cli-validation"
test ! -e "$validation" || { echo "preserve existing CLI validation directory" >&2; exit 2; }
mkdir -p "$validation"
python3 - "$runtime" "$validation" <<'PY'
from pathlib import Path
import shutil
import sys
runtime, validation = map(Path, sys.argv[1:])
manifest = (runtime / "Cargo.toml").read_text().split("[[bin]]", 1)[0]
manifest += '''
[[bin]]
name = "margarine"
path = "../../margarine/margarine.rs"

[profile.release]
lto = "thin"
codegen-units = 1

[lints.rust]
unexpected_cfgs = "allow"
'''
(validation / "Cargo.toml").write_text(manifest)
for name in ("Cargo.lock", "build.rs"):
    shutil.copyfile(runtime / name, validation / name)
PY
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$runtime/target}"
cargo build --release --locked --manifest-path "$validation/Cargo.toml" --bin margarine
cp "$CARGO_TARGET_DIR/release/margarine" "$root/bin/margarine-cli"
sha256sum "$root/bin/margarine-cli"
