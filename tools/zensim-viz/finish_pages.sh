#!/usr/bin/env bash
# Resume after the retained native/exporter gates; browser gates always execute.
set -euo pipefail
root=$1
bake=$2
bindgen=$3
source_commit=${4:-$(jj log -r @- --no-graph -T commit_id)}
export TMPDIR="$HOME/tmp/zensim-viz" ZENSIM_FORMULA_REV=7
export ZENSIM_WORKBENCH_BAKES="A=$bake" ZENSIM_WORKBENCH_LABEL='zensim A revision 7 browser'
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/pages-feasibility-target}"
site=${VIZ_SITE_URL:-http://dev.lan:3311}
out="$root/pages-final-native"
test -f "$out/maps/native.json"
cargo run --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --example viz_parity -- "$bake" "$out/maps-replay"
# Re-run the corrected fixture reader and independent direct-score oracle.
test ! -e "$out/lens-replay.json"
cargo run --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --example viz_lens_parity -- "$out" "$out/lens-replay.json"
cmp "$out/lens.json" "$out/lens-replay.json"
python3 - "$out" "${VIZ_BASELINE_AUDIT:-$root/../2026-11-04/review1-audit-v2-F0/audit.json}" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);a=json.loads((out/'audit/audit.json').read_text());b=json.loads(Path(sys.argv[2]).read_text())
assert a['flags']==b['flags'] and a['thresholds']==b['thresholds'] and a['counts']==b['counts']
(out/'audit-preservation.json').write_text(json.dumps(dict(flags=len(a['flags']),all_flags_thresholds_and_counts_identical=True))+'\n')
PY
cargo clippy --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --all-targets -- -D warnings
export RUSTFLAGS="-C target-feature=+simd128 -C strip=symbols --remap-path-prefix=$HOME=deps --remap-path-prefix=$PWD=zensimx --remap-path-prefix=$CARGO_TARGET_DIR=target"
cargo build --release --locked --manifest-path zensim-workbench/Cargo.toml --features pages-viz --target wasm32-unknown-unknown --lib
mv "$root/pages-stage/pkg" "$root/pages-stage/pkg-before-native-finish"
"$bindgen" --target web --out-dir "$root/pages-stage/pkg" "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/zensim_workbench.wasm"
cp "$out/audit/audit.json" "$out/audit.json"
cp "$out/weights/weights.json" "$out/weights.json"
cp "$out/function/function.json" "$out/function.json"
if [[ -n "${VIZ_REFERENCE_JSON:-}" ]]; then
    cp "$VIZ_REFERENCE_JSON" "$out/reference.json"
else
    python3 -B tools/zensim-viz/reference.py --paper "$HOME/work/zen/zenpapers" --catalog "$out/weights.json" --out "$out/reference.json"
fi
python3 -B tools/zensim-viz/pages_build.py assemble --exports "$out" --reference "$out/reference.json" --bake "$bake" --packer "$root/pages-pack-target/release/zensim-viz-pack" --out "$out/archive-preview" --source "$source_commit"
python3 -B tools/zensim-viz/pages_build.py site --source "$source_commit" --out "$root/pages-stage" --archive-url ./bundle.imijset
printf '<!doctype html><title>Wasm parity probe</title>\n' > "$root/pages-stage/parity.html"
test -L "$root/pages-stage/fixtures"
ln -sfnT "$out/maps-replay" "$root/pages-stage/fixtures"
ln -sfnT "$out" "$root/pages-stage/banks"
ln -sfnT "$out/archive-preview/bundle.imijset" "$root/pages-stage/bundle.imijset"
export VIZ_PLAYWRIGHT="${VIZ_PLAYWRIGHT:-$HOME/tmp/viewer-test/node_modules/playwright}"
export VIZ_CHROMIUM="${VIZ_CHROMIUM:-$HOME/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome}"
node tools/zensim-viz/tests/pages-parity.mjs "$site" "$out/maps-replay/native.json" "$out/browser-map-parity.json"
node tools/zensim-viz/tests/pages-lens.mjs "$site" "$out/lens.json" "$out/browser-lens-parity.json"
node tools/zensim-viz/tests/pages-smoke.mjs "$site" "$out/screenshots-preview"
printf '%s\n' "$source_commit" > "$out/qualified-source.txt"
printf 'complete\n' > "$out/COMPLETE"
