#!/usr/bin/env bash
# Optional detector PNG proof via the visualizer's map-only entry point.
set -euo pipefail
[[ $(hostname) = i270 ]] || { echo 'fusion maps require i270' >&2; exit 2; }
study=${1:?study root}
label=${2:?immutable label}
if [[ ${A_FUSION_MAP_LOCK_HELD:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- \
        env A_FUSION_MAP_LOCK_HELD=1 bash "$0" "$@"
fi
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$PATH"
export TMPDIR="$study/scratch" CARGO_TARGET_DIR="$study/target" CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0 RUSTFLAGS='-C force-frame-pointers=yes'
export ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1
cd "$study/source"
out="$study/$label/maps-proof"
maps="${A_FUSION_MAP_ROOT:-/mnt/tower/output/zensim/fusion-maps-2026-11-04}/$label"
mkdir -p "$out"
test ! -e "$out/zensim-viz-export"
test ! -e "$maps"
df -h .
cargo build --release --locked --manifest-path tools/zensim-viz/Cargo.toml 2>&1 | tee "$out/build.log"
cp "$CARGO_TARGET_DIR/release/zensim-viz-export" "$out/zensim-viz-export"
sha256sum "$out/zensim-viz-export" tools/zensim-viz/Cargo.lock > "$out/BINARY_SHA256SUMS"
rg --files zensim/src | sort | xargs sha256sum > "$out/SOURCE_SHA256SUMS"
mkdir -p "$maps"
for pair in 01 02 03 04 05 06; do
    mkdir -p "$maps/$pair/maps"
    ZENSIM_VIZ_MAP_DIR="$maps/$pair/maps" "$out/zensim-viz-export" atlas-pass \
        "$HOME/tmp/zensim-cost/production-f16.bin" \
        --reference "$study/fixtures/viz/$pair/reference.png" \
        --distorted "$study/fixtures/viz/$pair/distorted.png" \
        --out "$maps/$pair/features.json" --row-padding 7
 done
python3 - "$study" "$label" "$maps" <<'PY'
import hashlib,json,pathlib,sys
study=pathlib.Path(sys.argv[1]);label=sys.argv[2];maps=pathlib.Path(sys.argv[3]);files={};rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for pair in ('01','02','03','04','05','06'):
    fixture=study/'fixtures/viz'/pair
    expected={p.name:sha(p) for p in (fixture/'golden').glob('*.png')}
    actual={p.name:sha(p) for p in (maps/pair/'maps').glob('*.png')}
    assert len(expected)==len(actual)==348
    assert expected==actual,(pair,'detector PNG bytes differ from frozen baseline')
    bits=json.loads((maps/pair/'features.json').read_text())['bits']
    assert len(bits)==720
    rows.append(dict(pair=pair,detector_pngs=348,all_equal=True,layout_bits=bits,
                     reference_sha256=sha(fixture/'reference.png'),distorted_sha256=sha(fixture/'distorted.png')))
    files[pair]=actual
receipt=dict(label=label,pairs=6,detector_pngs=2088,all_pngs_equal=True,row_padding_pixels=7,
             rows=rows,files=files,output_directory=str(maps))
(study/label/'maps-proof/RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('detector PNG proof: 2088/2088 byte-identical, six strided pairs',flush=True)
PY
