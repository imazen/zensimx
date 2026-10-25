#!/usr/bin/env bash
# Score + every emitted feature, using the existing by-construction owners.
# Run inside run-heavy; build once with `just rev7-oracle-build` beforehand.
# The destination must be new. No corpus download, training or product switches.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$(realpath -m "${1:?usage: oracle_dumps.sh OUT_DIR}")
BIN_DIR=${REV7_BIN_DIR:-$ROOT/target/release}
G=$HOME/.cache/codec-corpus/v1/gb82
[[ ! -e $OUT ]] || { echo "destination exists: $OUT" >&2; exit 2; }
for binary in byc_score_timing byc_stub_bake bake_stamp_revision; do
  [[ -x $BIN_DIR/$binary ]] || { echo "missing $BIN_DIR/$binary; run just rev7-oracle-build" >&2; exit 2; }
done
for img in baby bulb city dog flowers; do
  [[ -f $G/$img-lossless.png ]] || { echo "missing codec-corpus gb82/$img-lossless.png" >&2; exit 2; }
done
while IFS= read -r name; do
  unset "$name"
done < <(compgen -v | sed -n '/^ZENSIM_/p')
mkdir -p "$OUT/bakes" "$OUT/dumps"
cd "$ROOT"
B=$ROOT/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin
"$BIN_DIR/bake_stamp_revision" "$B" 7 "$OUT/bakes/b_rev7.bin"
"$BIN_DIR/byc_stub_bake" --revision 7 --out "$OUT/bakes/full_rev7.bin"
"$BIN_DIR/byc_score_timing" --bake "$OUT/bakes/full_rev7.bin" --read-set > "$OUT/full_rev7.ids"
PERF_BITS_MODELS=<(printf 'r7b\tbake\t%s\nr7full\tbake\t%s\na\tprofile\tA\nb5\tbake\t%s\n' \
  "$OUT/bakes/b_rev7.bin" "$OUT/bakes/full_rev7.bin" "$B") \
  bash scripts/byconstruction/perf_bits.sh "$BIN_DIR/byc_score_timing" "$OUT/dumps"
# BHdr additionally uses the eleven same-pair absolute-nits conversions.
PERF_BITS_MODELS=<(printf 'bhdr\tprofile\tBHdr\n') PERF_BITS_HDR=1 \
  bash scripts/byconstruction/perf_bits.sh "$BIN_DIR/byc_score_timing" "$OUT/dumps"
# Repository-owned HDR fixtures (shared verbatim with the integration tests).
for tier in v4x v4 v3 scalar; do
  for threads in 1 8; do
    [[ $threads == 8 && $tier != v4x ]] && continue
    for fixture in native17 native97 chdr17 chdr64 chdr97 chdr256; do
      echo "BHdr fixture=$fixture tier=$tier threads=$threads"
      RAYON_NUM_THREADS=$threads "$BIN_DIR/byc_score_timing" --profile BHdr --hdr-fixture "$fixture" \
        --tier "$tier" --iters 1 --warmup 0 \
        --dump "$OUT/dumps/bhdr_${fixture}_${tier}_t${threads}.hex"
    done
  done
done
# Tier identity applies to canonical B5/R7; legacy A/BHdr preserve each tier's own bits.
for name in r7b r7full b5; do
  python3 scripts/byconstruction/perf_bits_compare.py --tiers "$OUT/dumps" --model "$name"
done
python3 - "$OUT" "$ROOT" "$BIN_DIR" "$G" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, root, binaries, corpus = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = [*sorted((root / 'zensim/src').rglob('*.rs')), root / 'Cargo.lock',
         root / 'zensim/tests/fixtures/hdr_pairs.rs',
         *sorted((root / 'zensim/weights').glob('*byid_2026-09-06.bin')),
         root / 'zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin']
manifest = {
    'source_files': {str(p.relative_to(root)): sha(p) for p in files},
    'binaries': {n: sha(binaries / n) for n in ['byc_score_timing', 'byc_stub_bake', 'bake_stamp_revision']},
    'inputs': {p.name: sha(p) for p in sorted(corpus.glob('*-lossless.png'))
               if p.stem.removesuffix('-lossless') in ['baby', 'bulb', 'city', 'dog', 'flowers']},
    'configurations': ['r7b', 'r7full', 'a', 'b5', 'bhdr'],
    'tiers_threads': ['v4x_t1', 'v4_t1', 'v3_t1', 'scalar_t1', 'v4x_t8'],
    'synthetic': ['97x63', '131x65', '255x129', '509x311', '1031x517', '2050x1030'],
    'hdr_fixtures': ['native17', 'native97', 'chdr17', 'chdr64', 'chdr97', 'chdr256'],
    'dump_counts': {n: len(list((out / 'dumps').glob(n + '_*.hex')))
                    for n in ['r7b', 'r7full', 'a', 'b5', 'bhdr']},
}
assert manifest['dump_counts'] == {'r7b': 55, 'r7full': 55, 'a': 55, 'b5': 55, 'bhdr': 85}
(out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
hashes = [f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file()]
(out / 'SHA256SUMS').write_text(''.join(hashes))
print(json.dumps(manifest['dump_counts'], sort_keys=True))
print(f'{len(hashes)} files hashed; manifest and SHA256SUMS in {out}')
PY
