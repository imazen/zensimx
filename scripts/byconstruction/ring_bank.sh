#!/bin/bash
# Re-extract every training/assessment set of the v2e29 root under a measurement arm (ring_root.py ARMS: hpm =
# ZENSIM_RING_FORM=hpm, linpyr = ZENSIM_PYRAMID=linear, rev7 = formula revision 7 with raw ringing), report each
# against the stored Rev5 bank (slots the arm does not replace must be bit-identical, and rev7's replaced values close;
# ring_root.py build enforces both), then the E15 coverage pool, then the derived root.
# Exploration record: benchmarks/byconstruction_2026-10-22.md.
# Usage: ring_bank.sh EXTRACTOR BUILD_COMMIT OUT_DIR THREADS [ARM]
set -euo pipefail
BIN=$1; COMMIT=$2; OUT=$3; THREADS=${4:-24}; ARM=${5:-hpm}
HERE=$(cd "$(dirname "$0")" && pwd)
FEATPOT=$HERE/../rev4_featpot
REV=5
case $ARM in
  hpm) ENV=ZENSIM_RING_FORM=hpm; ERA=rev5_localwin_ringhpm ;;
  linpyr) ENV=ZENSIM_PYRAMID=linear; ERA=rev5_localwin_linpyr ;;
  rev7) ENV=ZENSIM_FORMULA_REV=7; ERA=rev7_rowlanes; REV=7 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
for s in konfig_train konfig_val tid2013 kadid_select kadid_train cid22_a25 cid22_train safesyn; do
  if [ -f "$OUT/bank/$s/_MANIFEST.json" ]; then echo "skip $s"; continue; fi
  echo "== $s start $(date -u +%T)"
  env "$ENV" nice -n 15 python3 "$FEATPOT/rev5_bank.py" extract "$s" --bin "$BIN" --build-commit "$COMMIT" \
    --revision "$REV" --era "$ERA" --out "$OUT/bank" --threads "$THREADS" > "$OUT/bank_$s.log" 2>&1
  python3 "$FEATPOT/rev5_bank.py" verify-against "$s" --out "$OUT/bank" --reference /var/tmp/rev5-featbank \
    > "$OUT/verify_$s.json" || true
  echo "== $s done $(date -u +%T) $(cat "$OUT/verify_$s.json")"
done
if [ ! -f "$OUT/pool/_MANIFEST.json" ]; then
  echo "== pool start $(date -u +%T)"
  nice -n 15 python3 "$HERE/ring_root.py" pool --arm "$ARM" --bin "$BIN" --out "$OUT/pool" --threads "$THREADS"
fi
echo "== root start $(date -u +%T)"
python3 "$HERE/ring_root.py" build --arm "$ARM" --bank "$OUT/bank" --pool "$OUT/pool" --out "$OUT/root"
echo ALL-DONE
