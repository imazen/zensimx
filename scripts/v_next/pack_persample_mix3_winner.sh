#!/usr/bin/env bash
# EXP-PERSAMPLE-MIX3: pack the winning seed bake via zenpredict repack.
# Usage: pack_persample_mix3_winner.sh <seed>

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18"
ZENPREDICT="/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
WS="/home/lilith/work/zen/zensimx"
BAKE_VERDICT="${WS}/target/release/bake_verdict"

IN_BAKE="${OUT_DIR}/persample_mix3_s${SEED}_h128.bin"
PACKED="${OUT_DIR}/persample_mix3_s${SEED}_h128_packed.bin"

if [ ! -f "${IN_BAKE}" ]; then
    echo "ERROR: input bake not found: ${IN_BAKE}" >&2
    exit 1
fi

if [ ! -x "${ZENPREDICT}" ]; then
    echo "WARN: zenpredict binary not found at ${ZENPREDICT}"
    echo "  building..."
    (cd /home/lilith/work/zen/zenanalyze && cargo build --release --bin zenpredict -p zenpredict-bake) 2>&1 | tail -5
fi

"${ZENPREDICT}" repack "${IN_BAKE}" "${PACKED}" \
    --dtype i8 --zerobias 0.005 --compress --optimize

echo "--- packed: ${PACKED}"
ls -la "${IN_BAKE}" "${PACKED}"

# Verify CID22 SROCC drift after pack
ORIG_VERDICT="${OUT_DIR}/verdicts/persample_mix3_s${SEED}.md"
PACKED_VERDICT="${OUT_DIR}/verdicts/persample_mix3_s${SEED}_packed.md"
"${BAKE_VERDICT}" --bake "${PACKED}" \
    --corpora cid22,kadid,tid,konjnd,aic3 \
    --output "${PACKED_VERDICT}"

# Show CID22 SROCC drift
python3 <<EOF
import re
def cid22_srocc(p):
    if not p:
        return None
    try:
        txt = open(p).read()
    except FileNotFoundError:
        return None
    m = re.search(r'\|\s*CID22\s*\|\s*[0-9]+\s*\|\s*([0-9.]+)\s*\|', txt)
    return float(m.group(1)) if m else None

o = cid22_srocc("${ORIG_VERDICT}")
p = cid22_srocc("${PACKED_VERDICT}")
if o is not None and p is not None:
    print(f"  CID22 SROCC: orig={o:.4f} packed={p:.4f} drift={p-o:+.4f}")
EOF
