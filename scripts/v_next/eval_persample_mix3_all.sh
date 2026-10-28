#!/usr/bin/env bash
# Eval pipeline for EXP-PERSAMPLE-MIX3 5-seed CI + bake_compare § A.9 verdicts.
#
# Runs bake_verdict on each seed bake, then bake_compare A=<seed> B=<two ships>
# (compression-trail + balanced-trail). Picks champion seed by CID22 SROCC,
# packs it for shipping if it passes the trail gates.

set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18"
WS="/home/lilith/work/zen/zensimx"
BAKE_VERDICT="${WS}/target/release/bake_verdict"
BAKE_COMPARE="${WS}/target/release/bake_compare"
ZENPREDICT="/home/lilith/work/zen/zenanalyze/target/release/zenpredict"

# Ship comparators
SHIP_COMPRESSION="/home/lilith/work/zen/zensimx/zensim/weights/v_compression_persample_2026-05-18.bin"
SHIP_BALANCED="/home/lilith/work/zen/zensimx/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin"

mkdir -p "${OUT_DIR}/verdicts" "${OUT_DIR}/compares_vs_compression" "${OUT_DIR}/compares_vs_balanced"

# Step 1: bake_verdict each seed
for SEED in 1 2 3 4 5; do
  BAKE="${OUT_DIR}/persample_mix3_s${SEED}_h128.bin"
  if [ ! -f "$BAKE" ]; then
    echo "SKIP seed=${SEED} (no bake at ${BAKE})"
    continue
  fi
  VERDICT="${OUT_DIR}/verdicts/persample_mix3_s${SEED}.md"
  echo "=== bake_verdict seed=${SEED} ==="
  "${BAKE_VERDICT}" --bake "${BAKE}" --output "${VERDICT}" 2>&1 | tail -5
done

# Step 2: aggregate 5-seed mean SROCC per corpus from verdicts
echo ""
echo "=== 5-seed aggregate ==="
python3 <<EOF
import re
from pathlib import Path
import statistics

OUT = Path("${OUT_DIR}")
seeds = [1, 2, 3, 4, 5]
corpora = ['cid22', 'kadid', 'tid', 'konjnd', 'aic3']

# Parse each verdict's summary table — one row per corpus
# Format:
#   | CID22 | 4292 | 0.8641 | 0.8614 | ... |
def parse_verdict(path):
    """Pull aggregate SROCC per corpus from bake_verdict Summary table."""
    text = path.read_text() if path.exists() else ""
    out = {}
    corp_re = {
        'cid22': r'\|\s*CID22\s*\|',
        'kadid': r'\|\s*KADIK10k\s*\|',
        'tid':   r'\|\s*TID2013\s*\|',
        'konjnd': r'\|\s*KonJND-1k\s*\(full\)\s*\|',
        'aic3':  r'\|\s*AIC-3 CTC\s*\|',
    }
    for corp, pat in corp_re.items():
        m = re.search(pat + r'\s*([0-9]+)\s*\|\s*([0-9.]+)\s*\|', text)
        if m:
            out[corp] = float(m.group(2))
    return out

rows = []
for s in seeds:
    p = OUT / "verdicts" / f"persample_mix3_s{s}.md"
    r = parse_verdict(p)
    if r:
        rows.append((s, r))

print(f"\n5-seed aggregate SROCC per corpus (n={len(rows)} seeds parsed):\n")
print(f"{'corpus':>10} | " + " | ".join(f"s{s}" for s, _ in rows) + " | mean")
print("-" * 90)
for corp in corpora:
    vals = [r.get(corp) for _, r in rows if r.get(corp) is not None]
    if not vals:
        print(f"{corp:>10} | " + " | ".join("-" for _ in rows) + " | -")
        continue
    mean = statistics.mean(vals)
    vals_str = " | ".join(f"{r.get(corp, '-'):.4f}" if r.get(corp) is not None else "-" for _, r in rows)
    print(f"{corp:>10} | {vals_str} | {mean:.4f}")
EOF

# Step 3: pick champion seed (best CID22 SROCC)
CHAMPION_SEED=$(python3 <<EOF
import re
from pathlib import Path
OUT = Path("${OUT_DIR}")
best_seed = None
best_cid22 = -1.0
for s in range(1, 6):
    p = OUT / "verdicts" / f"persample_mix3_s{s}.md"
    if not p.exists():
        continue
    txt = p.read_text()
    m = re.search(r'\|\s*CID22\s*\|\s*[0-9]+\s*\|\s*([0-9.]+)\s*\|', txt)
    if m:
        v = float(m.group(1))
        if v > best_cid22:
            best_cid22 = v
            best_seed = s
print(best_seed if best_seed else 3)
EOF
)
echo ""
echo "=== Champion seed: ${CHAMPION_SEED} ==="

# Step 4: bake_compare champion seed vs compression ship + balanced ship
CHAMP="${OUT_DIR}/persample_mix3_s${CHAMPION_SEED}_h128.bin"
if [ -f "${CHAMP}" ]; then
  echo "=== bake_compare vs compression ship ==="
  "${BAKE_COMPARE}" --a "${CHAMP}" --b "${SHIP_COMPRESSION}" \
    --output "${OUT_DIR}/compares_vs_compression/persample_mix3_s${CHAMPION_SEED}_vs_compression.md" \
    --bootstrap-resamples 1000 2>&1 | tail -10
  echo ""
  echo "=== bake_compare vs balanced ship ==="
  "${BAKE_COMPARE}" --a "${CHAMP}" --b "${SHIP_BALANCED}" \
    --output "${OUT_DIR}/compares_vs_balanced/persample_mix3_s${CHAMPION_SEED}_vs_balanced.md" \
    --bootstrap-resamples 1000 2>&1 | tail -10
fi

echo ""
echo "Reports written:"
echo "  ${OUT_DIR}/verdicts/persample_mix3_s*.md"
echo "  ${OUT_DIR}/compares_vs_compression/persample_mix3_s${CHAMPION_SEED}_vs_compression.md"
echo "  ${OUT_DIR}/compares_vs_balanced/persample_mix3_s${CHAMPION_SEED}_vs_balanced.md"
