#!/usr/bin/env bash
# Tuner v11 end-to-end pipeline (task #6, 2026-05-24).
#
# Phases:
#   1. Run 5 seeds of run_tuner_v11_seed.sh (sequential).
#   2. Run compare_tuner_v11_vs_v10.sh → summary markdown.
#   3. Pick median seed by CID22 SROCC.
#   4. Run cross-codec consistency measurement on median seed.
#   5. Print ship-gate scorecard with numbers filled in.
#
# Total wall: ~3 hr (5 × ~30 min trainer + ~5 min cross-codec).
#
# Usage: bash scripts/v_next/tuner_v11_full_pipeline.sh
# Env:
#   KBATCH (default 32)
#   LR_OVERRIDE (default 5.66e-3)
#   KONJND_AGG_WEIGHT (default 0.3)
#   SEEDS (default "1 2 3 4 5")

set -euo pipefail

REPO="/home/lilith/work/zen/zensimx"
OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24"
SEEDS="${SEEDS:-1 2 3 4 5}"

mkdir -p "${OUT_DIR}"

echo "===== Tuner v11 pipeline ====="
echo "OUT_DIR: ${OUT_DIR}"
echo "SEEDS:   ${SEEDS}"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Phase 1: 5-seed training.
for s in ${SEEDS}; do
    BAKE="${OUT_DIR}/tuner_v11_s${s}.bin"
    if [ -f "${BAKE}" ]; then
        echo "  seed ${s}: already exists at ${BAKE}, skipping"
        continue
    fi
    echo "===== Phase 1.${s}: training seed ${s} ====="
    bash "${REPO}/scripts/v_next/run_tuner_v11_seed.sh" "${s}"
    echo "  seed ${s} done: $(date -u +%s)"
done

# Phase 2: comparison.
echo
echo "===== Phase 2: comparison vs v10 ====="
bash "${REPO}/scripts/v_next/compare_tuner_v11_vs_v10.sh" "${OUT_DIR}"

# Phase 3: median seed by CID22 SROCC.
echo
echo "===== Phase 3: pick median seed by CID22 SROCC ====="
declare -A CID22_BY_SEED
for s in ${SEEDS}; do
    V="${OUT_DIR}/verdicts/v11_s${s}_verdict.md"
    [ -f "${V}" ] || continue
    SROCC=$(awk '
        /^## Summary/ { in_section = 1; next }
        in_section && /\| CID22 \|/ {
            n = split($0, fields, "|")
            gsub(/ /, "", fields[4])
            print fields[4]
            exit
        }
    ' "${V}")
    CID22_BY_SEED[$s]="${SROCC}"
    echo "  seed ${s}: CID22 SROCC = ${SROCC}"
done

# Sort numerically by SROCC, pick the middle one.
SORTED=$(for s in ${SEEDS}; do printf "%s\t%s\n" "${CID22_BY_SEED[$s]}" "$s"; done | sort -g)
N=$(echo "${SORTED}" | wc -l)
MEDIAN_IDX=$(( (N + 1) / 2 ))
MEDIAN_SEED=$(echo "${SORTED}" | sed -n "${MEDIAN_IDX}p" | cut -f2)
MEDIAN_SROCC=$(echo "${SORTED}" | sed -n "${MEDIAN_IDX}p" | cut -f1)
echo "  → median seed: ${MEDIAN_SEED} (CID22=${MEDIAN_SROCC})"

# Phase 4: cross-codec on median seed.
echo
echo "===== Phase 4: cross-codec consistency on median seed ====="
MEDIAN_BAKE="${OUT_DIR}/tuner_v11_s${MEDIAN_SEED}.bin"
python3 "${REPO}/scripts/v_next/measure_tuner_v10_cross_codec.py" \
    --bake "${MEDIAN_BAKE}" \
    --out-md "${OUT_DIR}/tuner_v11_s${MEDIAN_SEED}_cross_codec.md" \
    --out-parquet "${OUT_DIR}/tuner_v11_s${MEDIAN_SEED}_cross_codec.parquet"

# Phase 5: print scorecard.
echo
echo "===== Phase 5: ship-gate scorecard (median seed s${MEDIAN_SEED}) ====="
echo "  See: ${OUT_DIR}/tuner_v11_vs_v10_summary_$(date -u +%Y-%m-%d).md"
echo "  Cross-codec: ${OUT_DIR}/tuner_v11_s${MEDIAN_SEED}_cross_codec.md"
echo
echo "Now manually fill in benchmarks/v_tuner_v11_methodology_2026-05-24.md"
echo "with the seed table + verdict. Ship decision per ≥4/5 criteria."
echo
echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
