#!/usr/bin/env bash
# Compare Tuner v11 5-seed CI vs V_tuner_v10 ship — task #6 ship-gate eval.
#
# Runs bake_verdict on each of v11 seeds 1..5 + v10, extracts aggregate
# SROCC/PWRC/Z-RMSE per corpus, emits a side-by-side markdown table.
# Also re-runs the cross-codec consistency baseline on the median-by-
# CID22 v11 seed so we can compare to task #1's v10 numbers.
#
# Usage: bash scripts/v_next/compare_tuner_v11_vs_v10.sh [<out_dir>]
#   out_dir defaults to /mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24

set -euo pipefail

OUT_DIR="${1:-/mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24}"
REPO="/home/lilith/work/zen/zensimx"
VERDICT="${REPO}/target/release/bake_verdict"
QSWEEP="${REPO}/target/release/qsweep_eval"
CROSS_CODEC="${REPO}/scripts/v_next/measure_tuner_v10_cross_codec.py"

V10_BAKE="${REPO}/zensim/weights/v_tuner_v10_2026-05-20.bin"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
SUMMARY="${OUT_DIR}/tuner_v11_vs_v10_summary_$(date -u +%Y-%m-%d).md"

[ -x "${VERDICT}" ] || { echo "bake_verdict not built — run: cargo build --release --bin bake_verdict -p zensim-validate" >&2; exit 1; }
[ -f "${V10_BAKE}" ] || { echo "v10 bake missing: ${V10_BAKE}" >&2; exit 1; }

mkdir -p "${OUT_DIR}/verdicts"
TMP=$(mktemp -d)
trap "rm -rf ${TMP}" EXIT

# Run bake_verdict on v10 + each v11 seed.
extract_srocc() {
    # Args: verdict_md_path corpus
    # Output: SROCC of the aggregate row for the corpus
    awk -v corpus="$2" '
        $0 ~ "## " corpus { in_section = 1; next }
        in_section && /^\| V_X bake \|/ {
            # Pipe-split, field 2 = SROCC
            n = split($0, fields, "|")
            gsub(/ /, "", fields[3])
            print fields[3]
            exit
        }
    ' "$1"
}

extract_summary_row() {
    # Args: verdict_md_path corpus
    # Output: pipe-delimited "SROCC|PLCC|KROCC|PWRC|Z-RMSE" for corpus aggregate
    awk -v corpus="$2" '
        # Match the "## Summary (one row per corpus)" table.
        /^## Summary/ { in_summary = 1; next }
        in_summary && $0 ~ "\\| " corpus " \\|" {
            n = split($0, fields, "|")
            # Format: | Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |
            gsub(/ /, "", fields[4]); gsub(/ /, "", fields[5]); gsub(/ /, "", fields[6])
            gsub(/ /, "", fields[8]); gsub(/ /, "", fields[9])
            print fields[4] "|" fields[5] "|" fields[6] "|" fields[8] "|" fields[9]
            exit
        }
    ' "$1"
}

# Run verdict on each bake.
echo "[1/3] Running bake_verdict on v10 + 5 v11 seeds …"
"${VERDICT}" --bake "${V10_BAKE}" --output "${TMP}/v10.md" > /dev/null 2>&1
echo "      v10 → ${TMP}/v10.md"
for s in 1 2 3 4 5; do
    BAKE="${OUT_DIR}/tuner_v11_s${s}.bin"
    if [ ! -f "${BAKE}" ]; then
        echo "      WARN: seed ${s} bake missing: ${BAKE}"
        continue
    fi
    "${VERDICT}" --bake "${BAKE}" --output "${TMP}/v11_s${s}.md" > /dev/null 2>&1
    echo "      v11 s${s} → ${TMP}/v11_s${s}.md"
    cp "${TMP}/v11_s${s}.md" "${OUT_DIR}/verdicts/v11_s${s}_verdict.md"
done
cp "${TMP}/v10.md" "${OUT_DIR}/verdicts/v10_verdict.md"

# Run qsweep_eval (monotonicity criterion #3) on each bake if fixtures available.
if [ -x "${QSWEEP}" ] && [ -f "${QSWEEP_FEATURES}" ] && [ -f "${QSWEEP_MANIFEST}" ]; then
    echo "[1b/3] Running qsweep_eval (monotonicity) …"
    QSWEEP_ARGS=( --features "${QSWEEP_FEATURES}" --manifest "${QSWEEP_MANIFEST}" \
                  --bake "tuner_v10=${V10_BAKE}" )
    for s in 1 2 3 4 5; do
        BAKE="${OUT_DIR}/tuner_v11_s${s}.bin"
        [ -f "${BAKE}" ] && QSWEEP_ARGS+=( --bake "v11_s${s}=${BAKE}" )
    done
    QSWEEP_ARGS+=( --out "${OUT_DIR}/qsweep_eval_v11_vs_v10.md" )
    "${QSWEEP}" "${QSWEEP_ARGS[@]}" 2>&1 | grep -E "monotonicity=" || true
    echo "      qsweep report: ${OUT_DIR}/qsweep_eval_v11_vs_v10.md"
fi

echo "[2/3] Building summary markdown …"
{
    echo "# Tuner v11 vs v10 — 5-seed CI summary"
    echo
    echo "**Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "**Bake dir:** \`${OUT_DIR}\`"
    echo
    echo "## SROCC per corpus (aggregate)"
    echo
    echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |"
    echo "|---|---:|---:|---:|---:|---:|---:|"
    for v in v10 v11_s1 v11_s2 v11_s3 v11_s4 v11_s5; do
        [ -f "${TMP}/${v}.md" ] || continue
        row="| ${v} |"
        for corpus in "CID22" "KADIK10k" "TID2013" "KonJND-1k (full)" "AIC-3 CTC" "AIC-4 sample"; do
            stats=$(extract_summary_row "${TMP}/${v}.md" "${corpus}")
            srocc=$(echo "${stats}" | cut -d'|' -f1)
            row="${row} ${srocc:-?} |"
        done
        echo "${row}"
    done
    echo
    echo "## Z-RMSE per corpus (lower is better)"
    echo
    echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |"
    echo "|---|---:|---:|---:|---:|---:|---:|"
    for v in v10 v11_s1 v11_s2 v11_s3 v11_s4 v11_s5; do
        [ -f "${TMP}/${v}.md" ] || continue
        row="| ${v} |"
        for corpus in "CID22" "KADIK10k" "TID2013" "KonJND-1k (full)" "AIC-3 CTC" "AIC-4 sample"; do
            stats=$(extract_summary_row "${TMP}/${v}.md" "${corpus}")
            zrmse=$(echo "${stats}" | cut -d'|' -f5)
            row="${row} ${zrmse:-?} |"
        done
        echo "${row}"
    done
    echo
    echo "## Ship gate scorecard (per benchmarks/v_tuner_v11_methodology_2026-05-24.md)"
    echo
    echo "| Criterion | V_tuner_v10 baseline | v11 median target | Status |"
    echo "|---|--:|--:|---|"
    echo "| 1. KonJND val SROCC ≥ 0.85 | 0.2317 | ≥ 0.85 | TBD per median seed |"
    echo "| 2. CID22 SROCC ≥ 0.864 | 0.8540 | ≥ 0.864 | TBD |"
    echo "| 3. Monotonicity ≥ 92.78% | 92.78% | ≥ 92.78% | requires JPEG sweep eval |"
    echo "| 4. Cross-codec p50 \|Δ\| ≤ 1.0 in 60-90 | 0.6-1.5 | ≤ 1.0 | run measure_tuner_v10_cross_codec.py on median seed |"
    echo "| 5. Score 0-55 dial recovers | FAIL (clamped at 55) | per-anchor stddev non-flat | run measure_tuner_v10_cross_codec.py |"
    echo
    echo "Pass requires ≥4/5 criteria. Median seed by CID22 SROCC is the candidate."
    echo
    echo "## Per-seed verdicts (full panel)"
    echo
    for s in 1 2 3 4 5; do
        echo "- [v11 seed ${s}](verdicts/v11_s${s}_verdict.md)"
    done
    echo "- [v10 baseline](verdicts/v10_verdict.md)"
    echo
} > "${SUMMARY}"

echo "[3/3] Summary written: ${SUMMARY}"
echo
echo "Next step: pick median-CID22 seed and run cross-codec consistency:"
echo "  python3 ${CROSS_CODEC} --bake <median_seed_bake.bin> \\"
echo "    --out-md ${OUT_DIR}/tuner_v11_median_cross_codec.md"
