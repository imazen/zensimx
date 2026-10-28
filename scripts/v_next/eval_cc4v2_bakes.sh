#!/usr/bin/env bash
# EXP-CROSS-CODEC-V2 evaluation driver — runs bake_verdict (Mohammadi
# panel on canonical val parquets) AND cross-codec consistency at T=63
# for each candidate bake. Aggregates results into a markdown table.
#
# Usage: $0
# Produces:
#   benchmarks/v_cross_codec_v2_eval_2026-05-19.md  — verdict table
#   /mnt/v/output/zensim/cross_codec_metric_2026-05-19-v2/<label>/
#     eval_t63_n20.tsv  — per-image cross-codec results per bake

set -euo pipefail

ROOT="/home/lilith/work/zen/zensimx"
BAKES_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19"
EVAL_OUT_DIR="/mnt/v/output/zensim/cross_codec_metric_2026-05-19-v2"
VERDICT_DIR="${BAKES_DIR}/verdicts"
TOOL="${ROOT}/target/release/score_pair_with_bake"
ZEN_METRICS="/home/lilith/work/zen/zenmetrics/target/release/zenmetrics"
CC_SCRIPT="${ROOT}/scripts/v_next/cross_codec_consistency.py"
BAKE_VERDICT="${ROOT}/target/release/bake_verdict"

mkdir -p "${EVAL_OUT_DIR}" "${VERDICT_DIR}"

# Iterate over all bakes in this experiment.
shopt -s nullglob
bakes=( "${BAKES_DIR}"/cc4v2_*.bin )
if [ ${#bakes[@]} -eq 0 ]; then
    echo "ERROR: no cc4v2_*.bin under ${BAKES_DIR}" >&2
    exit 2
fi

declare -A T63_BMAX
declare -A T63_BP3
declare -A CID22 KADID TID KONJND AIC3

for bake in "${bakes[@]}"; do
    label="$(basename "${bake}" .bin)"
    echo "=== ${label} ==="
    # 1. bake_verdict for Mohammadi panel.
    if [ ! -f "${VERDICT_DIR}/${label}.md" ]; then
        "${BAKE_VERDICT}" --bake "${bake}" \
            --corpora cid22,kadid,tid,konjnd,aic3 \
            --output "${VERDICT_DIR}/${label}.md"
    fi
    # Pull aggregate SROCC per corpus from the .md.
    # Format: "| CID22 | 4292 | 0.8797 | ..." — SROCC is field 4 (after corpus, n).
    CID22[$label]=$(grep -E '^\| CID22 ' "${VERDICT_DIR}/${label}.md" | head -1 | awk -F'|' '{gsub(/ /,"",$4); print $4}' || echo "n/a")
    KADID[$label]=$(grep -E '^\| KADIK10k ' "${VERDICT_DIR}/${label}.md" | head -1 | awk -F'|' '{gsub(/ /,"",$4); print $4}' || echo "n/a")
    TID[$label]=$(grep -E '^\| TID2013 ' "${VERDICT_DIR}/${label}.md" | head -1 | awk -F'|' '{gsub(/ /,"",$4); print $4}' || echo "n/a")
    KONJND[$label]=$(grep -E '^\| KonJND' "${VERDICT_DIR}/${label}.md" | head -1 | awk -F'|' '{gsub(/ /,"",$4); print $4}' || echo "n/a")
    AIC3[$label]=$(grep -E '^\| AIC-3 ' "${VERDICT_DIR}/${label}.md" | head -1 | awk -F'|' '{gsub(/ /,"",$4); print $4}' || echo "n/a")

    # 2. Cross-codec consistency at T=63 (n=20).
    cc_out_dir="${EVAL_OUT_DIR}/${label}"
    mkdir -p "${cc_out_dir}"
    cc_tsv="${cc_out_dir}/eval_t63_n20.tsv"
    if [ ! -s "${cc_tsv}" ]; then
        python3 "${CC_SCRIPT}" \
            --bake "${bake}" \
            --bake-post clamp \
            --tool "${TOOL}" \
            --zen-metrics "${ZEN_METRICS}" \
            --target 63 \
            --n-images 20 \
            --out "${cc_tsv}" 2> "${cc_out_dir}/t63.log" || \
            { echo "  ERROR: cross-codec eval failed for ${label}"; }
    fi
    if [ -s "${cc_tsv}" ]; then
        T63_BMAX[$label]=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="pairwise_butter_max_mean") idx=i; next} {sum+=$idx; n+=1} END {if(n>0) printf "%.3f", sum/n; else print "NaN"}' "${cc_tsv}")
        T63_BP3[$label]=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="pairwise_butter_p3_mean") idx=i; next} {sum+=$idx; n+=1} END {if(n>0) printf "%.3f", sum/n; else print "NaN"}' "${cc_tsv}")
    else
        T63_BMAX[$label]="n/a"
        T63_BP3[$label]="n/a"
    fi
    echo "  CID22=${CID22[$label]}  AIC3=${AIC3[$label]}  KADID=${KADID[$label]}  TID=${TID[$label]}  KonJND=${KONJND[$label]}"
    echo "  T=63 butter_max=${T63_BMAX[$label]}  butter_p3=${T63_BP3[$label]}"
done

# Emit markdown table.
REPORT="${ROOT}/benchmarks/v_cross_codec_v2_eval_2026-05-19.md"
{
    echo "# EXP-CROSS-CODEC-V2 — eval table"
    echo ""
    echo "**Date:** 2026-05-19"
    echo "**Substrate:** tighter equivalence parquet (gap ≤ 0.3, 30 levels, avif 2× row weight) at \`/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet\` (68,788 pairs)."
    echo "**Recipe:** Tuner-v2 + cross-codec-eq parquet. Per-sample-α head, 372→128→128 identity."
    echo ""
    echo "## Mohammadi panel (SROCC aggregate, full panel in per-bake verdict.md)"
    echo ""
    echo "| Bake | CID22 | AIC-3 | KADID | TID | KonJND | T=63 butter_max | T=63 butter_p3 |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|"
    echo "| **Ship (cc4_s1_w1.0)** | 0.880 | 0.806 | 0.800 | 0.822 | 0.327 | **5.52** | 2.16 |"
    echo "| Tuner baseline | 0.879 | 0.813 | 0.770 | 0.748 | 0.235 | 8.07 | 3.03 |"
    for bake in "${bakes[@]}"; do
        label="$(basename "${bake}" .bin)"
        echo "| ${label} | ${CID22[$label]:-n/a} | ${AIC3[$label]:-n/a} | ${KADID[$label]:-n/a} | ${TID[$label]:-n/a} | ${KONJND[$label]:-n/a} | ${T63_BMAX[$label]:-n/a} | ${T63_BP3[$label]:-n/a} |"
    done
    echo ""
    echo "## Gate evaluation"
    echo ""
    echo "Strict gate: T=63 butter_max < 2.5"
    echo "Relaxed gate: T=63 butter_max < 3.0"
    echo ""
    echo "Secondary gates:"
    echo "- CID22 SROCC ≥ 0.86 (within 0.02 of W=1.0 ship)"
    echo "- KADID SROCC ≥ 0.70 (within 0.10 of W=1.0 ship)"
    echo "- TID SROCC ≥ 0.72 (within 0.10 of W=1.0 ship)"
    echo "- AIC-3 SROCC ≥ 0.78 (within 0.02 of W=1.0 ship)"
} > "${REPORT}"

echo
echo "DONE — report at ${REPORT}"
