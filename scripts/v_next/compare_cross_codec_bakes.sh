#!/usr/bin/env bash
# Compare N bakes' cross-codec consistency at multiple targets.
# Args: <out_dir> <label1>:<bake1> [<label2>:<bake2> ...]
#
# For each (label, bake) pair, runs cross_codec_consistency.py at T=63
# on n=20 images. Aggregates results into a single comparison table.

set -euo pipefail

OUT_DIR="${1:?usage: $0 <out_dir> <label:bake> [<label:bake> ...]}"
shift

ROOT="/home/lilith/work/zen/zensimx"
SCRIPT="${ROOT}/scripts/v_next/cross_codec_consistency.py"
TOOL="/home/lilith/work/zen/zensimx/target/release/score_pair_with_bake"
ZEN_METRICS="/home/lilith/work/zen/zenmetrics/target/release/zenmetrics"
N_IMAGES="${N_IMAGES:-20}"
TARGETS="${TARGETS:-30,50,63,70,80,90}"

mkdir -p "${OUT_DIR}"

echo "compare_cross_codec_bakes"
echo "  out_dir: ${OUT_DIR}"
echo "  n_images: ${N_IMAGES}"
echo "  targets: ${TARGETS}"

declare -A RESULTS_BMAX
declare -A RESULTS_BP3

for pair in "$@"; do
    LABEL="${pair%%:*}"
    BAKE="${pair#*:}"
    if [ ! -f "${BAKE}" ]; then
        echo "ERROR: missing ${BAKE}" >&2
        continue
    fi
    echo "=== ${LABEL} ==="
    for T in $(echo "${TARGETS}" | tr ',' ' '); do
        OUT_TSV="${OUT_DIR}/${LABEL}_t${T}.tsv"
        LOG="${OUT_DIR}/${LABEL}_t${T}.log"
        if [ ! -f "${OUT_TSV}" ]; then
            python3 "${SCRIPT}" \
                --bake "${BAKE}" \
                --bake-post clamp \
                --tool "${TOOL}" \
                --zen-metrics "${ZEN_METRICS}" \
                --target "${T}" \
                --n-images "${N_IMAGES}" \
                --out "${OUT_TSV}" 2> "${LOG}"
        fi
        BMAX=$(awk -F'\t' '
            NR==1 {
                for (i=1; i<=NF; i++) if ($i == "pairwise_butter_max_mean") idx=i
                next
            }
            $idx == $idx { sum += $idx; n += 1 }
            END { if (n > 0) printf "%.4f", sum/n; else print "NaN" }
        ' "${OUT_TSV}")
        BP3=$(awk -F'\t' '
            NR==1 {
                for (i=1; i<=NF; i++) if ($i == "pairwise_butter_p3_mean") idx=i
                next
            }
            $idx == $idx { sum += $idx; n += 1 }
            END { if (n > 0) printf "%.4f", sum/n; else print "NaN" }
        ' "${OUT_TSV}")
        echo "  T=${T}  butter_max=${BMAX}  butter_p3=${BP3}"
        RESULTS_BMAX["${LABEL}_${T}"]="${BMAX}"
        RESULTS_BP3["${LABEL}_${T}"]="${BP3}"
    done
done

# Emit comparison table
TABLE="${OUT_DIR}/comparison.md"
{
    echo "# Cross-codec consistency comparison"
    echo
    echo "Mean pairwise butter_max (lower = more consistent across codecs)"
    echo
    printf "| Target |"
    for pair in "$@"; do
        LABEL="${pair%%:*}"
        printf " %s |" "${LABEL}"
    done
    printf "\n|---|"
    for pair in "$@"; do
        printf "---:|"
    done
    printf "\n"
    for T in $(echo "${TARGETS}" | tr ',' ' '); do
        printf "| T=%s |" "${T}"
        for pair in "$@"; do
            LABEL="${pair%%:*}"
            printf " %s |" "${RESULTS_BMAX["${LABEL}_${T}"]:-NaN}"
        done
        printf "\n"
    done
    echo
    echo "Mean pairwise butter_pnorm3 (parallel table)"
    echo
    printf "| Target |"
    for pair in "$@"; do
        LABEL="${pair%%:*}"
        printf " %s |" "${LABEL}"
    done
    printf "\n|---|"
    for pair in "$@"; do
        printf "---:|"
    done
    printf "\n"
    for T in $(echo "${TARGETS}" | tr ',' ' '); do
        printf "| T=%s |" "${T}"
        for pair in "$@"; do
            LABEL="${pair%%:*}"
            printf " %s |" "${RESULTS_BP3["${LABEL}_${T}"]:-NaN}"
        done
        printf "\n"
    done
} > "${TABLE}"

cat "${TABLE}"
