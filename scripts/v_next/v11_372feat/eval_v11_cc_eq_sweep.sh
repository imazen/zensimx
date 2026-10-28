#!/usr/bin/env bash
# V11-A-CC-EQ-WEIGHT-SWEEP (task #197) eval: run bake_verdict on every
# (weight, seed) bake. Emit summary TSV per weight tier with CID22 /
# KADID / TID / KonJND / AIC-3 / AIC-4 SROCC per seed.
set -euo pipefail

BAKE_VERDICT="/home/lilith/work/zen/zensimx/target/release/bake_verdict"
SWEEP_DIR="${SWEEP_DIR:-/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20}"
VERDICTS_DIR="${SWEEP_DIR}/verdicts"
mkdir -p "${VERDICTS_DIR}"

# Map W_TAG → cross_codec_eq_weight string for the summary.
declare -A W_FROM_TAG=([05]=0.05 [10]=0.10 [20]=0.20 [50]=0.50)

SUMMARY_TSV="${SWEEP_DIR}/summary_2026-05-20.tsv"
printf "w_tag\tcc_eq_weight\tseed\tcid22\tkadid\ttid\tkonjnd\taic3\taic4\n" > "${SUMMARY_TSV}"

for w_tag in 05 10 20 50; do
    cc_eq_w="${W_FROM_TAG[$w_tag]}"
    for s in 1 2 3 4 5; do
        BAKE="${SWEEP_DIR}/cc4v11_w${w_tag}_s${s}.bin"
        OUT="${VERDICTS_DIR}/cc4v11_w${w_tag}_s${s}.md"
        if [ ! -f "${BAKE}" ]; then
            echo "SKIP w=${cc_eq_w} s=${s}: bake missing"
            continue
        fi
        if [ ! -f "${OUT}" ]; then
            echo "=== bake_verdict w=${cc_eq_w} s=${s} ==="
            "${BAKE_VERDICT}" --bake "${BAKE}" \
                --corpora cid22,kadid,tid,konjnd,aic3,aic4 \
                --output "${OUT}" 2>&1 | tail -3
        fi
        # Extract per-corpus SROCC from md
        cid=$(grep -E '^\| CID22 \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        kad=$(grep -E '^\| KADIK10k \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        tid=$(grep -E '^\| TID2013 \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        kon=$(grep -E '^\| KonJND-1k \(full\) \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        aic3=$(grep -E '^\| AIC-3 CTC \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        aic4=$(grep -E '^\| AIC-4 sample \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${w_tag}" "${cc_eq_w}" "${s}" \
            "${cid:-NA}" "${kad:-NA}" "${tid:-NA}" \
            "${kon:-NA}" "${aic3:-NA}" "${aic4:-NA}" >> "${SUMMARY_TSV}"
    done
done

echo
echo "=== summary written to ${SUMMARY_TSV} ==="
cat "${SUMMARY_TSV}"

# Per-w-tier median
echo
echo "=== per-w-tier MEDIAN ==="
echo "w	cid22	kadid	tid	konjnd	aic3	aic4"
for w_tag in 05 10 20 50; do
    cc_eq_w="${W_FROM_TAG[$w_tag]}"
    awk -F'\t' -v w="${w_tag}" -v cc="${cc_eq_w}" '
        $1 == w {
            cid[++ic]=$4; kad[++ik]=$5; tid[++it]=$6;
            kon[++iko]=$7; a3[++ia3]=$8; a4[++ia4]=$9;
        }
        function med(arr, n,    i, sorted, mid) {
            for (i=1; i<=n; i++) sorted[i]=arr[i]+0
            asort(sorted)
            mid = int((n+1)/2)
            if (n % 2 == 1) return sorted[mid]
            return (sorted[mid] + sorted[mid+1])/2
        }
        END {
            printf "%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                cc, med(cid,ic), med(kad,ik), med(tid,it),
                med(kon,iko), med(a3,ia3), med(a4,ia4)
        }
    ' "${SUMMARY_TSV}"
done
