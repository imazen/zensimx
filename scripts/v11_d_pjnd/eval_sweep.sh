#!/usr/bin/env bash
# EXP-V11-D-PJND-DOMINANT (task #198) sweep eval. Runs bake_verdict on
# every (pjnd_weight, seed) bake and emits a summary TSV + per-tier
# medians. Mirrors the V11-A-CC-EQ-WEIGHT-SWEEP eval shape so the
# ship-table decision matches the V11 falsification documentation.
set -euo pipefail

BAKE_VERDICT="/home/lilith/work/zen/zensimx/target/release/bake_verdict"
SWEEP_DIR="${SWEEP_DIR:-/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20}"
VERDICTS_DIR="${SWEEP_DIR}/verdicts"
mkdir -p "${VERDICTS_DIR}"

# PJND-passthrough weights to evaluate (matches launch_sweep.sh).
WEIGHTS=("2.0" "5.0" "10.0")
SEEDS=(1 2 3 4 5)

SUMMARY_TSV="${SWEEP_DIR}/summary_2026-05-20.tsv"
printf "pjnd_w\tseed\tcid22\tkadid\ttid\tkonjnd\taic3\taic4\n" > "${SUMMARY_TSV}"

for pjnd_w in "${WEIGHTS[@]}"; do
    for s in "${SEEDS[@]}"; do
        BAKE="${SWEEP_DIR}/cc4v11d_pjnd${pjnd_w}_s${s}.bin"
        OUT="${VERDICTS_DIR}/cc4v11d_pjnd${pjnd_w}_s${s}.md"
        if [ ! -f "${BAKE}" ]; then
            echo "SKIP pjnd_w=${pjnd_w} s=${s}: bake missing"
            continue
        fi
        if [ ! -f "${OUT}" ]; then
            echo "=== bake_verdict pjnd_w=${pjnd_w} s=${s} ==="
            "${BAKE_VERDICT}" --bake "${BAKE}" \
                --corpora cid22,kadid,tid,konjnd,aic3,aic4 \
                --output "${OUT}" 2>&1 | tail -3
        fi
        cid=$(grep -E '^\| CID22 \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        kad=$(grep -E '^\| KADIK10k \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        tid=$(grep -E '^\| TID2013 \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        kon=$(grep -E '^\| KonJND-1k \(full\) \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        aic3=$(grep -E '^\| AIC-3 CTC \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        aic4=$(grep -E '^\| AIC-4 sample \|' "${OUT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
        printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${pjnd_w}" "${s}" \
            "${cid:-NA}" "${kad:-NA}" "${tid:-NA}" \
            "${kon:-NA}" "${aic3:-NA}" "${aic4:-NA}" >> "${SUMMARY_TSV}"
    done
done

echo
echo "=== per-seed summary at ${SUMMARY_TSV} ==="
cat "${SUMMARY_TSV}"

# Per-tier median + KonJND-median bake selection
echo
echo "=== per-pjnd_w MEDIAN (over seeds) ==="
echo "pjnd_w	cid22	kadid	tid	konjnd	aic3	aic4"
for pjnd_w in "${WEIGHTS[@]}"; do
    awk -F'\t' -v w="${pjnd_w}" '
        $1 == w {
            cid[++ic]=$3; kad[++ik]=$4; tid[++it]=$5;
            kon[++iko]=$6; a3[++ia3]=$7; a4[++ia4]=$8;
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
                w, med(cid,ic), med(kad,ik), med(tid,it),
                med(kon,iko), med(a3,ia3), med(a4,ia4)
        }
    ' "${SUMMARY_TSV}"
done

# Highest-KonJND seed per tier (the spec says: pick median bake by
# KonJND SROCC for the panel emission). We do it inline here by
# emitting the bake basename per-tier with the median KonJND.
echo
echo "=== KonJND-median bake per pjnd_w (spec: 'pick median bake by KonJND SROCC') ==="
echo "pjnd_w	kon_median	seed	bake_path"
for pjnd_w in "${WEIGHTS[@]}"; do
    awk -F'\t' -v w="${pjnd_w}" -v dir="${SWEEP_DIR}" '
        $1 == w { vals[++n]=$6; seeds[n]=$2 }
        END {
            # Sort by val, find median row index.
            for (i=1; i<=n; i++) idx[i]=i
            # Sort idx by vals[idx[i]] ascending.
            for (i=1; i<n; i++) for (j=i+1; j<=n; j++)
                if (vals[idx[i]]+0 > vals[idx[j]]+0) { t=idx[i]; idx[i]=idx[j]; idx[j]=t }
            mid = int((n+1)/2)
            si = idx[mid]
            printf "%s\t%s\t%s\t%s/cc4v11d_pjnd%s_s%s.bin\n",
                w, vals[si], seeds[si], dir, w, seeds[si]
        }
    ' "${SUMMARY_TSV}"
done
