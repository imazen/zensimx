#!/usr/bin/env bash
# EXP-V11-D-PJND-DOMINANT (task #198) cross-codec consistency eval.
# Runs cross_codec_consistency.py at T=80 (JND) and T=50 (JOD) on the
# KonJND-median bake per pjnd_w tier (chosen by eval_sweep.sh's
# inline KonJND-median picker).
#
# Outputs per-bake TSVs with per-image (q_jpeg, q_webp, q_avif,
# pairwise butter_max + butter_p3) under
# /mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20/cc_consistency/.
set -euo pipefail

SWEEP_DIR="${SWEEP_DIR:-/mnt/v/output/zensim/exp_v11_d_pjnd_2026-05-20}"
CCC_DIR="${SWEEP_DIR}/cc_consistency"
mkdir -p "${CCC_DIR}"

TARGET_PREDICT_TOOL="/home/lilith/work/zen/zensimx/target/release/predict_features_with_bake"
TARGET_ZEN_METRICS="/home/lilith/work/zen/zenmetrics/target/release/zenmetrics"
CONSISTENCY=/home/lilith/work/zen/zensimx/scripts/v_next/cross_codec_consistency.py

# Tier → KonJND-median seed (populated by the caller via env / arg).
# Default fallback: median = s3 across all tiers.
declare -A TIER_TO_SEED
TIER_TO_SEED[2.0]="${TIER_2_0_SEED:-3}"
TIER_TO_SEED[5.0]="${TIER_5_0_SEED:-3}"
TIER_TO_SEED[10.0]="${TIER_10_0_SEED:-3}"

for pjnd_w in 2.0 5.0 10.0; do
    seed="${TIER_TO_SEED[$pjnd_w]}"
    BAKE="${SWEEP_DIR}/cc4v11d_pjnd${pjnd_w}_s${seed}.bin"
    if [ ! -f "${BAKE}" ]; then
        echo "SKIP pjnd_w=${pjnd_w} s=${seed}: bake missing at ${BAKE}"
        continue
    fi
    for tgt in 80 50; do
        OUT="${CCC_DIR}/cc4v11d_pjnd${pjnd_w}_s${seed}_t${tgt}_n20.tsv"
        if [ -f "${OUT}" ]; then
            echo "skip pjnd_w=${pjnd_w} s=${seed} t=${tgt} — exists"
            continue
        fi
        echo "=== cc-consistency pjnd_w=${pjnd_w} s=${seed} target=${tgt} ==="
        python3 "${CONSISTENCY}" \
            --target ${tgt} \
            --bake "${BAKE}" \
            --bake-post clamp \
            --n-images 20 \
            --predict-tool "${TARGET_PREDICT_TOOL}" \
            --zen-metrics "${TARGET_ZEN_METRICS}" \
            --out "${OUT}" 2>&1 | tail -8 || echo "  driver failed"
    done
done

echo
echo "=== aggregated cc means per (pjnd_w, target) ==="
echo "pjnd_w	seed	target	n	mean_bp3	mean_bmax"
for pjnd_w in 2.0 5.0 10.0; do
    seed="${TIER_TO_SEED[$pjnd_w]}"
    for tgt in 80 50; do
        OUT="${CCC_DIR}/cc4v11d_pjnd${pjnd_w}_s${seed}_t${tgt}_n20.tsv"
        if [ ! -f "${OUT}" ]; then
            echo "${pjnd_w}	${seed}	${tgt}	NA	NA	NA"
            continue
        fi
        awk -F'\t' 'NR>1 {bp3+=$NF; bmax+=$(NF-1); n++} END {if (n>0) printf "%.4f\t%.4f\n", bp3/n, bmax/n; else printf "NA\tNA\n"}' "${OUT}" \
            | awk -F'\t' -v pj="${pjnd_w}" -v sd="${seed}" -v tg="${tgt}" -v n="$(($(wc -l < "${OUT}") - 1))" \
                '{printf "%s\t%s\t%s\t%s\t%s\t%s\n", pj, sd, tg, n, $1, $2}'
    done
done
