#!/usr/bin/env bash
# PreviewV0_5TunerV2 (EXP-TUNER-V2, 2026-05-19) — full evaluation across
# all 3 seeds:
#   1. Cross-codec JND eval for each seed.
#   2. bake_verdict standard A.9 panel (CID22/KADID/TID/KonJND/AIC-3).
#   3. Compare against PreviewV0_5Tuner (today's ship).
#
# Outputs:
#   - /mnt/v/output/zensim/cross_codec_consistency_2026-05-19/tuner_v2_s{1,2,3}/
#   - /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/verdict_s{1,2,3}.md
#   - /tmp/exp_tuner_v2_eval_*.log
set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19"
ZENSIM_ROOT="/home/lilith/work/zen/zensimx"
SCORE_BAKE="${ZENSIM_ROOT}/target/release/score_pair_with_bake"
BAKE_VERDICT="${ZENSIM_ROOT}/target/release/bake_verdict"
FEAT_ROOT="/mnt/v/zen/zensim-training/canonical-2026-05-18/val"

for SEED in 1 2 3; do
    BAKE="${OUT_DIR}/tuner_v2_s${SEED}_h128.bin"
    if [ ! -f "${BAKE}" ]; then
        echo "ERROR: bake not found: ${BAKE}" >&2
        exit 1
    fi
    LABEL="tuner_v2_s${SEED}"
    CCC_LOG="/tmp/exp_tuner_v2_eval_ccc_s${SEED}.log"
    VERDICT_LOG="/tmp/exp_tuner_v2_eval_verdict_s${SEED}.log"
    VERDICT_MD="${OUT_DIR}/verdict_s${SEED}.md"

    echo "## seed=${SEED} cross-codec consistency"
    # Map bake-post: pure-MSE bakes are score-shaped → use clamp (default)
    python3 "${ZENSIM_ROOT}/scripts/v_next/cross_codec_jnd_eval_bake.py" \
        --bake "${BAKE}" --label "${LABEL}" --bake-post clamp \
        2>&1 | tee "${CCC_LOG}"

    echo "## seed=${SEED} bake_verdict"
    "${BAKE_VERDICT}" --bake "${BAKE}" \
        --corpora cid22,kadid,tid,konjnd,aic3 \
        --features-root "/mnt/v/zen/zensim-training/2026-05-15-full-features" \
        --output "${VERDICT_MD}" \
        2>&1 | tee "${VERDICT_LOG}"
done

echo "## qsweep eval (all 3 seeds + baseline Tuner)"
QSWEEP_LOG="/tmp/exp_tuner_v2_eval_qsweep.log"
"${ZENSIM_ROOT}/target/release/qsweep_eval" \
    --features /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv \
    --manifest /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv \
    --bake baseline_tuner=${ZENSIM_ROOT}/zensim/weights/v_tuner_2026-05-18.bin:clamp \
    --bake tuner_v2_s1=${OUT_DIR}/tuner_v2_s1_h128.bin:clamp \
    --bake tuner_v2_s2=${OUT_DIR}/tuner_v2_s2_h128.bin:clamp \
    --bake tuner_v2_s3=${OUT_DIR}/tuner_v2_s3_h128.bin:clamp \
    --out ${OUT_DIR}/qsweep_v2_vs_baseline.md \
    2>&1 | tee "${QSWEEP_LOG}"

echo "## aggregate 3 seeds"
python3 "${ZENSIM_ROOT}/scripts/v_next/aggregate_tuner_v2.py" \
    2>&1 | tee "${OUT_DIR}/aggregate_3seed.md"

echo "DONE all 3 seeds evaluated"
