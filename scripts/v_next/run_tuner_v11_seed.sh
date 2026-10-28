#!/usr/bin/env bash
# Tuner v11 trainer driver — task #6 (2026-05-24).
#
# Recipe: V_tuner_v10 (PreviewV0_5TunerV4) hyperparams + ALL FOUR
# improvements on the codec-target dial:
#
#   1. CID22 train-only-subset (17,611 pairs, ssim2-anchored) added
#      as 2nd training group (canonical-2026-05-21, task #5).
#   2. KONJND-AGGREGATION-HEAD enabled (task #4, 2026-05-24) — pools
#      predictions per ref then regresses against pjnd_target. This
#      unblocks raising konjnd-dense's training-weight allocation
#      from 0.02 to the score-floor-fix budget without KonJND
#      collapse.
#   3. CVVDP + IW-SSIM backfilled on cid22_train (task #7,
#      2026-05-24), enabling mix_cv40_iw60 target on the new group.
#   4. canonical-2026-05-21 substrate switches the trainer over from
#      the older canonical-2026-05-18 path.
#
# Ship gate (per docs/KONJND_AGGREGATION_HEAD_DESIGN_2026-05-24.md):
#   - KonJND val SROCC ≥ 0.85 (v10 floor 0.2317)
#   - CID22 SROCC ≥ 0.864 (Compression-trail parity)
#   - Monotonicity ≥ 92.78% (v10 floor)
#   - Cross-codec p50 |Δ| ≤ 1.0 in score 60-90 (v10 floor)
#   - Score 0-55 dial recovers (v10 floor pathology disappears)
#
# Args: <seed>
# Env: KBATCH (default 32), LR_OVERRIDE (default 5.66e-3)

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
LR="${LR_OVERRIDE:-5.66e-3}"
# 2026-05-24 falsification: w=0.3 collapses rank corpora (CID22
# 0.85→0.51, KADID 0.48→0.33, TID 0.66→0.35) while lifting
# KonJND +0.53 (0.23→0.76). The aggregation gradient overwhelms
# the per-pair rank signal. Default lowered to 0.05 + step_p 0.10
# (~5.5% of original effective rate) per task #6 iteration plan.
# Saved evidence: /mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24/
# tuner_v11_w0.3_s1_evidence.bin + qsweep + verdict.
KONJND_AGG_WEIGHT="${KONJND_AGG_WEIGHT:-0.05}"
KONJND_AGG_STEP_P="${KONJND_AGG_STEP_P:-0.10}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_tuner_v11_2026-05-24"
REPO="/home/lilith/work/zen/zensimx"
TRAINER="${REPO}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"

# Use the V10 anchor parquet (8 bands, [0, 100] range, JND=60, JOD=30).
# canonical-2026-05-21 doesn't ship a new anchor; reuse V10's.
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/tuner_v11_s${SEED}.bin"
LOG="${OUT_DIR}/tuner_v11_s${SEED}.log"
STDOUT="${OUT_DIR}/tuner_v11_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

# Sanity-check the trainer is built.
if [[ ! -x "${TRAINER}" ]]; then
    echo "ERROR: trainer binary missing — build with:" >&2
    echo "  cargo build --release --bin zensim_mlp_train -p zensim-validate" >&2
    exit 2
fi

echo "Tuner v11: seed=${SEED} KBATCH=${KBATCH} LR=${LR} konjnd_agg_w=${KONJND_AGG_WEIGHT} step_p=${KONJND_AGG_STEP_P}"
echo "  trainer:           ${TRAINER}"
echo "  anchor (V10):      ${ANCHOR}"
echo "  cross-codec-eq:    ${EQUIV}"
echo "  konjnd-agg pool:   ${PARQ_DIR}/konjnd-dense.parquet"
echo "  cid22-train pool:  ${PARQ_DIR}/cid22_train.parquet"
echo "  out_bake:          ${BAKE}"
echo

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "cid22_train:${PARQ_DIR}/cid22_train.parquet:0.5:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr "${LR}" --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size "${KBATCH}" \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 0.5 \
    --anchor-target-score 60.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 \
    --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --konjnd-aggregation-parquet "${PARQ_DIR}/konjnd-dense.parquet" \
    --konjnd-aggregation-weight "${KONJND_AGG_WEIGHT}" \
    --konjnd-aggregation-step-p "${KONJND_AGG_STEP_P}" \
    --konjnd-aggregation-samples-per-ref 5 \
    --konjnd-aggregation-refs-per-step 8 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
