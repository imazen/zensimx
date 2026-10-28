#!/usr/bin/env bash
# zensim-b-phone: train a zensim metric that emulates CVVDP at
# iPhone-14-Pro display conditions (67 PPD, 1025 nits peak).
#
# Target = iPhone-14 CVVDP JOD mapped to a 0..100 dial via the V12 band
# table (build_iphone14_cvvdp_dial_parquets.py writes human_score).
#
# Recipe = V39 working-dial recipe (per zensim CLAUDE.md "V39 ship +
# dial/spline/anchor learnings"): rank-honest hybrid MSE+RankNet w/
# per-sample-alpha head + tanh output head, multi-band anchor at low
# weight for the post-training spline shape.
#
# Args: <seed> (default 17)
set -euo pipefail

SEED="${1:-17}"
REPO="/home/lilith/work/zen/zensimx"
TRAINER="${REPO}/target/release/zensim_mlp_train"
DIAL_DIR="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25"
BAKE_DIR="/mnt/v/output/zensim/bakes"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v12-cvvdp-substrate/anchors_cvvdp_372col_continuous.parquet"
AUTOTRANS="${REPO}/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"

BAKE="${BAKE_DIR}/zensim_b_phone_seed${SEED}_2026-05-25.bin"
LOG="${BAKE_DIR}/zensim_b_phone_seed${SEED}_2026-05-25.train.log"

mkdir -p "${BAKE_DIR}"
[ -x "${TRAINER}" ] || { echo "trainer missing: ${TRAINER}" >&2; exit 2; }
[ -f "${ANCHOR}" ] || { echo "anchor missing: ${ANCHOR}" >&2; exit 2; }
[ -f "${AUTOTRANS}" ] || { echo "auto-transforms screen missing: ${AUTOTRANS}" >&2; exit 2; }

echo "zensim-b-phone train: seed=${SEED}"
echo "  target = iPhone-14 CVVDP dial (human_score in *_iphone14_cvvdptgt.parquet)"
echo "  anchor = ${ANCHOR} (DESKTOP-CVVDP-derived target_score — see caveat)"
echo

"${TRAINER}" \
    --group "kadid_ip14:${DIAL_DIR}/kadid_iphone14_cvvdptgt.parquet:1.0:0.0" \
    --group "tid_ip14:${DIAL_DIR}/tid_iphone14_cvvdptgt.parquet:1.0:0.0" \
    --hidden 128 --n-hidden-layers 2 \
    --per-sample-alpha-head \
    --epochs 200 --lr 0.001 --l2 0.0001 --seed "${SEED}" \
    --max-features 372 \
    --mse-weight 0.6 --ranknet-weight 0.6 \
    --monotonicity-reg 1.0 \
    --tanh-output-head-scale 30.0 \
    --minibatch-size 32 \
    --auto-transforms "${AUTOTRANS}" \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 0.01 --anchor-step-p 0.05 \
    --target-column human_score --target-scale 100.0 \
    --out-dtype f32 \
    --out "${BAKE}" --log-path "${LOG}"

echo
echo "Bake written: ${BAKE}"
echo "Verdict: ${BAKE%.bin}.verdict.md"
