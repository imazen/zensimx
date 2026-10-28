#!/usr/bin/env bash
# zensim-b-phone (OLED indoor): train a zensim metric emulating CVVDP at
# the modern_oled_phone_indoor display (109.97 ppd, 400 nit SDR setpoint,
# OLED black washed to ~1000:1 by 250 lux ambient).
#
# CORRECTION 2 vs the prior attempt: the spline anchor is now built FROM
# the phone-CVVDP training data itself (modern_oled_anchor.parquet,
# target_score stratified across the phone-CVVDP dial [0,100]), NOT the
# DESKTOP-CVVDP-derived anchors_cvvdp_372col_continuous.parquet that gave
# the broken dial (G1=0.00, p5=96 p95=138).
#
# Recipe = V39 working-dial recipe (per zensim CLAUDE.md "V39 ship +
# dial/spline/anchor learnings"): rank-honest hybrid MSE+RankNet with a
# per-sample-alpha head + tanh output head; multi-band phone anchor at
# anchor-loss-weight 0.01 (spline-fit only — higher fights RankNet).
#
# Target = phone-CVVDP JOD -> 0..100 V12 dial (human_score in the
# *_phone_cvvdptgt.parquet, /100 so --target-scale 100 lands in 0..100).
#
# A held-out slice is carved by ref_basename so the tracking eval (SROCC
# of bake output vs held-out phone-CVVDP) is honest.
#
# Args: <seed> (default 17)
set -euo pipefail

SEED="${1:-17}"
REPO="/home/lilith/work/zen/zensimx"
TRAINER="${REPO}/target/release/zensim_mlp_train"
DIAL_DIR="/mnt/v/output/zensim/iphone14-cvvdp-2026-05-25"
BAKE_DIR="/mnt/v/output/zensim/bakes"
ANCHOR="${DIAL_DIR}/modern_oled_anchor.parquet"
AUTOTRANS="${REPO}/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"

# Held-out-trimmed training parquets (built by the python slicer below).
KADID_TR="${DIAL_DIR}/kadid_phone_cvvdptgt_train.parquet"
TID_TR="${DIAL_DIR}/tid_phone_cvvdptgt_train.parquet"

BAKE="${BAKE_DIR}/zensim_b_phone_oled_notanh_seed${SEED}_2026-05-26.bin"
LOG="${BAKE_DIR}/zensim_b_phone_oled_notanh_seed${SEED}_2026-05-26.train.log"

mkdir -p "${BAKE_DIR}"
[ -x "${TRAINER}" ] || { echo "trainer missing: ${TRAINER}" >&2; exit 2; }
[ -f "${ANCHOR}" ] || { echo "phone anchor missing: ${ANCHOR}" >&2; exit 2; }
[ -f "${AUTOTRANS}" ] || { echo "auto-transforms screen missing: ${AUTOTRANS}" >&2; exit 2; }
[ -f "${KADID_TR}" ] || { echo "kadid train slice missing: ${KADID_TR}" >&2; exit 2; }
[ -f "${TID_TR}" ] || { echo "tid train slice missing: ${TID_TR}" >&2; exit 2; }

echo "zensim-b-phone OLED train: seed=${SEED}"
echo "  target = phone-CVVDP dial (human_score in *_phone_cvvdptgt_train.parquet)"
echo "  anchor = ${ANCHOR} (PHONE-CVVDP-derived stratified target_score — CORRECTION 2)"
echo

"${TRAINER}" \
    --group "kadid_ph:${KADID_TR}:1.0:0.0" \
    --group "tid_ph:${TID_TR}:1.0:0.0" \
    --hidden 128 --n-hidden-layers 2 \
    --per-sample-alpha-head \
    --epochs 200 --lr 0.001 --l2 0.0001 --seed "${SEED}" \
    --max-features 372 \
    --mse-weight 0.6 --ranknet-weight 0.6 \
    --monotonicity-reg 1.0 \
    --tanh-output-head-scale 0.0 \
    --minibatch-size 32 \
    --auto-transforms "${AUTOTRANS}" \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 0.01 --anchor-step-p 0.05 \
    --target-column human_score --target-scale 100.0 \
    --out-dtype f32 \
    --out "${BAKE}" --log-path "${LOG}"

echo
echo "Bake written: ${BAKE}"
