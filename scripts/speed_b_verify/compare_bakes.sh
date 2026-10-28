#!/usr/bin/env bash
# Run bake_verdict on K=1 and K=32 bakes for SROCC comparison.
set -euo pipefail

VERIFY_DIR="/mnt/v/zen/zensim-eval/speed_b_verify_2026-05-19"
BV="/home/lilith/work/zen/zensimx/target/release/bake_verdict"

if [ ! -x "${BV}" ]; then
    echo "bake_verdict not found at ${BV}" >&2
    exit 1
fi

for K in 1 32; do
    BAKE="${VERIFY_DIR}/speedb_k${K}_s1.bin"
    OUT="${VERIFY_DIR}/speedb_k${K}_s1_verdict.md"
    if [ ! -f "${BAKE}" ]; then
        echo "MISSING ${BAKE} -- skipping K=${K}"
        continue
    fi
    echo "=== K=${K} verdict ==="
    "${BV}" --bake "${BAKE}" --corpora cid22,kadid,tid,konjnd,aic3 --output "${OUT}" 2>&1 | tail -25
done

echo "=== Side-by-side aggregate ==="
for K in 1 32; do
    OUT="${VERIFY_DIR}/speedb_k${K}_s1_verdict.md"
    if [ -f "${OUT}" ]; then
        echo "--- K=${K} ---"
        grep -E "^\| (CID22|KADID|TID|KonJND|AIC-3) " "${OUT}" | head -5 || head -25 "${OUT}"
    fi
done
