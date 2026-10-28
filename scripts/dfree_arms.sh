#!/usr/bin/env bash
# D + free — free-set variants of the D-id100-negrich recipe (2026-09-05).
#
#   base  <label> <slice-file>              lasso fit on the pools-944 leg (the FF probe recipe, lam=0.3)
#   chain <label> <slice-file> <anchor.pq>  same fit with the DIAL anchor, then extend-top -> graded bake
#   grade <label> <bake.bin>                G-ADDR + rank at the folded720append2pools era
#
# Every fit flag except `--slice-file` and `--anchor-*` is byte-identical to the
# recipe embedded in `FF_treatment_156plusfree_l0.3.bin`'s `zentrain.repro`, so a
# difference between arms is attributable to the coordinate slice / anchor alone.
#
# Binaries come from the MAIN repo by default (never a sibling worktree — those
# get cleaned up and take committed scripts down with them); override ZL_BDR/ZL_BV.
set -euo pipefail
REPO=/home/lilith/work/zen/zensimx
R="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
V="${ZL_BV:-$REPO/target/release/bake_verdict}"
W=/mnt/v/output/zensim/dfree-2026-09-05
ROOT=/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30
GRAM=/mnt/v/output/zensim/freefeats-2026-09-01/quality-probe/g_safe_human944.npz
SAFESYN=$ROOT/ext_safesyn_full.parquet
GRID=/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet
IDPROBE=$W/probes/identity_probe_944pools_2026-09-05.parquet
NTPROBE=$W/probes/negtail_probe_944pools_2026-09-05.parquet
TOPANCHOR=$W/anchors/anchor944_r1b_dial_clamped.parquet
LAM=0.3
mkdir -p "$W"/{bakes,fits,gaddr,verdicts,work}

fit_common=(--gram "$GRAM" --space raw --target human_score --lam "$LAM")

case "${1:-}" in
base)
    LABEL="$2"; SLICE="$3"
    "$R" fit-lasso "${fit_common[@]}" --slice-file "$SLICE" \
        --anchor-parquet "$SAFESYN" --anchor-target human_score \
        --anchor-scale 100 --anchor-clip-min -100 --anchor-stride 37 \
        --embed-repro --out "$W/bakes/${LABEL}.bin"
    sha256sum "$W/bakes/${LABEL}.bin"
    ;;
chain)
    LABEL="$2"; SLICE="$3"; ANCHOR="$4"
    "$R" fit-lasso "${fit_common[@]}" --slice-file "$SLICE" \
        --anchor-parquet "$ANCHOR" --anchor-target target_score \
        --embed-repro --out "$W/bakes/${LABEL}_raw.bin"
    "$R" extend-top --in "$W/bakes/${LABEL}_raw.bin" \
        --out "$W/bakes/${LABEL}.bin" --anchor "$TOPANCHOR" --target-col target_score
    sha256sum "$W/bakes/${LABEL}.bin"
    ;;
grade)
    LABEL="$2"; BAKE="$3"
    "$V" --bake "$BAKE" --regime 944 --cross-regime --features-root "$ROOT" \
         --dial-grid "$GRID" --negtail-probe "$NTPROBE" --identity-probe "$IDPROBE" \
         --corpora cid22,konjnd,kadid,tid,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy \
         --gaddr-json "$W/gaddr/gaddr_${LABEL}.json" \
         --full-json "$W/verdicts/verdict_${LABEL}.json" \
         > "$W/verdicts/verdict_${LABEL}.log" 2>&1
    echo "== $LABEL =="
    python3 "$(dirname "$0")/dfree_report.py" "$W/verdicts/verdict_${LABEL}.json"
    ;;
*) echo "usage: $0 base <label> <slice> | chain <label> <slice> <anchor.pq> | grade <label> <bake.bin>" >&2; exit 2;;
esac
