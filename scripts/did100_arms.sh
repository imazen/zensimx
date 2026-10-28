#!/usr/bin/env bash
# D-id100 — identity-pinned ADD156 (Profile D lineage) arm driver.
#
#   ctl                          rebuild the shipped-D chain and assert both shas
#   fit  <label> <W> [anchorpq]  fit-lasso(safesyn @1.0 + identity38 @W) -> extend-top -> grade
#   grade <label> <bake.bin>     grade an existing bake (delegates to dialgate_arms.sh)
#
# EVERY fit flag except the added identity gram is byte-identical to the control
# recipe recovered for `ADD156_safesyn_only_raw_lasso.bin` (campaign gate G-T0):
#   --space raw --target human_score --lam 2e-3 --tau 0 --n-sweeps 400 --tol 1e-10
#   --slice-file <0..155> --gram <safesyn.npz> --weight 1.0 --anchor <val/anchor.npz>
# so any measured difference is attributable to the identity group alone.
#
# Binaries come from the MAIN repo by default; override with ZL_BDR / ZL_BV.
set -euo pipefail
REPO=/home/lilith/work/zen/zensimx
R="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
W=/mnt/v/output/zensim/did100-2026-09-04
GRAM_SAFESYN=/mnt/v/output/zensim-multicodec-probe/linear-probe/grams/safesyn.npz
GRAM_ID=$W/grams/identity38.npz
ANCHOR_NPZ=/mnt/v/output/zensim-multicodec-probe/linear-probe/val/anchor.npz
ANCHOR_PQ=/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet
ID_ANCHOR_PQ=$W/work/identity_anchor_rows_372.parquet
SLICE=$W/slices/a156.idx
LAM=2e-3
mkdir -p "$W"/{bakes,fits,gaddr,verdicts,work}

# Shared fit flags — the ONLY thing an arm may vary is what follows.
fit_common=(--space raw --target human_score --lam "$LAM" --tau 0
            --n-sweeps 400 --tol 1e-10 --slice-file "$SLICE")

case "${1:-}" in
ctl)
    "$R" fit-lasso "${fit_common[@]}" \
        --gram "$GRAM_SAFESYN" --weight 1.0 --anchor "$ANCHOR_NPZ" \
        --emit-fit-npz "$W/fits/CTL_add156.fit.npz" \
        --out "$W/bakes/CTL_add156_rawlasso.bin" \
        --expect-sha256 51437a34f04887ce850b25eff4f72a6bcd12926873ce060a12878d558a7517db
    "$R" extend-top --in "$W/bakes/CTL_add156_rawlasso.bin" \
        --out "$W/bakes/CTL_D_dense_dial.bin" --anchor "$ANCHOR_PQ" --target-col target_score
    sha256sum "$W/bakes/CTL_D_dense_dial.bin"
    cmp "$W/bakes/CTL_D_dense_dial.bin" "$REPO/zensim/weights/d_sdr_add156_dense_dial_2026-08-31.bin" \
        && echo "CTL: BYTE-IDENTICAL to shipped Profile D"
    ;;
fit)
    LABEL="$2"; IDW="$3"; AMODE="${4:-npz}"
    A=(--anchor "$ANCHOR_NPZ")
    [ "$AMODE" = "pq" ]   && A=(--anchor-parquet "$ANCHOR_PQ" --anchor-target target_score)
    [ "$AMODE" = "pqid" ] && A=(--anchor-parquet "$ANCHOR_PQ" --anchor-parquet "$ID_ANCHOR_PQ" --anchor-target target_score)
    "$R" fit-lasso "${fit_common[@]}" \
        --gram "$GRAM_SAFESYN" --weight 1.0 \
        --gram "$GRAM_ID" --weight "$IDW" \
        "${A[@]}" \
        --emit-fit-npz "$W/fits/${LABEL}.fit.npz" \
        --out "$W/bakes/${LABEL}_rawlasso.bin"
    "$R" extend-top --in "$W/bakes/${LABEL}_rawlasso.bin" \
        --out "$W/bakes/${LABEL}.bin" --anchor "$ANCHOR_PQ" --target-col target_score
    ;;
*) echo "usage: $0 ctl | fit <label> <identity_weight> [npz|pq|pqid]" >&2; exit 2;;
esac
