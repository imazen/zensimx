#!/usr/bin/env bash
# G-ADDR board re-grade — run the gate over every FAIR board cell under BOTH
# negative-tail pin sets, then graft the result onto the board fullevals.
#
#   scripts/gaddr_board_regrade.sh grade   # re-run all cells (both pin sets)
#   scripts/gaddr_board_regrade.sh graft   # graft the `product` reads onto the board
#
# WHY IT EXISTS. The 2026-09-04 board grading was done by an ad-hoc loop that
# was never committed, so re-running it after the 2026-09-05 tail re-pin meant
# reconstructing 97 invocations by hand. This script reads them back out of that
# run's own as-run LOGS — which record the bake, the ensemble members, the
# features root and the dial grid verbatim — so the re-grade is provably the
# SAME invocation with one flag changed, not a fresh guess at what was run.
#
# Binaries come from the MAIN repo (never a sibling worktree — those get cleaned
# up and take committed scripts down with them); override with ZL_BV.
set -euo pipefail
REPO=/home/lilith/work/zen/zensimx
BV="${ZL_BV:-$REPO/target/release/bake_verdict}"
SRC="${ZL_SRC:-/mnt/v/output/zensim/gaddr-board-2026-09-04}"   # the 2026-09-04 as-run
OUT="${ZL_OUT:-/mnt/v/output/zensim/gaddr-board-2026-09-05}"   # this re-grade
BOARD="${ZL_BOARD:-/mnt/v/output/zensim/reports/fulleval}"
GRIDTRUTH="${ZL_GRIDTRUTH:-/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv}"
# OWNER-EXTENSION, opt-in (2026-09-06): which A7r window to grade the board
# under. Default `distinct` is byte-identical to every prior board grade.
FLOORRULE="${ZL_FLOORRULE:-distinct}"
FLOORMARGIN_ARGS=()
[ -n "${ZL_FLOORMARGIN:-}" ] && FLOORMARGIN_ARGS=(--floor-margin "$ZL_FLOORMARGIN")

case "${1:-}" in
  grade) python3 "$REPO/scripts/gaddr_board_regrade.py" grade \
            --bv "$BV" --src "$SRC" --out "$OUT" --grid-truth "$GRIDTRUTH" \
            --floor-rule "$FLOORRULE" "${FLOORMARGIN_ARGS[@]}" ;;
  graft) python3 "$REPO/scripts/gaddr_board_regrade.py" graft \
            --src "$SRC" --out "$OUT" --board "$BOARD" ;;
  *) echo "usage: $0 {grade|graft}" >&2; exit 2 ;;
esac
