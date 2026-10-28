#!/usr/bin/env bash
# G-ADDR candidate driver — TWO modes, one scoring path.
#
#   build:  dialgate_arms.sh <label> <anchor.parquet> [n_edges] [extend_top_anchor]
#             shared-anchor -> add-winsor -> extend-top, then grade.
#   score:  dialgate_arms.sh score <label> <bake.bin> [regime]
#             grade an EXISTING bake (a shipped profile, a prior arm) with no
#             rebuild. `regime` is 372 (default) / 720 / 944.
#
# Both modes end in the SAME `grade` function, so a re-grade after a bar change
# can never accidentally be a different measurement from the build-time one.
#
# Binaries come from the MAIN repo (never a sibling worktree — those get cleaned
# up and take committed scripts down with them); override with ZL_BV / ZL_BDR.
#
# ERA. `ZL_ERA` selects WHICH 372 instruments + features root the grade reads.
# The default `canonical` is the 2026-09-04 set and is byte-unchanged.
#   canonical  the 2026-05-29 grid + the 2026-09-04 probes + the DEFAULT root.
#              The grid's own pixels no longer exist (decode cache deleted
#              2026-06-22) and the root predates option C, so this reads the
#              published era, NOT the shipped runtime's.
#   postC      the 2026-09-05 re-extraction at HEAD -- THE RUNTIME ERA.
#   preC       the same pixels extracted at 27cfde15, the commit immediately
#              before option C. Exists so `postC - preC` is an era measurement
#              with the pixels held fixed.
#   ladder     the 2026-09-05 FLOOR-DENSE instrument: each codec's own lowest
#              DISTINCT settings at the bottom (jpeg emits ONE bitstream for
#              every q in 0..10, so the older grids' bottom three jpeg steps are
#              one setting sampled three times), and TWO avif ladders. Carries
#              its own peer_ssim2 bars, keyed by grid sha256.
# Individual overrides (ZL_GRID / ZL_NEGTAIL / ZL_IDENTITY / ZL_ROOT / ZL_OUT)
# still win, so a one-off instrument needs no new era name.
set -euo pipefail
REPO=/home/lilith/work/zen/zensimx
D=/mnt/v/output/zensim/dialgate-2026-09-04
I=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments
L=/mnt/v/output/zensim/ladder-2026-09-05/instruments
R="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
V="${ZL_BV:-$REPO/target/release/bake_verdict}"
INC=/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet
BASE=$REPO/zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin

case "${ZL_ERA:-canonical}" in
  canonical)
    GRID_DEF=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet
    NEGTAIL_DEF=$D/negtail_probe_372_2026-09-04.parquet
    IDENTITY_DEF=$D/identity_probe_372_2026-09-04.parquet
    ROOT_DEF=""; OUT_DEF=$D/arms ;;
  postC)
    GRID_DEF=$I/dial_grid_372col_postC_2026-09-05.parquet
    NEGTAIL_DEF=$I/negtail_probe_372_postC_2026-09-05.parquet
    IDENTITY_DEF=$I/identity_probe_372_postC_2026-09-05.parquet
    ROOT_DEF=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
    OUT_DEF=/mnt/v/output/zensim/dpeaks372-2026-09-05/arms/postC ;;
  preC)
    GRID_DEF=$I/dial_grid_372col_preC_2026-09-05.parquet
    NEGTAIL_DEF=$I/negtail_probe_372_preC_2026-09-05.parquet
    # byte-identical to the postC probe (the identity vector is the zero vector
    # at w372 in BOTH eras -- measured, sha256 e6f9096b…, 0 differing cells).
    IDENTITY_DEF=$I/identity_probe_372_postC_2026-09-05.parquet
    ROOT_DEF=/mnt/v/zen/zensim-training/2026-08-30-full-features-372
    OUT_DEF=/mnt/v/output/zensim/dpeaks372-2026-09-05/arms/preC ;;
  ladder)
    # The 2026-09-05 FLOOR-DENSE instrument: five per-codec ladders whose bottom
    # is each codec's own lowest DISTINCT settings, on the 2026-07-27 pixels
    # re-encoded at the current encoder era (zenav1-svt @ 2d75a105f). It is a
    # DIFFERENT instrument from `postC`, not a re-read of it — same 39 sources,
    # but a denser q axis, saturated steps removed by encode hash, and TWO avif
    # ladders (`avif-svt` / `avif-rav1e`) where the older grids have one `avif`.
    # Its own peer_ssim2 bars are registered separately, keyed by grid sha256.
    # 372 by default. For the 944-class lineage BOTH overrides are required —
    # ROOT_DEF below is a 372 features root, and a 944 bake against it is refused
    # loudly (LayoutDiffers + 511 unpopulated slots), which is correct but is not
    # something to rediscover:
    #
    #   ZL_ERA=ladder \
    #   ZL_GRID=/mnt/v/output/zensim/ladder-2026-09-05/instruments/dial_grid_944col_ladder.parquet \
    #   ZL_ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01 \
    #     scripts/dialgate_arms.sh score <label> <bake.bin> 944
    #
    # (that root is `zensim_validate::eval_roots::FEATURES_ROOT_944`; eval_roots
    # names feature ROOTS, not dial grids, so the grid path lives here.)
    # Both widths carry the SAME cells and therefore the same peer_ssim2 bars —
    # verified: identical per-codec fractions on both.
    GRID_DEF=$L/dial_grid_372col_ladder.parquet
    NEGTAIL_DEF=$I/negtail_probe_372_postC_2026-09-05.parquet
    IDENTITY_DEF=$I/identity_probe_372_postC_2026-09-05.parquet
    ROOT_DEF=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
    GRIDTRUTH_DEF=$L/dialcells_ssim2_ladder.tsv
    OUT_DEF=/mnt/v/output/zensim/ladder-2026-09-05/arms ;;
  *) echo "unknown ZL_ERA=${ZL_ERA} (canonical|postC|preC|ladder)" >&2; exit 2 ;;
esac
# The REFERENCE metric's own per-cell scores on the dial grid. Only the
# report-only per-codec column needs it; A7r's bars come from the registry. Same
# table --dial-peer-scores reads, and the ssim2 truth is a property of the
# PIXELS, so one file serves all three 372 eras.
GRIDTRUTH="${ZL_GRIDTRUTH:-${GRIDTRUTH_DEF:-/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv}}"
# Which NEGATIVE-TAIL pin set is in force: `product` (the 2026-09-05 per-codec
# FLOOR REPRESENTABILITY rule, the default) or `retired` (the pre-ruling mentor
# probe-depth pins).
TAILPINS="${ZL_TAILPINS:-product}"
# Which A7r WINDOW `bake_verdict` tests — `distinct` (the pinned rule, the
# default, byte-identical to every prior grade) or `resolvable`/`spaced`
# (owner-extension, 2026-09-06 — see zensim_validate::dial_addressability::
# FloorRule + benchmarks/ladder_floor_resolution_2026-09-05.md). The latter two
# REQUIRE $GRIDTRUTH to resolve to a real file (already true for every era
# above); `bake_verdict` refuses loudly otherwise.
FLOORRULE="${ZL_FLOORRULE:-distinct}"
FLOORMARGIN="${ZL_FLOORMARGIN:-}"
GRID="${ZL_GRID:-$GRID_DEF}"
NEGTAIL="${ZL_NEGTAIL:-$NEGTAIL_DEF}"
IDENTITY="${ZL_IDENTITY:-$IDENTITY_DEF}"
ROOT="${ZL_ROOT:-$ROOT_DEF}"
ARMS="${ZL_OUT:-$OUT_DEF}"

grade() {  # <label> <bake.bin> [regime]
    local LABEL="$1" BAKE="$2" REGIME="${3:-372}"
    local EXTRA=() CORPORA=cid22,konjnd,kadid,tid,aic3
    [ "$REGIME" != "372" ] && EXTRA=(--regime "$REGIME")
    [ -n "$ROOT" ] && EXTRA+=(--features-root "$ROOT")
    mkdir -p "$ARMS"
    # --gaddr-json carries the G-ADDR block at FULL f64 precision; the markdown
    # rounds to 4dp, which is not enough to append a registry pin from.
    [ -n "$GRIDTRUTH" ] && [ -f "$GRIDTRUTH" ] && EXTRA+=(--gaddr-grid-truth "$GRIDTRUTH")
    # Always pass --floor-rule explicitly (same discipline as --gaddr-tail-pins
    # above) so a grade's argv states its rule rather than relying on the
    # binary's default staying `distinct` forever.
    EXTRA+=(--floor-rule "$FLOORRULE")
    [ -n "$FLOORMARGIN" ] && EXTRA+=(--floor-margin "$FLOORMARGIN")
    # --dial-grid is passed for 372 (whose GRID_DEF is a 372 grid) and, for any
    # other regime, ONLY when the caller set ZL_GRID explicitly. Without the second
    # case a `score <bake> 944` silently fell back to bake_verdict's built-in 944
    # grid while LOOKING like it honoured ZL_GRID — the tell is codec names from the
    # other instrument (`avif` rather than `avif-svt`/`avif-rav1e`) and an
    # "unregistered dial grid" note on a grid that IS registered.
    local GRID_ARG=()
    if [ "$REGIME" = "372" ] || [ -n "${ZL_GRID:-}" ]; then
        GRID_ARG=(--dial-grid "$GRID")
    fi
    "$V" --bake "$BAKE" "${EXTRA[@]}" "${GRID_ARG[@]}" \
         --negtail-probe "$NEGTAIL" \
         --identity-probe "$IDENTITY" \
         --corpora "$CORPORA" \
         --gaddr-tail-pins "$TAILPINS" \
         --gaddr-json "$ARMS/gaddr_${LABEL}.json" \
         --full-json "$ARMS/verdict_${LABEL}.json" \
         > "$ARMS/verdict_${LABEL}.log" 2>&1
    echo "== $LABEL (era ${ZL_ERA:-canonical}) =="
    python3 - "$ARMS/verdict_${LABEL}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); dl=d['dial']; a=dl['addressability']
print('  ', a['headline'])
print('   reach %.3f  min %.3f  max %.3f  p5 %.3f  p95 %.3f  DR %.3f  mono %.4f  tied %.4f'%(
    dl['reach'],dl['min'],dl['max'],dl['p5'],dl['p95'],dl['dynamic_range'],dl['mono_pct'],dl['tied_pct']))
m=a['measured']
if m.get('negtail') and m.get('identity'):
    print('   negtail min %.3f p1 %.3f frac<0 %.4f | identity %.4f | above-identity %d'%(
        m['negtail']['min'],m['negtail']['p1'],m['negtail']['frac_below_zero'],
        m['identity']['dial_max'],m['identity']['n_above_identity']))
print('   bar set:', a.get('reference'), '| tail pins:', a.get('tail_pins'),
      '| floor-rule:', a.get('floor_rule'))
for c in (m.get('codec_floor') or []):
    print('   %-5s repr %.4f (bar %s, incumbent %s) %-13s order_fail=%3d clamp_fail=%3d dial_min %9.4f med %s'%(
        c['codec'], c['represented_frac'],
        ('%.4f'%c['represented_frac_reference']) if c.get('represented_frac_reference') is not None else '  -   ',
        ('%.4f'%c['represented_frac_incumbent']) if c.get('represented_frac_incumbent') is not None else '  -   ',
        c['state'], c['n_fail_order'], c['n_fail_clamp'], c['dial_min'],
        ' / '.join('%.2f'%x for x in c['bottom_medians'])))
print('   fails:', [c['id'] for c in a['checks'] if c['state']=='fail'],
      '| not measured:', [c['id'] for c in a['checks'] if c['state']=='not_measured'])
try:
    print('   SROCC cid22 %.5f konjnd %.5f aic3 %.5f tid %.5f kadid %.5f'%tuple(
        d['rank'][k]['srocc_signed'] for k in ['cid22','konjnd','aic3','tid','kadid']))
except KeyError:
    pass
PY
}

if [ "${1:-}" = "score" ]; then
    grade "$2" "$3" "${4:-372}"
    exit 0
fi

LABEL="$1"; ANCHOR="$2"; NEDGES="${3:-18}"; TOPANCHOR="${4:-$2}"
mkdir -p "$ARMS"
"$R" shared-anchor --in "$BASE" --out "$ARMS/${LABEL}_anch.bin" --anchor "$ANCHOR" --n-edges "$NEDGES"
"$R" add-winsor --in "$ARMS/${LABEL}_anch.bin" --out "$ARMS/${LABEL}_wins.bin" \
     --fit-corpus "$INC" --lo-pct 0.1 --hi-pct 99.9 >/dev/null
"$R" extend-top --in "$ARMS/${LABEL}_wins.bin" --out "$ARMS/${LABEL}.bin" \
     --anchor "$TOPANCHOR" --target-col target_score
grade "$LABEL" "$ARMS/${LABEL}.bin"
