#!/bin/bash
# JPEG XL steering maps: our maps against butteraugli and DVIFM 0.3 (benchmarks/byconstruction_2026-10-22.md, "JPEG XL
# steering"). Every arm is encoded by one steer_rd build (zensim-workbench/loop-converge) on the TRAIN corpus written
# by jxl_map_corpus.py ($RD/corpus/corpus.tsv): no loop, the built-in butteraugli loop, the workbench maps of B, Tga
# and M64a at their shipped placements and at their tail-fit placements (*tail: loop_calibrate LOOP_CAL_TAIL, each
# pair's largest tile, "JPEG XL reachability with Tga"), and DVIFM 0.3's map through the private aic4-wasm
# loop_map_server at both kinds of placement. Then the shared judge panel (SSIMULACRA2, butteraugli, zensim and CVVDP
# from ZM_BIN), DVIFM 0.3 (its frozen runtime, dvifm_e21.py judge) and the analysis against no loop and against the
# butteraugli loop. Arms jxl_reach.sh already ran in the same RD_DIR (butteraugli, Tga, Tgatail) are reused.
# Opt-in arms ("JPEG XL steering with the full-data monotone models"): Mfull, Tfull (the full-data certified-monotone
# fits, workbench metrics $MFULL and $TFULL) at their bulk placements (*_BULK, loop_calibrate's shipped method) and as
# Mfulltail, Tfulltail at their tail placements; Bcheck, B's arm again, which a new steer_rd build must reproduce byte
# for byte; margarine, zenmetrics' Butteraugli approximation steering with its own diffmap (MARGARINE: its CLI, served by
# margarine_map_server.py), and further Margarine builds as arms named in MARGARINE_ARMS. Judge tables already in RD_DIR
# move to judged-<UTC time>/ before the panel runs again.
# Linear-input arms ("Correction ... the 8-bit rounding"): LINEAR_ARMS="label=METRIC@SCALE,EXPONENT ..." runs that
# workbench map's diffmap on the loop's unrounded float reconstruction (steer_rd --linear; a "#MAP" suffix names another
# workbench map, e.g. label=METRIC@SCALE,EXPONENT#diffmap_rem); LINEAR_SERVER_ARMS=
# "label=/path/to/server ..." starts `server --server`, which must advertise "input":"linear-rgb-f32" (zenmetrics'
# margarine-steer). Each label is also listed in ARMS.
#
#   STEER_RD=<steer_rd> DVIFM_SERVER=<loop_map_server> ZM_BIN=<zenmetrics CLI> [*_TAIL=SCALE,EXPONENT] \
#     [*_BULK=SCALE,EXPONENT] [ARMS="none butteraugli B Tga M64a dvifm03 Btail Tgatail M64atail dvifm03tail"] \
#     [THREADS=12] jxl_steer_rd.sh RD_DIR
set -euo pipefail
RD=$1
: "${STEER_RD:?set STEER_RD}" "${ZM_BIN:?set ZM_BIN}"
H=$(cd "$(dirname "$0")" && pwd)
DISTANCES=${DISTANCES:-0.3,0.45,0.6,1.0,1.6,2.5,4.0,6.4,10.0}
ITERS=${ITERS:-4}
EFFORT=${EFFORT:-7}
THREADS=${THREADS:-12}
ARMS=${ARMS:-none butteraugli B Tga M64a dvifm03 Btail Tgatail M64atail dvifm03tail}
TGA_TAIL=${TGA_TAIL:-2.2571e-2,0.7}
M64A_TAIL=${M64A_TAIL:-2.4053e-2,0.7}
MFULL=${MFULL:-zensim-Mfull@rev5}
TFULL=${TFULL:-zensim-Tfull@rev5}
: "${B_TAIL:?set B_TAIL (loop_calibrate LOOP_CAL_TAIL, zensim-B@rev5 diffmap, tail max)}"
: "${DVIFM_TAIL:?set DVIFM_TAIL (aic4-wasm loop_calibrate LOOP_CAL_TAIL, aic4-dvifm-lap-v0.3 diffmap, tail max)}"
mkdir -p "$RD"/jxl "$RD"/logs
log="$RD/logs/jxl.log"
run() { # label steer-arguments...
  local label=$1
  shift
  case " $ARMS " in *" $label "*) ;; *) return ;; esac
  if [ -f "$RD/jxl/manifest_$label.tsv" ]; then
    echo "exists: $label"
    return
  fi
  echo "-- $label $(date -u +%H:%M:%SZ)" | tee -a "$log"
  "$STEER_RD" --corpus "$RD/corpus/corpus.tsv" --out "$RD/jxl" --label "$label" --distances "$DISTANCES" \
    --iters "$ITERS" --effort "$EFFORT" --threads "$THREADS" --steer "$@" >>"$log" 2>&1
}
run none none
run butteraugli butteraugli
# The bulk arms name their placements (loop_calibration_2026-10-24c.tsv): the workbench defaults became the tail fits
# on 2026-10-25, and a rebuilt steer_rd must not change what these arms mean.
run B workbench:zensim-B@rev5:diffmap --placement 1.302887e-1,0.6
run Tga workbench:zensim-Tga@rev6:diffmap --placement 1.329536e-1,0.6
run M64a workbench:zensim-M64a-noKADID@rev6:diffmap --placement 1.372091e-1,0.6
case " $ARMS " in *" dvifm03"*) : "${DVIFM_SERVER:?set DVIFM_SERVER}" ;; esac
run dvifm03 server --server "${DVIFM_SERVER:-}" aic4-dvifm-lap-v0.3 diffmap
run Btail workbench:zensim-B@rev5:diffmap --placement "$B_TAIL"
run Tgatail workbench:zensim-Tga@rev6:diffmap --placement "$TGA_TAIL"
run M64atail workbench:zensim-M64a-noKADID@rev6:diffmap --placement "$M64A_TAIL"
run dvifm03tail server --server "${DVIFM_SERVER:-}" aic4-dvifm-lap-v0.3 diffmap "${DVIFM_TAIL%,*}" "${DVIFM_TAIL#*,}"
# CHECK_LABEL names this build's reproduction of B's arm (default Bcheck; a later build takes a new label).
CHECK=${CHECK_LABEL:-Bcheck}
run "$CHECK" workbench:zensim-B@rev5:diffmap --placement 1.302887e-1,0.6
case " $ARMS " in *" margarine "*) : "${MARGARINE:?set MARGARINE (zenmetrics: just margarine-build)}" ;; esac
run margarine server --server python3 "$H/margarine_map_server.py" --margarine "${MARGARINE:-}"
# Further Margarine builds as their own arms: MARGARINE_ARMS="label=/path/to/margarine[@SCALE,EXPONENT] ..." (each label
# also listed in ARMS); the placement defaults to Margarine's own map.
for spec in ${MARGARINE_ARMS:-}; do
  label=${spec%%=*}; bin=${spec#*=}; place=
  case $bin in *@*) place=${bin#*@}; bin=${bin%@*} ;; esac
  run "$label" server --server python3 "$H/margarine_map_server.py" --margarine "$bin" \
    ${place:+--scale "${place%,*}" --exponent "${place#*,}"}
done
for spec in ${LINEAR_ARMS:-}; do
  label=${spec%%=*}; rest=${spec#*=}; map=diffmap
  case $rest in *#*) map=${rest##*#}; rest=${rest%#*} ;; esac
  run "$label" "workbench:${rest%@*}:$map" --placement "${rest##*@}" --linear
done
for spec in ${LINEAR_SERVER_ARMS:-}; do
  label=${spec%%=*}
  run "$label" server --server "${spec#*=}" --server
done
for arm in Mfull Tfull Mfulltail Tfulltail; do
  case " $ARMS " in *" $arm "*) ;; *) continue ;; esac
  case $arm in Mfull*) metric=$MFULL var=MFULL ;; *) metric=$TFULL var=TFULL ;; esac
  case $arm in *tail) var=${var}_TAIL ;; *) var=${var}_BULK ;; esac
  : "${!var:?set $var (loop_calibrate fit of $metric diffmap)}"
  run "$arm" "workbench:$metric:diffmap" --placement "${!var}"
done
case " $ARMS " in *" $CHECK "*)
  python3 - "$RD/jxl" "$CHECK" <<'PY' | tee -a "$log"
import csv, sys
d, check = sys.argv[1], sys.argv[2]
def rows(label):
    return {(r['image'], r['distance']): r['bitstream_sha256']
            for r in csv.DictReader(open(f"{d}/loop_{label}.tsv"), delimiter='\t')}
a, b = rows('B'), rows(check)
bad = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
if bad:
    sys.exit(f"{check}: {len(bad)} of {len(a)} bitstreams differ from B's (first {bad[:3]}); the build changed the encode")
print(f"{check}: all {len(a)} bitstreams equal B's")
PY
  ;;
esac
if ls "$RD"/judge_*.tsv >/dev/null 2>&1; then
  keep="$RD/judged-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir "$keep"
  mv "$RD"/judge_*.tsv "$keep"/
  echo "previous judge tables: $keep" | tee -a "$log"
fi
RD="$RD" ZM_BIN="$ZM_BIN" JUDGES="ssim2 butteraugli zensim cvvdp" bash "$H/../v_next/rd_probe_2026-07-18.sh" judge
python3 "$H/dvifm_e21.py" judge --rd "$RD" --threads "$THREADS" | tee -a "$log"
python3 "$H/../v_next/rd_probe_analyze_2026-07-18.py" "$RD" | tee "$RD/analysis_stdout.md"
for base in none butteraugli B; do
  python3 "$H/../v_next/rd_probe_analyze_2026-07-18.py" "$RD" --paired-vs "$base" >"$RD/paired_vs_$base.md"
done
echo ALL-DONE
