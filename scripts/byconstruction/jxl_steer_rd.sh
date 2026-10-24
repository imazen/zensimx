#!/bin/bash
# JPEG XL steering maps: our maps against butteraugli and DVIFM 0.3 (benchmarks/byconstruction_2026-10-22.md, "JPEG XL
# steering"). Every arm is encoded by one steer_rd build (zensim-workbench/loop-converge) on the TRAIN corpus written
# by jxl_map_corpus.py ($RD/corpus/corpus.tsv): no loop, the built-in butteraugli loop, the workbench maps of B, Tga
# and M64a at their shipped placements and at their tail-fit placements (*tail: loop_calibrate LOOP_CAL_TAIL, each
# pair's largest tile, "JPEG XL reachability with Tga"), and DVIFM 0.3's map through the private aic4-wasm
# loop_map_server at both kinds of placement. Then the shared judge panel (SSIMULACRA2, butteraugli, zensim and CVVDP
# from ZM_BIN), DVIFM 0.3 (its frozen runtime, dvifm_e21.py judge) and the analysis against no loop and against the
# butteraugli loop. Arms jxl_reach.sh already ran in the same RD_DIR (butteraugli, Tga, Tgatail) are reused.
#
#   STEER_RD=<steer_rd> DVIFM_SERVER=<loop_map_server> ZM_BIN=<zenmetrics CLI> [*_TAIL=SCALE,EXPONENT] \
#     [ARMS="none butteraugli B Tga M64a dvifm03 Btail Tgatail M64atail dvifm03tail"] [THREADS=12] \
#     jxl_steer_rd.sh RD_DIR
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
run B workbench:zensim-B@rev5:diffmap
run Tga workbench:zensim-Tga@rev6:diffmap
run M64a workbench:zensim-M64a-noKADID@rev6:diffmap
case " $ARMS " in *" dvifm03"*) : "${DVIFM_SERVER:?set DVIFM_SERVER}" ;; esac
run dvifm03 server --server "${DVIFM_SERVER:-}" aic4-dvifm-lap-v0.3 diffmap
run Btail workbench:zensim-B@rev5:diffmap --placement "$B_TAIL"
run Tgatail workbench:zensim-Tga@rev6:diffmap --placement "$TGA_TAIL"
run M64atail workbench:zensim-M64a-noKADID@rev6:diffmap --placement "$M64A_TAIL"
run dvifm03tail server --server "${DVIFM_SERVER:-}" aic4-dvifm-lap-v0.3 diffmap "${DVIFM_TAIL%,*}" "${DVIFM_TAIL#*,}"
RD="$RD" ZM_BIN="$ZM_BIN" JUDGES="ssim2 butteraugli zensim cvvdp" bash "$H/../v_next/rd_probe_2026-07-18.sh" judge
rm -f "$RD/judge_dvifm.tsv"
python3 "$H/dvifm_e21.py" judge --rd "$RD" --threads "$THREADS" | tee -a "$log"
python3 "$H/../v_next/rd_probe_analyze_2026-07-18.py" "$RD" | tee "$RD/analysis_stdout.md"
for base in none butteraugli; do
  python3 "$H/../v_next/rd_probe_analyze_2026-07-18.py" "$RD" --paired-vs "$base" >"$RD/paired_vs_$base.md"
done
echo ALL-DONE
