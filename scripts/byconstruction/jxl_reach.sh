#!/bin/bash
# Can Tga's map steer jxl-encoder's loop to the requested distance, and does a miss need better placement fits or a
# different map? (benchmarks/byconstruction_2026-10-22.md, "JPEG XL reachability with Tga"). One steer_rd build
# (zensim-workbench/loop-converge) on the TRAIN corpus written by jxl_map_corpus.py ($RD/corpus/corpus.tsv):
#   none          no loop at 20 encoder distances (0.05-10): can any uniform quality reach a target? (also the
#                 steering study's no-loop baseline: the same bitstreams, on a denser grid)
#   butteraugli   the built-in loop (the reference behaviour)
#   Tga           Tga's diffmap at its shipped placement
#   Tgatail       the same map at the tail-fit placement (TAIL, loop_calibrate LOOP_CAL_TAIL on the calibration pairs)
#   Tgatail8      the same with 8 iterations
#   noneamp, Tgaamp, M64aamp, Bamp
#                 the amplitude split (diffmap_amp) at its tail-fit placements (AMP_TGA, AMP_M64A, AMP_B): the no-loop
#                 sweep probed with it, and each map steering the loop
# Every run is probed on its decoded output by Tga's map at both placements, and the loop runs dump their frames
# ($RD/frames) for jxl_reach.py elastic. The arms share labels and settings with jxl_steer_rd.sh, which reuses them.
#
#   STEER_RD=<steer_rd> [TAIL=SCALE,EXPONENT] [THREADS=12] jxl_reach.sh RD_DIR
set -euo pipefail
RD=$1
: "${STEER_RD:?set STEER_RD}"
H=$(cd "$(dirname "$0")" && pwd)
THREADS=${THREADS:-12}
DISTANCES=${DISTANCES:-0.3,0.45,0.6,1.0,1.6,2.5,4.0,6.4,10.0}
STATIC=${STATIC:-0.05,0.075,0.1,0.15,0.2,0.3,0.45,0.6,0.8,1.0,1.3,1.6,2.0,2.5,3.2,4.0,5.0,6.4,8.0,10.0}
TGA=zensim-Tga@rev6
TAIL=${TAIL:-2.2571e-2,0.7}
AMP_TGA=${AMP_TGA:-4.7213e-2,0.9}
AMP_M64A=${AMP_M64A:-4.6644e-2,1.0}
AMP_B=${AMP_B:-4.5157e-2,0.9}
mkdir -p "$RD"/jxl "$RD"/logs "$RD"/frames
log="$RD/logs/reach.log"
run() { # label steer-arguments...
  local label=$1
  shift
  if [ -f "$RD/jxl/manifest_$label.tsv" ]; then
    echo "exists: $label"
    return
  fi
  echo "-- $label $(date -u +%H:%M:%SZ)" | tee -a "$log"
  "$STEER_RD" --corpus "$RD/corpus/corpus.tsv" --out "$RD/jxl" --label "$label" --threads "$THREADS" --effort 7 \
    --frames "$RD/frames" --probe "workbench:$TGA:diffmap" --probe "workbench:$TGA:diffmap@$TAIL" "$@" >>"$log" 2>&1
}
run none --steer none --distances "$STATIC"
run butteraugli --steer butteraugli --iters 4 --distances "$DISTANCES"
run Tga --steer "workbench:$TGA:diffmap" --iters 4 --distances "$DISTANCES"
run Tgatail --steer "workbench:$TGA:diffmap" --placement "$TAIL" --iters 4 --distances "$DISTANCES"
run Tgatail8 --steer "workbench:$TGA:diffmap" --placement "$TAIL" --iters 8 --distances "$DISTANCES"
AMPPROBE="workbench:$TGA:diffmap_amp@$AMP_TGA"
run noneamp --steer none --distances "$STATIC" --probe "$AMPPROBE"
run Tgaamp --steer "workbench:$TGA:diffmap_amp" --placement "$AMP_TGA" --iters 4 --distances "$DISTANCES" --probe "$AMPPROBE"
run M64aamp --steer "workbench:zensim-M64a-noKADID@rev6:diffmap_amp" --placement "$AMP_M64A" --iters 4 \
  --distances "$DISTANCES" --probe "$AMPPROBE"
run Bamp --steer "workbench:zensim-B@rev5:diffmap_amp" --placement "$AMP_B" --iters 4 --distances "$DISTANCES" \
  --probe "$AMPPROBE"
{
  for tag in "$TGA:diffmap@1.329536e-1,0.6" "$TGA:diffmap@${TAIL%,*},${TAIL#*,}"; do
    python3 "$H/jxl_reach.py" static --rd "$RD" --label none --probe "$tag"
  done
  python3 "$H/jxl_reach.py" loop --rd "$RD" --label butteraugli --label Tga --label Tgatail --label Tgatail8 \
    --probe "$TGA:diffmap@${TAIL%,*},${TAIL#*,}" --by-class --exclude "$RD/corpus/tail_fit_overlap.txt"
  python3 "$H/jxl_reach.py" elastic --rd "$RD" --label butteraugli --label Tga --label Tgatail
  AMPTAG="$TGA:diffmap_amp@${AMP_TGA%,*},${AMP_TGA#*,}"
  python3 "$H/jxl_reach.py" static --rd "$RD" --label noneamp --probe "$AMPTAG"
  python3 "$H/jxl_reach.py" loop --rd "$RD" --label Tgaamp --label M64aamp --label Bamp --probe "$AMPTAG" --by-class \
    --exclude "$RD/corpus/tail_fit_overlap.txt"
  python3 "$H/jxl_reach.py" elastic --rd "$RD" --label Tgaamp --label M64aamp --label Bamp
} | tee "$RD/reach_summary.txt"
echo ALL-DONE
