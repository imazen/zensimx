#!/bin/bash
# JXL map-utility arms for the map todo list (benchmarks/byconstruction_2026-10-22.md, "Map todo"): the zensim
# loop's trained diffmap under profile B with each coarse-scale upsample (nearest = today, the E-JBU guided
# redistribution, interpolating kernels and their mass-conserving forms), against the no-loop baseline, on the
# TRAIN-only corpus written by jxl_map_corpus.py ($RD/corpus/corpus.tsv); then the shared judge panel and analysis
# (scripts/v_next/rd_probe_2026-07-18.sh judge, scripts/v_next/rd_probe_analyze_2026-07-18.py).
#
#   JXL_RD_BIN=<zensim_diffmap_rd example built against this checkout's zensim> ZM_BIN=<zenmetrics CLI> \
#     [ARMS="none nearest lanczos3"] jxl_map_rd.sh RD_DIR
set -euo pipefail
RD=$1
: "${JXL_RD_BIN:?set JXL_RD_BIN}" "${ZM_BIN:?set ZM_BIN}"
H=$(cd "$(dirname "$0")" && pwd)
DISTANCES=${DISTANCES:-0.6,1.0,1.6,2.5,4.0,6.4,10.0}
ITERS=${ITERS:-4}
mkdir -p "$RD"/jxl "$RD"/logs
log="$RD/logs/jxl.log"
ARMS=${ARMS:-none nearest guided bilinear catmull-rom bspline lanczos3 bilinear-mass catmull-rom-mass}
run() { # label metric iters [env k=v ...]
  local label=$1 metric=$2 iters=$3
  shift 3
  case " $ARMS " in *" $label "*) ;; *) return ;; esac
  if [ -f "$RD/jxl/manifest_$label.tsv" ]; then
    echo "exists: $label"
    return
  fi
  echo "-- $label $(date -u +%H:%M:%SZ)" | tee -a "$log"
  env "$@" "$JXL_RD_BIN" --metric "$metric" --label "$label" --out-dir "$RD/jxl" \
    --corpus-file "$RD/corpus/corpus.tsv" --distances "$DISTANCES" --iters "$iters" >>"$log" 2>&1
}
run none butteraugli 0
run nearest zensim "$ITERS" JXL_ZENSIM_RD_PROFILE=b
run guided zensim "$ITERS" JXL_ZENSIM_RD_PROFILE=b JXL_ZENSIM_GUIDED_COARSE=1
for k in bilinear catmull-rom bspline lanczos3 bilinear-mass catmull-rom-mass; do
  run "$k" zensim "$ITERS" JXL_ZENSIM_RD_PROFILE=b ZENSIM_DIFFMAP_UPSAMPLE="$k"
done
RD="$RD" ZM_BIN="$ZM_BIN" bash "$H/../v_next/rd_probe_2026-07-18.sh" judge
python3 "$H/../v_next/rd_probe_analyze_2026-07-18.py" "$RD" | tee "$RD/analysis_stdout.md"
