#!/usr/bin/env bash
# diffmap-RD probe matrix runner (2026-07-18) — phase 3 of
# docs/RD_TARGET_EVAL_DESIGN_2026-07-18.md.
#
# Drives BOTH codec probes across the driver matrix, then the independent judge
# panel. Codec binaries are built in the diffmap-RD worktrees and passed via env
# (committed scripts must not hardcode worktree paths — see CLAUDE.md):
#
#   JXL_RD_BIN=<...>/examples/zensim_diffmap_rd \
#   ZQ_RD_BIN=<...>/examples/zq_rd_probe \
#   ZM_BIN=~/work/zen/zenmetrics/target/release/zenmetrics \
#   bash scripts/v_next/rd_probe_2026-07-18.sh [jxl|zenjpeg|judge|all]
#
# JUDGES (default "ssim2 butteraugli zensim") names the zenmetrics metrics the
# judge phase runs, e.g. with `cvvdp` added.
#
# Everything streams to $RD/logs/*.log (tail-able) + accumulating TSVs.
set -euo pipefail

RD=${RD:-/mnt/v/output/zensim/rd-target-eval-2026-07}
CORPUS="$RD/corpus"
WINNER=${WINNER:-/mnt/v/output/zensim/corr-lq/Ebothg_hfgain_winsor.bin}
ADD156=${ADD156:-/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin}
ZM_BIN=${ZM_BIN:-$HOME/work/zen/zenmetrics/target/release/zenmetrics}
DISTANCES=${DISTANCES:-0.6,1.0,1.6,2.5,4.0,6.4}
TARGETS=${TARGETS:-25,40,55,70,80,90}
ITERS=${ITERS:-4}
mkdir -p "$RD"/{jxl,zenjpeg,logs}

phase=${1:-all}

jxl_probe() {
  : "${JXL_RD_BIN:?set JXL_RD_BIN to the worktree zensim_diffmap_rd example}"
  local log="$RD/logs/jxl_probe.log"
  echo "== jxl probe → $log" | tee -a "$log"
  run_jxl() { # label metric iters [env k=v ...]
    local label=$1 metric=$2 iters=$3; shift 3
    echo "-- driver $label ($(date -u +%H:%M:%SZ))" | tee -a "$log"
    env "$@" nice -n19 "$JXL_RD_BIN" \
      --metric "$metric" --label "$label" --out-dir "$RD/jxl" \
      --corpus-file "$CORPUS/jxl_corpus.tsv" \
      --distances "$DISTANCES" --iters "$iters" >>"$log" 2>&1
  }
  run_jxl none        butteraugli 0
  run_jxl butteraugli butteraugli "$ITERS"
  run_jxl zensimB_trained    zensim "$ITERS" JXL_ZENSIM_RD_PROFILE=b
  run_jxl zensimB_model_abs  zensim "$ITERS" JXL_ZENSIM_RD_PROFILE=b JXL_ZENSIM_MODEL_MAP=abs
  run_jxl winner_model_signed zensim "$ITERS" JXL_ZENSIM_RD_PROFILE="bake:$WINNER" JXL_ZENSIM_MODEL_MAP=signed
  run_jxl add156_model_abs    zensim "$ITERS" JXL_ZENSIM_RD_PROFILE="bake:$ADD156" JXL_ZENSIM_MODEL_MAP=abs
  echo "== jxl probe DONE" | tee -a "$log"
}

zenjpeg_probe() {
  : "${ZQ_RD_BIN:?set ZQ_RD_BIN to the worktree zq_rd_probe example}"
  local log="$RD/logs/zenjpeg_probe.log" tsv="$RD/zenjpeg/probe.tsv"
  echo "== zenjpeg probe → $log / $tsv" | tee -a "$log"
  printf 'image\tdriver\ttarget\tbytes\tachieved_score\tpasses\tmax_block_artifact\tencode_ms\n' >"$tsv"
  run_zq() { # label driver [env k=v ...]
    local label=$1 driver=$2; shift 2
    echo "-- driver $label ($(date -u +%H:%M:%SZ))" | tee -a "$log"
    for ppm in "$CORPUS"/*.ppm; do
      env "$@" nice -n19 "$ZQ_RD_BIN" \
        --image "$ppm" --targets "$TARGETS" --driver "$driver" \
        --label "$label" --out-dir "$RD/zenjpeg" >>"$tsv" 2>>"$log"
    done
  }
  run_zq global  global
  run_zq aq      aq
  # NOTE: the winner bake outputs RAW (no dial spline in its bytes) — unusable
  # as a TARGETING scalar here; it runs on the jxl side (distance ladder, no
  # target scale needed). ADD156 carries a dial spline → valid scalar+map driver.
  run_zq aq_add156_abs    aq ZENJPEG_ZQ_PROFILE="bake:$ADD156" ZENJPEG_ZQ_MODEL_MAP=abs
  run_zq picker  picker
  echo "== zenjpeg probe DONE ($(wc -l <"$tsv") rows)" | tee -a "$log"
}

judge() {
  local log="$RD/logs/judge.log"
  echo "== judge panel → $log" | tee -a "$log"
  # zenjpeg decodes are PPM → convert for the judge (image-crate PNG path).
  if compgen -G "$RD/zenjpeg/*.ppm" >/dev/null; then
    (cd "$RD/zenjpeg" && mogrify -format png ./*.ppm) 2>>"$log"
  fi
  python3 - "$RD" <<'EOF'
import csv, glob, os, re, sys
rd = sys.argv[1]
rows = []
# jxl decodes: manifest TSVs carry ref/dist/label/image/distance/bytes.
for mf in glob.glob(f"{rd}/jxl/manifest_*.tsv"):
    with open(mf) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["ref_path"], r["dist_path"], r["label"], r["image"],
                         "d" + r["distance"], r["bytes"], "jxl"))
# zenjpeg decodes: <label>__<stem>__t<T>.png next to the probe TSV.
for png in glob.glob(f"{rd}/zenjpeg/*.png"):
    m = re.match(r"(.+)__(.+)__t(\d+)\.png$", os.path.basename(png))
    if not m:
        continue
    label, stem, t = m.groups()
    rows.append((f"{rd}/corpus/{stem}.png", png, label, stem, f"t{t}", "", "zenjpeg"))
with open(f"{rd}/judge_pairs.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["ref_path", "dist_path", "label", "image", "op", "bytes", "codec"])
    w.writerows(rows)
print(f"{len(rows)} judge pairs")
EOF
  for metric in ${JUDGES:-ssim2 butteraugli zensim}; do
    echo "-- judging with $metric ($(date -u +%H:%M:%SZ))" | tee -a "$log"
    nice -n19 "$ZM_BIN" batch --metric "$metric" \
      --pairs "$RD/judge_pairs.tsv" --output "$RD/judge_$metric.tsv" >>"$log" 2>&1
  done
  echo "== judge DONE" | tee -a "$log"
}

case "$phase" in
  jxl) jxl_probe ;;
  zenjpeg) zenjpeg_probe ;;
  judge) judge ;;
  all) jxl_probe; zenjpeg_probe; judge ;;
  *) echo "usage: $0 [jxl|zenjpeg|judge|all]"; exit 2 ;;
esac
