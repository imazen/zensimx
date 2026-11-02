#!/usr/bin/env bash
# E36 experiment 3a (benchmarks/e36_subset_test_2026-11-01.md): F3, p30L and p2-30 maps in the JPEG XL loop, on the
# owners of dataset_loop.sh (byc-loop-bins, loop_calibrate tail placements, jxl_steer_rd.sh, rd_probe_analyze BD-rate).
# Caller supplies heavy.lock/run-heavy per stage.   e36_loop.sh build|calibrate|run|analyze STUDY_DIR
set -euo pipefail
stage=${1:?build|calibrate|run|analyze}
study=$(realpath "${2:?study directory}")
cd "$(dirname "$0")/../.."
A=$HOME/tmp/e36test
m7="$HOME/tmp/byc/full/mono-thr-mp32-r7_s0/refit/production-f16.bin"
cal="$study/loop-calibration"
rd=/var/tmp/byconstruction-jxl-steer
export ZENSIM_WORKBENCH_BAKES="zensim-B@rev5=$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin;zensim-Mfull@rev7=$m7;zensim-F3@rev7=$HOME/tmp/fchroma-bakes/F3.bin;zensim-p30L@rev7=$A/arms/p30L-s0/pack_e059/refit/production-f16.bin;zensim-p230@rev7=$A/arms/p2-30-s0/pack_e059/refit/production-f16.bin"
placement() {
    python3 - "$cal.tail.tsv" "$1" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')
        if r['metric'] == sys.argv[2] and r['map'] == 'diffmap' and r['tail'] == 'max']
assert len(rows) == 1, (sys.argv[2], len(rows))
print(rows[0]['scale']+','+rows[0]['exponent'])
PY
}
case "$stage" in
build)
    mkdir -p "$study"
    CARGO_TARGET_DIR="$study/loop-target" just byc-loop-bins e36 'E36 exp 3a: F3, p30L, p2-30 maps; B and M7 controls' \
        > "$study/loop-build-native.log" 2>&1
    tail -1 "$study/loop-build-native.log" > "$study/loop-bin"
    test -x "$(cat "$study/loop-bin")/steer_rd" && test -x "$(cat "$study/loop-bin")/loop_calibrate"
    ;;
calibrate)
    bin=$(cat "$study/loop-bin")
    LOOP_CAL_TAIL=1 LOOP_CAL_MAPS=diffmap LOOP_CAL_LINEAR=/mnt/v/output/margarine/steering-2026-10-25/linear-replay \
      "$bin/loop_calibrate" /mnt/v/zenmetrics-private/aic4/loop-pairs/2026-10-24-hq "$cal.tsv" \
      > "$study/loop-calibration-native.log" 2>&1
    python3 - "$cal.tail.tsv" "$study" <<'PY'
import csv, json, sys
from pathlib import Path
old = Path('/mnt/v/zenmetrics-private/zensimx-workbench/loop_calibration_2026-10-25r7win.tail.tsv')
def rows(p):
    return {(r['metric'],r['tail']):r for r in csv.DictReader(open(p),delimiter='\t') if r['map']=='diffmap'}
a,b = rows(old),rows(sys.argv[1])
keys = [k for k in a if k[0] in ('zensim-B@rev5','zensim-Mfull@rev7')]
bad = [k for k in keys if k not in b or any(a[k][c]!=b[k][c] for c in ('scale','exponent','fit_log_r','hold_log_r'))]
assert keys and not bad, bad
(Path(sys.argv[2])/'LOOP_PLACEMENT_CONTROLS.json').write_text(json.dumps(dict(control_rows=len(keys),differences=bad),indent=1)+'\n')
print(f'{len(keys)} B/M7 placement rows reproduce exactly')
PY
    for m in F3 p30L p230; do echo "$m $(placement zensim-$m@rev7)"; done
    ;;
run)
    bin=$(cat "$study/loop-bin")
    test -s "$study/LOOP_PLACEMENT_CONTROLS.json"
    STEER_RD="$bin/steer_rd" ZM_BIN=/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/zenmetrics \
      B_TAIL=1.815210e-2,0.7 DVIFM_TAIL=2.960654e-1,0.35 CHECK_LABEL=Be36chk THREADS=12 \
      RD_COMPARE_MODE=bdrate JUDGE_REPRO_RECEIPT="$study/LOOP_JUDGE_CONTROLS.json" \
      ARMS='Be36chk M7e36chk F3lintail p30Llintail p230lintail' \
      LINEAR_ARMS="M7e36chk=zensim-Mfull@rev7@$(placement zensim-Mfull@rev7) F3lintail=zensim-F3@rev7@$(placement zensim-F3@rev7) p30Llintail=zensim-p30L@rev7@$(placement zensim-p30L@rev7) p230lintail=zensim-p230@rev7@$(placement zensim-p230@rev7)" \
      bash scripts/byconstruction/jxl_steer_rd.sh --historical-replay "$rd" > "$study/loop-native.log" 2>&1
    python3 - "$rd/jxl" "$study" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def rows(label):
    return {(r['image'],r['distance']):r['bitstream_sha256'] for r in csv.DictReader(open(root/f'loop_{label}.tsv'),delimiter='\t')}
receipt = {}
for original,check in [('B','Be36chk'),('M7lintail','M7e36chk')]:
    a,b = rows(original),rows(check)
    bad = [k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]
    assert len(a)==702 and not bad, (original,len(a),len(bad))
    receipt[check] = dict(rows=len(a),different_bitstreams=len(bad))
(Path(sys.argv[2])/'LOOP_BITSTREAM_CONTROLS.json').write_text(json.dumps(receipt,indent=1)+'\n')
print(json.dumps(receipt))
PY
    ;;
analyze)
    for base in butteraugli F3lintail; do
        python3 scripts/v_next/rd_probe_analyze_2026-07-18.py "$rd" --bdrate-vs "$base" > "$study/bdrate-vs-$base.md"
    done
    ;;
*) exit 2 ;;
esac
