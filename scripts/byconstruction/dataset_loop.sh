#!/usr/bin/env bash
# HISTORICAL: blocked by default; prepend --historical-replay for intentional reproduction.
# Fixed Mdata comparison through the existing float calibration, JXL and BD-rate owners.
# Caller supplies shared heavy.lock and run-heavy for each build/calibrate/run stage.
set -euo pipefail
# Fail before argument expansion, output creation, data reads or builds.
python3 "$(dirname "${BASH_SOURCE[0]}")/historical_guard.py" "${BASH_SOURCE[0]}" "$@"
shift # explicit --historical-replay, validated above
stage=${1:?build|calibrate|run|analyze}
study=${2:?study directory}
cd "$(dirname "$0")/../.."
study=$(realpath "$study")
model="$study/rev7data/full/refit/production-f16.bin"
m5="$HOME/tmp/byc/full/mono-thr-mp32_s0/refit/production-f16.bin"
m7="$HOME/tmp/byc/full/mono-thr-mp32-r7_s0/refit/production-f16.bin"
cal="$study/loop-calibration"
rd=/var/tmp/byconstruction-jxl-steer
export ZENSIM_WORKBENCH_BAKES="zensim-Mfull@rev5=$m5;zensim-B@rev5=$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin;zensim-Mfull@rev7=$m7;zensim-Mdata@rev7=$model"
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
    test -s "$model"
    test -s "$study/rev7data/full/result.json"
    /var/tmp/byconstruction-bin/be72a3dfb039/bake_dial_refit certify-monotone --bake "$model" \
        > "$study/FULL_MONOTONE.txt"
    python3 - "$study" "$model" "$m5" "$m7" <<'PY'
import hashlib, json, sys
from pathlib import Path
study = Path(sys.argv[1])
def pin(p):
    return dict(path=p,sha256=hashlib.file_digest(open(p,'rb'),'sha256').hexdigest())
record = dict(schema='mdata-candidate-freeze-v1',models={k:pin(p) for k,p in zip(('Mdata','M5','M7'),sys.argv[2:])},
              root_frozen=pin(str(study/'rev7data/root/wide/frozen.json')),selection='fixed-final-epoch119',
              recipe='sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1:mono:thr:mp32')
(study/'CANDIDATE_FREEZE.json').write_text(json.dumps(record,indent=1)+'\n')
PY
    CARGO_TARGET_DIR="$study/loop-target" just byc-loop-bins datalegs 'Mdata private TRAIN fit; LAN only; unchanged M5/M7/B controls' \
        > "$study/loop-build-native.log" 2>&1
    python3 - "$study" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
lines = (p/'loop-build-native.log').read_text().splitlines()
binary = Path(lines[-1])
assert (binary/'steer_rd').is_file() and (binary/'loop_calibrate').is_file(), lines[-10:]
(p/'loop-bin').write_text(str(binary)+'\n')
print(binary)
PY
    ;;
calibrate)
    bin=$(cat "$study/loop-bin")
    LOOP_CAL_TAIL=1 LOOP_CAL_MAPS=diffmap \
      LOOP_CAL_LINEAR=/mnt/v/output/margarine/steering-2026-10-25/linear-replay \
      "$bin/loop_calibrate" /mnt/v/zenmetrics-private/aic4/loop-pairs/2026-10-24-hq "$cal.tsv" \
      > "$study/loop-calibration-native.log" 2>&1
    python3 - "$cal.tail.tsv" "$study" <<'PY'
import csv, json, sys
from pathlib import Path
old = Path('/mnt/v/zenmetrics-private/zensimx-workbench/loop_calibration_2026-10-25r7win.tail.tsv')
def rows(p):
    return {(r['metric'],r['tail']):r for r in csv.DictReader(open(p),delimiter='\t') if r['map']=='diffmap'}
a,b = rows(old),rows(sys.argv[1])
keys = [k for k in a if k[0] in ('zensim-Mfull@rev5','zensim-B@rev5','zensim-Mfull@rev7')]
bad = [k for k in keys if k not in b or any(a[k][c]!=b[k][c] for c in ('scale','exponent','fit_log_r','hold_log_r'))]
assert len(keys)==6 and not bad, bad
receipt = dict(control_rows=len(keys),differences=bad,controls_reproduce=True)
(Path(sys.argv[2])/'LOOP_PLACEMENT_CONTROLS.json').write_text(json.dumps(receipt,indent=1)+'\n')
print('All six original M5/M7/B float placement rows reproduce exactly')
PY
    placement zensim-Mdata@rev7
    ;;
run)
    bin=$(cat "$study/loop-bin")
    test -s "$study/LOOP_PLACEMENT_CONTROLS.json"
    STEER_RD="$bin/steer_rd" ZM_BIN=/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/zenmetrics \
      B_TAIL=1.815210e-2,0.7 DVIFM_TAIL=2.960654e-1,0.35 CHECK_LABEL=Bdatachk THREADS=12 \
      RD_COMPARE_MODE=bdrate JUDGE_REPRO_RECEIPT="$study/LOOP_JUDGE_CONTROLS.json" \
      ARMS='Bdatachk MdataMchk Mdata7chk Mdatalintail' \
      LINEAR_ARMS="MdataMchk=zensim-Mfull@rev5@$(placement zensim-Mfull@rev5) Mdata7chk=zensim-Mfull@rev7@$(placement zensim-Mfull@rev7) Mdatalintail=zensim-Mdata@rev7@$(placement zensim-Mdata@rev7)" \
      bash scripts/byconstruction/jxl_steer_rd.sh --historical-replay "$rd" > "$study/loop-native.log" 2>&1
    python3 - "$rd/jxl" "$study" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def rows(label):
    return {(r['image'],r['distance']):r['bitstream_sha256'] for r in csv.DictReader(open(root/f'loop_{label}.tsv'),delimiter='\t')}
receipt = {}
for original,check in [('B','Bdatachk'),('Mlintail','MdataMchk'),('M7lintail','Mdata7chk')]:
    a,b = rows(original),rows(check)
    bad = [k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]
    assert len(a)==702 and not bad, (original,len(a),bad)
    receipt[check] = dict(rows=len(a),different_bitstreams=len(bad))
(Path(sys.argv[2])/'LOOP_BITSTREAM_CONTROLS.json').write_text(json.dumps(receipt,indent=1)+'\n')
print(json.dumps(receipt))
PY
    ;;
analyze)
    for base in M7lintail Mlintail butteraugli; do
        python3 scripts/v_next/rd_probe_analyze_2026-07-18.py "$rd" --bdrate-vs "$base" \
            > "$study/bdrate-vs-$base.md"
    done
    ;;
*) exit 2 ;;
esac
