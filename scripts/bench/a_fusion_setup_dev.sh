#!/usr/bin/env bash
# Reuse the verified local archive; do not copy data from a measuring peer.
set -euo pipefail
[[ $(hostname) = dev ]] || exit 2
study=${1:?dev study root}
archive=${2:?read-only frozen cost archive}
test -f "$study/A_REPORTED.json"
if [[ ${A_FUSION_DEV_SETUP_LOCK_HELD:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 24G --jobs 2 -- \
        env A_FUSION_DEV_SETUP_LOCK_HELD=1 bash "$0" "$@"
fi
export TMPDIR="$study/scratch"
mkdir -p "$TMPDIR"
out="$study/setup"
test ! -e "$out"
test ! -e "$study/pairs"
mkdir -p "$out"
python3 - "$archive" <<'PY'
import hashlib,pathlib,sys
r=pathlib.Path(sys.argv[1])
for name,expected in {
 'pairs/pairs.json':'460eedfaa3b8830944a0b969b384f0fabd16077869e0b7d1de6d97cf2512f13c',
 'inputs-A0.tsv':'fe29972ecb019b64fff0a9f8c48afaedeb43bfe606d4753982987d75e11fc224',
 'production-f16.bin':'185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088',
}.items():
 with (r/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected,name
PY
cp -r "$archive/pairs" "$out/pairs"
cp "$archive/production-f16.bin" "$archive/inputs-A0.tsv" "$study/"
python3 - "$out" "$archive" <<'PY'
import collections,hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]);r=out/'pairs'
rows=json.loads((r/'pairs.json').read_text());assert len(rows)==80
sizes=collections.Counter(row['size'] for row in rows)
assert sizes=={'64':20,'256':20,'1024':20,'large':20},sizes
for row in rows:
 for arm,key in [('ref','reference_sha256'),('dist','distorted_sha256')]:
  p=r/f"{row['id']}.{arm}.rgb"
  assert p.stat().st_size==row['width']*row['height']*3,p
  with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==row[key],p
(out/'INPUTS.json').write_text(json.dumps(dict(archive_host='dev',archive=sys.argv[2],
 records=len(rows),sizes=dict(sizes),all_reference_and_distorted_hashes_match=True),indent=2)+'\n')
PY
mv "$out/pairs" "$study/pairs"
status=0
sudo modprobe amd_uncore > "$out/amd-uncore.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$out/amd-uncore.returncode"
ls /sys/bus/event_source/devices > "$out/PMU_DEVICES.txt"
perf list -j PipelineL1 PipelineL2 l2_cache l3_cache memory_controller > "$out/PERF_METRICS.json"
lscpu > "$out/CPU.txt"
for cpu in 0 8; do
    cat "/sys/devices/system/cpu/cpu$cpu/cache/index3/shared_cpu_list" \
        "/sys/devices/system/cpu/cpu$cpu/cache/index3/size" > "$out/L3-cpu$cpu.txt"
done
sha256sum "$study/pairs/pairs.json" "$study/production-f16.bin" "$study/inputs-A0.tsv" "$0" > "$out/SHA256SUMS"
printf '%s dev setup complete\n' "$(date -u +%FT%TZ)" > "$out/COMPLETE.txt"
