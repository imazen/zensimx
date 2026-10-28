#!/usr/bin/env python3
"""Prove source-built frozen metric parity before any Stage B declaration."""
import json
from pathlib import Path
import struct
import subprocess
import sys
from chroma_pilot import save,sha
base=Path('/var/tmp/byc-chroma-20261028');out=Path(sys.argv[1]);binary=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True)
records=[]
for r in json.loads((base/'a2/judge-replay/REPLAY.json').read_text())['controls']:
    records.append(dict(reference=r['reference'],distorted=r['distorted'],expected=r['expected_safesyn_score'],expected_ref_hash=None,expected_dist_hash=r['expected_distorted_pixels_sha256'],population='stored-safesyn-control'))
views=['cid00_m256','cid05_m512','im2404_m512','im5044_m256','im9026_m512','safe00_m256','safe20_m512','safe39_m512']
for family in ['ycc444','ycc420','xyb-capped','xyb-extended']:
    root=base/'a2'/family
    ladders={Path(r['image']).stem:r for r in json.loads((root/'LADDERS.json').read_text())}
    audits={r['distorted']:r for r in map(json.loads,(root/'teacher.jsonl').read_text().splitlines())}
    for view in views:
        ladder=ladders[view]
        for i in [0,3,9,len(ladder['cells'])-1]:
            c=ladder['cells'][i];a=audits[c['distorted']]
            records.append(dict(reference=c['image'],distorted=c['distorted'],expected=c['teacher_ssim2'],expected_ref_hash=a['reference_pixels_sha256'],expected_dist_hash=a['distorted_pixels_sha256'],population=family,rung=i))
input=''.join(json.dumps(r)+'\n' for r in records);(out/'pairs.jsonl').write_text(input)
with (out/'source-result.jsonl').open('w') as stdout,(out/'source-result.stderr.log').open('w') as stderr:
    subprocess.run([str(binary)],input=input,text=True,stdout=stdout,stderr=stderr,check=True)
actual=list(map(json.loads,(out/'source-result.jsonl').read_text().splitlines()))
if len(actual)!=len(records):raise ValueError('source replay lost cells')
failures=[]
for e,a in zip(records,actual):
    expected_bits=struct.pack('>d',e['expected']).hex()
    if (a['reference'],a['distorted'])!=(e['reference'],e['distorted']):raise ValueError('replay identities changed')
    bad=[]
    if a['value_bits']!=expected_bits:bad.append('score-bits')
    if a['distorted_pixels_sha256']!=e['expected_dist_hash']:bad.append('candidate-pixels')
    if e['expected_ref_hash'] and a['reference_pixels_sha256']!=e['expected_ref_hash']:bad.append('reference-pixels')
    if bad:failures.append(dict(reference=e['reference'],distorted=e['distorted'],expected=e['expected'],actual=a['value'],difference=a['value']-e['expected'],checks=bad))
receipt=dict(passed=not failures,required=len(records),matched=len(records)-len(failures),binary_sha256=sha(binary),source_revision='b7c2b4b33f3a26b753caabd5264a7fef9c4f4545',package='fast-ssim2=0.8.2',package_checksum='8dcca2000fbfa5923eb54b5bedddadbe5f7c83f7ef65285ebe0ed10daa78d0f0',frozen_target_executable_sha256='7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87',failures=failures)
save(out/'PARITY.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
if failures:raise ValueError('frozen teacher parity failed; Stage B declaration forbidden')
