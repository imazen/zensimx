#!/usr/bin/env python3
"""A2 native HDR pilot, E26 binary/conditions, no new teacher implementation."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from chroma_pilot import admission,save,sha
import pyarrow.parquet as pq
root=Path(sys.argv[1]);base=root.parent
sources=json.loads((root/'SOURCES.json').read_text())
teacher=base/'bin/hdr-teacher'; driver=base/'bin/chroma_hdr'
assert sha(teacher)=='1278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721'
conditions=['--metric','hdrvdp3','--hdr','--hdr-common-primaries','--gpu-runtime','cpu','--hdrvdp3-ppd','60','--hdrvdp3-task','quality','--hdrvdp3-input','rgb-bt709','--hdrvdp3-emission','led-lcd-srgb','--hdrvdp3-age','24','--hdrvdp3-surround','none','--group-by-ref']
(root/'logs').mkdir(exist_ok=True)
def cmd(label,args):
    start=time.monotonic()
    with (root/'logs'/(label+'.log')).open('w') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-start
control=next(s for s in sources if s.get('pilot_control_dist'))
for s in sources:
    assert sha(s['pilot_ref'])==s['ref_sha256']
assert sha(control['pilot_control_dist'])==control['dist_sha256']
pairs=root/'control.tsv'
with pairs.open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path']);w.writerow([control['pilot_ref'],control['pilot_control_dist']])
control_seconds=None if (root/'control.parquet').exists() else cmd('control',[teacher,'score-pairs',*conditions,'--pairs-tsv',pairs,'--out-parquet',root/'control.parquet'])
rows=pq.read_table(root/'control.parquet').to_pylist()
column=next(k for k in rows[0] if k.startswith('hdrvdp3') and isinstance(rows[0][k],(float,int)))
actual=rows[0][column];expected=control['hdrvdp3_q_jod']
if not (root/'CONTROL.json').exists():save(root/'CONTROL.json',dict(actual=actual,expected=expected,delta=actual-expected,exact=actual==expected,column=column,teacher_sha256=sha(teacher),conditions=conditions))
if actual!=expected:raise ValueError('E26 frozen teacher replay differs: inspect before admitting new HDR labels')
records=[];render={}
for source in sources:
    identity=source['source_family'];out=root/identity
    if not out.exists():render[identity]=cmd('render-'+identity,[driver,source['pilot_ref'],out])
    for r in json.loads((out/'encodes.json').read_text()):
        path=out/r['file']
        previous=root/'first-encodes'/identity/r['file']
        if previous.exists() and sha(previous)!=sha(path):raise ValueError('deterministic AVIF replay differs')
        png=path.with_suffix('.pq.png')
        if not png.exists():cmd('decode-'+identity+'-'+path.stem,[driver,'decode',path,png])
        r.update(ref_path=source['pilot_ref'],dist_path=str(png),encoded_path=str(path),source_family=identity,encode_sha256=sha(path),rendition_sha256=sha(png));records.append(r)
save(root/'ENCODES.json',records)
pairs=root/'pairs.tsv'
with pairs.open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path','image_path','codec','q','knob_tuple_json'])
    for r in records:w.writerow([r['ref_path'],r['dist_path'],r['ref_path'],'zenavif-svt-hdr',95,json.dumps(dict(chroma_q=r['chroma_q']))])
seconds=cmd('teacher',[teacher,'score-pairs',*conditions,'--pairs-tsv',pairs,'--out-parquet',root/'scores.parquet'])
score_rows=pq.read_table(root/'scores.parquet').to_pylist();save(root/'SCORES_RAW.json',score_rows)
# Identity must survive the scoring transport; never rely on output order.
def key(r):return (r['image_path'],json.dumps(json.loads(r['knob_tuple_json']),sort_keys=True))
bykey={key(r):r for r in score_rows}
if len(bykey)!=len(records):raise ValueError('HDR score transport lost or duplicated identities')
results=[]
for s in sources:
    cells=[]
    for r in records:
        if r['source_family']!=s['source_family']:continue
        sr=bykey[(r['ref_path'],json.dumps(dict(chroma_q=r['chroma_q']),sort_keys=True))]
        r['teacher_q_jod']=sr[column];r['teacher_score']=10*sr[column];cells.append(r)
    cells.sort(key=lambda c:c['chroma_q'][0]);results.append(dict(source=s['source_family'],admission=admission([r['teacher_score'] for r in cells]),cells=cells))
save(root/'LADDERS.json',results)
save(root/'SUMMARY.json',dict(ladders=len(results),passed=sum(bool(r['admission']['kept_indices']) for r in results),kept=sum(len(r['admission']['kept_indices']) for r in results),cells=len(records),teacher_seconds=seconds,control_seconds=control_seconds,render_seconds=render,teacher_sha256=sha(teacher),driver_sha256=sha(driver),teacher_conditions=conditions,missing='SVT pin explicitly refuses 4:4:4; HDR samples stored as files, not legacy RGB8 EVIF'))
