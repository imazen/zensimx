#!/usr/bin/env python3
"""HDR-A3 sensitivity, unchanged E26 teacher and native PQ ingress."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pyarrow.parquet as pq
from chroma_pilot import admission, save, sha
base=Path(sys.argv[1]); root=base/'b/hdr-a3'; root.mkdir(parents=True,exist_ok=True)
(root/'logs').mkdir(exist_ok=True)
sources=json.loads((base/'a2/hdr/SOURCES.json').read_text())
driver=base/'b/bin/chroma_hdr'; teacher=base/'a2/bin/hdr-teacher'
assert sha(teacher)=='1278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721'
conditions=['--hdr','--hdr-common-primaries','--gpu-runtime','cpu','--group-by-ref']
def cmd(name,args):
    start=time.monotonic()
    with (root/'logs'/(name+'.log')).open('a') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-start
records=[]
for s in sources:
    out=root/s['source_family']
    assert sha(s['pilot_ref'])==s['ref_sha256']
    if not (out/'encodes.json').exists():cmd('render-'+s['source_family'],[driver,'a3',s['pilot_ref'],out])
    for r in json.loads((out/'encodes.json').read_text()):
        path=out/r['file'];png=path.with_suffix('.pq.png')
        if not png.exists():cmd('decode-'+s['source_family'],[driver,'decode',path,png])
        if r['quality']==95:
            old=base/'a2/hdr'/s['source_family']/('chroma-%02d.avif'%r['chroma_q'][0])
            assert sha(old)==sha(path),'q95 must replay A2 bitstream'
        r.update(source=s['source_family'],ref_path=s['pilot_ref'],dist_path=str(png),encoded_path=str(path),encode_sha256=sha(path),rendition_sha256=sha(png));records.append(r)
save(root/'ENCODES.json',records)
with (root/'pairs.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path','image_path','codec','q','knob_tuple_json'])
    for r in records:w.writerow([r['ref_path'],r['dist_path'],r['ref_path'],'zenavif-svt-hdr',int(r['quality']),json.dumps(dict(chroma_q=r['chroma_q']))])
seconds={}
# E26 comparison already scored with its exact frozen conditions. The
# owner-selected candidate uses standard cube-root opsin, never PU21.
comparison=root/'hdrvdp3.parquet'
if not comparison.exists():
    seconds['hdrvdp3']=cmd('hdrvdp3-identity-bound',[teacher,'score-pairs','--metric','hdrvdp3',*conditions,'--hdrvdp3-ppd','60','--hdrvdp3-task','quality','--hdrvdp3-input','rgb-bt709','--hdrvdp3-emission','led-lcd-srgb','--hdrvdp3-age','24','--hdrvdp3-surround','none','--pairs-tsv',root/'pairs.tsv','--out-parquet',comparison])
scored=pq.read_table(comparison).to_pylist()
def key(r):return (r['image_path'],float(r['q']),json.dumps(json.loads(r['knob_tuple_json']),sort_keys=True))
bykey={key(r):r for r in scored};assert len(bykey)==len(records)
for r in records:r['hdrvdp3']=10*bykey[(r['ref_path'],r['quality'],json.dumps(dict(chroma_q=r['chroma_q']),sort_keys=True))]['hdrvdp3_jod_imazen_v0_0_1']
natural=base/'b/bin/chroma-natural-hdr'
pairs=''.join(json.dumps(dict(reference=r['ref_path'],distorted=r['dist_path']))+'\n' for r in records)
(root/'natural-pairs.jsonl').write_text(pairs)
natural_out=root/'natural-scores.jsonl'
if not natural_out.exists():
    start=time.monotonic()
    with natural_out.open('x') as stdout,(root/'logs/natural.log').open('w') as stderr:
        subprocess.run([str(natural)],input=pairs,text=True,stdout=stdout,stderr=stderr,check=True)
    seconds['natural_ssim2']=time.monotonic()-start
rows=list(map(json.loads,natural_out.read_text().splitlines()));assert len(rows)==len(records)
by_pair={(r['reference'],r['distorted']):r for r in rows};assert len(by_pair)==len(records)
for r in records:r['natural_ssim2']=by_pair[(r['ref_path'],r['dist_path'])]['value']

ladders=[]
for s in sources:
    for label,q in [('luma',None),('chroma-q95',95),('chroma-q85',85),('chroma-q75',75),('chroma-q60',60)]:
        cc=[r for r in records if r['source']==s['source_family'] and (r['chroma_q'][0]==0 if q is None else r['quality']==q)]
        cc.sort(key=(lambda r:-r['quality']) if q is None else (lambda r:r['chroma_q'][0]))
        values=[r['hdrvdp3'] for r in cc];natural_values=[r['natural_ssim2'] for r in cc]
        ladders.append(dict(source=s['source_family'],ladder=label,natural_ssim2_values=natural_values,hdrvdp3_values=values,natural_endpoint_drop=natural_values[0]-natural_values[-1],comparison_endpoint_drop=values[0]-values[-1],natural_admission=admission(natural_values),comparison_admission=admission(values),cells=cc))
save(root/'LADDERS.json',ladders)
save(root/'SUMMARY.json',dict(unique_cells=len(records),ladders=len(ladders),comparison_sha256=sha(teacher),candidate_sha256=sha(natural),candidate_implementation=rows[0]['implementation'],driver_sha256=sha(driver),seconds=seconds,conditions=conditions,results=[{k:v for k,v in l.items() if k!='cells'} for l in ladders]))
save(root/'PROGRESS.json',dict(stage='complete',completed_encodes=len(records),ladders=len(ladders)))
