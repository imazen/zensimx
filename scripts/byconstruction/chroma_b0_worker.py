#!/usr/bin/env python3
"""Real zenfleet one-cell proof; two arms, CAS objects, exact A2 label readback."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time
import pyarrow.parquet as pq
from chroma_pilot import save,sha
from chroma_container import chunks

def run(a):
    base=Path('/var/tmp/byc-chroma-20261028');out=base/'b'/a.trial;out.mkdir()
    ladders=json.loads((base/'a2/ycc444/LADDERS.json').read_text());cell=next(l for l in ladders if l['image'].endswith('cid00_m256.png'))['cells'][3]
    params=json.loads(cell['params']);paramstr=json.dumps(params,sort_keys=True,separators=(',',':'))
    import hashlib
    inputs=[sha(cell['image']),hashlib.sha256(paramstr.encode()).hexdigest(),'2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6']
    identity=dict(image_path=cell['image'],codec='evif:zenjpeg-expert',q=95,knob_tuple_json=paramstr)
    jobs=[dict(kind=kind,inputs=inputs,cell=identity,requires=['evif-cas-v1','ssim2-safesyn-b7c2b4b3']) for kind in [dict(kind='encode',codec=identity['codec'],q=95,knobs=paramstr),dict(kind='metric',metric='ssim2-safesyn-b7c2b4b3')]]
    save(out/'manifest.json',jobs);save(out/'EXPECTED.json',cell)
    args=['docker','run','--name','chroma-'+a.trial,'--cpu-shares=2048','--memory=4g','--memory-swap=4g','--network','none','-e','TMPDIR=/scratch','-e','ZEN_CHUNK_WALL_SEC=0','-v',str(base)+':'+str(base)+':ro','-v',str(out)+':/output','-v',str(base/'b/scratch')+':/scratch']
    if a.executor:args+=['-v',str(a.executor)+':/usr/local/bin/evif-exec:ro','-v',str(base/'b/bin/zenmetrics-diagnostics')+':/opt/chroma/bin/zenmetrics-diagnostics:ro']
    args+=['--entrypoint','nice',a.image,'-n','19','ionice','-c','3','zenfleet-worker','--manifest',str(out/'manifest.json'),'--ledger-out','/output/ledger.parquet','--blobs','/output/blobs','--exec','/usr/local/bin/evif-exec','--worker','i270-chroma-b0','--provider','lan']
    start=time.monotonic()
    with (out/'worker.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    seconds=time.monotonic()-start;rows=pq.read_table(out/'ledger.parquet').to_pylist()
    assert len(rows)==2 and all(r['status']=='done' for r in rows),'inspect worker/ledger; both arms must succeed'
    metric=None
    for row in rows:
        path=out/'blobs'/row['output_sha'];assert sha(path)==row['output_sha']
        c=chunks(path.read_bytes());jpeg=c['BITS'][4:]
        assert hashlib.sha256(jpeg).hexdigest()==cell['encode_sha256']
        assert (out/'blobs'/cell['encode_sha256']).read_bytes()==jpeg
        with (out/'verify.log').open('a') as log:subprocess.run([str(base/'a2/bin/evif'),'verify',path],stdout=log,stderr=subprocess.STDOUT,check=True)
        if json.loads(row['kind_json'])['kind']=='metric':metric=json.loads(c['scor'])['scores']
    assert metric is not None
    target=next(r for r in metric if r['metric']=='ssim2' and r['variant']=='safesyn-teacher')
    assert struct.pack('>d',target['value'])==struct.pack('>d',cell['teacher_ssim2'])
    assert len(metric)==4 and any(r['metric'].startswith('cvvdp_cpu_') for r in metric)
    save(out/'PROOF.json',dict(passed=True,cells=1,arms=2,seconds=seconds,teacher_value=target['value'],teacher_value_bits=struct.pack('>d',target['value']).hex(),jpeg_sha256=cell['encode_sha256'],scores=metric,ledger=rows,image=a.image,executor_sha256=sha(a.executor) if a.executor else None,checks=['both ledger rows DONE','CAS EVIF hashes verified','both EVIFs natively verify','both contain exact A2 JPEG','separate native JPEG CAS matches BITS','teacher f64 exactly replays A2','both diagnostic ports persisted']))
    print(json.dumps(dict(passed=True,seconds=seconds,teacher=target['value'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trial');p.add_argument('image');p.add_argument('--executor',type=Path);run(p.parse_args())
