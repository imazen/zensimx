#!/usr/bin/env python3
"""Measured serial full-cell costs using zenfleet's real S3 BlobStore."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import pyarrow.parquet as pq
from chroma_evif import specs
from chroma_container import chunks
from chroma_pilot import save,sha
nas=Path('/mnt/tower/output/zensim/byc-chroma-20261028');root=nas/'b/cost';root.mkdir(exist_ok=True)
endpoint=os.environ['CHROMA_S3_ENDPOINT'];bucket='zentrain';prefix='jobs/chroma-ladders-b-cost-20261028/blobs'
pins=['232434619478f3ce076e895f3b4342b00f86e8b633a0da5123867d2e45dab800','2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6','37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f']
results=[];group=0
for family,(codec,scales,params) in specs().items():
    if family.startswith('band'):continue
    for label in ['m32','m512','m960','crop512']:
        out=root/(family+'-'+label);out.mkdir(exist_ok=True)
        if (out/'RESULT.json').exists():results.append(json.loads((out/'RESULT.json').read_text()));continue
        views=sorted((nas/'a2/size/src').glob('*_'+label+'.png'));assert len(views)==3
        indices=[0,7,len(params)-1];jobs=[]
        for view in views:
            for i in indices:
                p=json.dumps(params[i],sort_keys=True,separators=(',',':'))
                inputs=[sha(view),hashlib.sha256(p.encode()).hexdigest(),hashlib.sha256((view.stem+'|'+family).encode()).hexdigest(),*pins]
                jobs.append(dict(kind=dict(kind='metric',metric='ssim2-safesyn-b7c2b4b3'),inputs=inputs,cell=dict(image_path=str(view),codec='evif:'+codec,q=95,knob_tuple_json=p),requires=['evif-cas-v1','ssim2-safesyn-b7c2b4b3']))
        save(out/'manifest.json',jobs)
        start=time.monotonic()
        cmd=['/usr/bin/time','-v','-o',str(out/'worker.time'),'zenfleet-worker','--manifest',str(out/'manifest.json'),'--ledger-out',str(out/'ledger.parquet'),'--blobs-r2-bucket',bucket,'--blobs-r2-prefix',prefix,'--r2-endpoint',endpoint,'--exec','/usr/local/bin/evif-exec','--worker','i265-chroma-cost','--provider','lan']
        with (out/'worker.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        seconds=time.monotonic()-start;rows=pq.read_table(out/'ledger.parquet').to_pylist()
        assert len(rows)==len(jobs) and all(r['status']=='done' for r in rows),'cost cells failed: inspect ledger'
        # Check persistence on the actual S3 store before declaring the full grid.
        (out/'artifacts').mkdir(exist_ok=True)
        for row in rows:
            dest=out/'artifacts'/row['output_sha']
            with (out/'download.log').open('a') as log:subprocess.run(['s5cmd','--endpoint-url',endpoint,'cp',f's3://{bucket}/{prefix}/{row["output_sha"]}',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
            assert sha(dest)==row['output_sha'];parts=chunks(dest.read_bytes());native=hashlib.sha256(parts['BITS'][4:]).hexdigest()
            with (out/'native-existence.log').open('a') as log:subprocess.run(['s5cmd','--endpoint-url',endpoint,'ls',f's3://{bucket}/{prefix}/{native}'],stdout=log,stderr=subprocess.STDOUT,check=True)
            scores=json.loads(parts['scor'])['scores'];assert len(scores)==4
        rss=int(next(l.split(':',1)[1] for l in (out/'worker.time').read_text().splitlines() if 'Maximum resident set size' in l))
        result=dict(family=family,view=label,cells=len(jobs),serial_batch_wall_seconds=seconds,serial_seconds_per_cell=seconds/len(jobs),max_rss_kib=rss,host='i265',store='zenfleet S3 BlobStore',native_and_evif_artifacts_checked=len(rows))
        save(out/'RESULT.json',result);results.append(result);group+=1
        save(root/'PROGRESS.json',dict(pid=os.getpid(),stage='cost',groups=len(results),required_groups=16,cells=sum(r['cells'] for r in results)));print(json.dumps(result),flush=True)
save(root/'COST.json',dict(cells=sum(r['cells'] for r in results),groups=len(results),results=results,serial_seconds_per_cell_median=statistics.median(r['serial_seconds_per_cell'] for r in results),serial_seconds_per_cell_max=max(r['serial_seconds_per_cell'] for r in results),max_rss_kib=max(r['max_rss_kib'] for r in results),note='Observed serial batch wall per cell includes worker/process startup, frozen teacher, both diagnostic ports and both S3 CAS writes; not a full-grid runtime projection. Artifact readbacks excluded from timing.'))
save(root/'PROGRESS.json',dict(stage='complete',cells=sum(r['cells'] for r in results)))
