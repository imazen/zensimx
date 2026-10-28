#!/usr/bin/env python3
"""The frozen144-cell before/after gate, through the zenfleet worker owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import pyarrow.parquet as pq
from chroma_pilot import save,sha
from chroma_container import chunks

NAS=Path('/mnt/tower/output/zensim/byc-chroma-20261028'); ROOT=NAS/'b/walltime'

def cas(root,digest):return root/digest[:2]/digest[2:4]/digest

def prepare(ctl,teacher):
    ROOT.mkdir(exist_ok=True);before=[];families={};views={};expected=[]
    for result in sorted((NAS/'b/cost').glob('*/RESULT.json')):
        info=json.loads(result.read_text());folder=result.parent
        jobs=json.loads((folder/'manifest.json').read_text());rows=pq.read_table(folder/'ledger.parquet').to_pylist()
        bykey={(r['image_path'],r['knob_tuple_json']):r for r in rows}
        for job in jobs:
            before.append(job);cell=job['cell'];path=Path(cell['image_path']);params=cell['knob_tuple_json']
            view=views.setdefault(str(path),dict(image_path=str(path),source_sha256=sha(path),identity=path.stem))
            w,h=struct.unpack('>II',path.read_bytes()[16:24]);view.update(width=w,height=h)
            family=families.setdefault(info['family'],dict(label=info['family'],codec=cell['codec'],params=[]))
            if params not in family['params']:family['params'].append(params)
            original=bykey[(str(path),params)]['output_sha'];file=folder/'artifacts'/original
            assert sha(file)==original
            expected.append(dict(image_path=str(path),family=info['family'],params_json=params,evif_sha256=original,path=str(file)))
    assert len(before)==144 and len(views)==12 and len(expected)==144
    pairs=''.join(json.dumps(dict(reference=p,distorted=p))+'\n' for p in views)
    checked=subprocess.run([str(teacher)],input=pairs,text=True,stdout=subprocess.PIPE,check=True)
    decoded={r['reference']:r for r in map(json.loads,checked.stdout.splitlines())}
    for path,view in views.items():view['pixels_sha256']=decoded[path]['reference_pixels_sha256']
    save(ROOT/'before.json',before);save(ROOT/'EXPECTED.json',expected)
    pins=['232434619478f3ce076e895f3b4342b00f86e8b633a0da5123867d2e45dab800','2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6','37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f']
    for mode in ['all','none']:
        directory=ROOT/('plan-'+mode);directory.mkdir()
        spec=dict(sources=list(views.values()),families=list(families.values()),implementation_shas=pins,peak_mem_bytes=256*1024*1024,diagnostics=mode)
        save(directory/'spec.json',spec)
        subprocess.run([str(ctl),'declare-evif','--spec',str(directory/'spec.json'),'--out',str(directory/'manifest.json'),'--grid-out',str(directory/'rungs.jsonl')],check=True)
    save(ROOT/'PREPARED.json',dict(cells=144,views=12,families=4,expected_sha256=sha(ROOT/'EXPECTED.json'),before_sha256=sha(ROOT/'before.json')))

def prepare_both():
    path=ROOT/'before-both.json';assert not path.exists()
    jobs=[]
    for metric in json.loads((ROOT/'before.json').read_text()):
        encode=json.loads(json.dumps(metric));cell=encode['cell']
        encode['kind']=dict(kind='encode',codec=cell['codec'],q=cell['q'],knobs=cell['knob_tuple_json'])
        jobs.extend([encode,metric])
    assert len(jobs)==288
    save(path,jobs)

def measure(host,case,serial):
    manifest=ROOT/(case+'.json') if case.startswith('before') else ROOT/('plan-'+('all' if case=='batch-all' else 'none'))/'manifest.json'
    directory=ROOT/host/case;directory.mkdir(parents=True);archive=directory/'cas';archive.mkdir()
    env=os.environ.copy();env['ZEN_ARCHIVE_BLOBS_DIR']=str(archive)
    env['ZEN_CHUNK_WALL_SEC']='0' if serial else '300'
    start=time.monotonic()
    cmd=['/usr/bin/time','-v','-o',str(directory/'worker.time'),'zenfleet-worker','--manifest',str(manifest),'--ledger-out',str(directory/'ledger.parquet'),'--blobs-r2-bucket','zentrain','--blobs-r2-prefix','unused-benchmark','--r2-endpoint','http://127.0.0.1:1','--exec','/usr/local/bin/evif-exec','--worker',host+'-chroma-'+case,'--provider','lan']
    with (directory/'worker.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    seconds=time.monotonic()-start
    save(directory/'TIMING.json',dict(wall_seconds=seconds))
    files=[directory/'ledger.parquet'] if (directory/'ledger.parquet').exists() else sorted(directory.glob('ledger.chunk-*.parquet'))
    assert files,'worker ledger missing'
    rows=[r for f in files for r in pq.read_table(f).to_pylist()];assert len(rows)==(288 if case=='before-both' else 144 if case=='before' else 12)
    assert all(r['status']=='done' for r in rows),'inspect failed worker rows before timing admission'
    expected={(r['image_path'],r['family'],r['params_json']):r for r in json.loads((ROOT/'EXPECTED.json').read_text())}
    checked=0;output=[]
    for row in rows:
        path=cas(archive,row['output_sha']);assert sha(path)==row['output_sha']
        if case=='before-both' and json.loads(row['kind_json'])['kind']=='encode':
            parts=chunks(path.read_bytes());native=hashlib.sha256(parts['BITS'][4:]).hexdigest()
            assert cas(archive,native).read_bytes()==parts['BITS'][4:]
            continue
        if case.startswith('before'):
            params=json.loads(row['knob_tuple_json']);family='xyb-capped' if row['codec'].endswith('color') else 'xyb-extended' if params.get('xyb') else 'ycc444' if params['subsampling']=='s444' else 'ycc420'
            cells=[dict(family=family,params_json=row['knob_tuple_json'],evif_sha256=row['output_sha'])]
        else:cells=json.loads(path.read_text())['cells']
        for cell in cells:
            old=expected[(row['image_path'],cell['family'],cell['params_json'])]
            data=cas(archive,cell['evif_sha256']).read_bytes();assert hashlib.sha256(data).hexdigest()==cell['evif_sha256']
            prior=Path(old['path']).read_bytes();current_chunks=chunks(data);prior_chunks=chunks(prior)
            source_scores=json.loads(prior_chunks['scor'])['scores'];target_scores=json.loads(current_chunks['scor'])['scores']
            target=next(r for r in target_scores if r['metric']=='ssim2');baseline=next(r for r in source_scores if r['metric']=='ssim2')
            assert struct.pack('>d',target['value'])==struct.pack('>d',baseline['value']),'target bits differ'
            if case!='batch-none':assert data==prior,'container bytes differ'
            else:
                for tag in ['HEAD','BITS','PNGF','DIGS','knob','srce']:assert current_chunks[tag]==prior_chunks[tag],'media/recipe differs'
            native=hashlib.sha256(current_chunks['BITS'][4:]).hexdigest();assert cas(archive,native).read_bytes()==current_chunks['BITS'][4:]
            checked+=1;output.append(dict(image_path=row['image_path'],family=cell['family'],params_json=cell['params_json'],evif_sha256=cell['evif_sha256'],target_bits=struct.pack('>d',target['value']).hex()))
    assert checked==144
    save(directory/'PARITY.json',dict(passed=True,cells=checked,full_container_identity=case!='batch-none',target_bits_exact=True,media_and_recipes_exact=True))
    save(directory/'RESULT.json',dict(host=host,case=case,cells=checked,views=12,wall_seconds=seconds,seconds_per_cell=seconds/checked,seconds_per_view=seconds/12,cells_per_second=checked/seconds,view_jobs=len(rows),serial=serial,includes_separate_encode_arm=case=='before-both',worker_time_scope='worker process; warm children are separate processes',parity='144 exact target doubles; full containers exact in before/batch-all'))
    save(directory/'CELLS.json',output);print((directory/'RESULT.json').read_text(),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('ctl',type=Path);p.add_argument('teacher',type=Path)
    sub.add_parser('prepare-both')
    p=sub.add_parser('measure');p.add_argument('host');p.add_argument('case',choices=['before','before-both','batch-all','batch-none']);p.add_argument('--serial',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.ctl,args.teacher)
    elif args.command=='prepare-both':prepare_both()
    else:measure(args.host,args.case,args.serial)
