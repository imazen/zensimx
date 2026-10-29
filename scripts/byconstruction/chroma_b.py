#!/usr/bin/env python3
"""Scientific Stage B specification/admission; zenfleet owns execution and retries."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import shutil
import gzip
from datetime import datetime, timezone
from pathlib import Path
import struct
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq
from chroma_evif import specs
from chroma_container import chunks
from chroma_pilot import admission, save, sha

NAS = Path('/mnt/tower/output/zensim/byc-chroma-20261028')
ROOT = NAS / 'b/full'
PINS = ['232434619478f3ce076e895f3b4342b00f86e8b633a0da5123867d2e45dab800',
        '2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6',
        '37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f',
        '6ec626d6c13ae45a68516a77676eb3da477322f9b5ca70adc637e09019ef7476',
        '062d96f9fd881d82c279f1dfbbb2dee1783f968cec7316d757088bad0ef87758']


def cas(root, digest):
    return root / digest[:2] / digest[2:4] / digest


def diagnostic_reason(view_id, family, rung, endpoints):
    sample=int(hashlib.sha256((view_id+'|'+family+'|'+str(rung)).encode()).hexdigest(),16)%20==0
    return 'endpoint+sample' if sample and rung in endpoints else 'endpoint' if rung in endpoints else 'sample' if sample else None


def specification(out, one_view):
    assert not out.exists(), 'preserve existing declaration specification'
    views = json.loads((ROOT / 'VIEWS.json').read_text())
    frozen = {r['id']: r for r in json.loads((ROOT / 'FROZEN_REFERENCE_PIXELS.json').read_text())}
    assert json.loads((ROOT / 'REFERENCE_REPLAY.json').read_text())['passed']
    if one_view:
        views = [next(v for v in views if v['width'] == v['height'] == 960)]
    sources = []
    for v in views:
        assert v['pixels_sha256'] == frozen[v['id']]['reference_pixels_sha256']
        assert v['sha256'] == frozen[v['id']]['sha256']
        sources.append(dict(image_path=v['path'], source_sha256=v['sha256'], pixels_sha256=v['pixels_sha256'],
                            identity=json.dumps({k:v[k] for k in ['id','reference_group','mlic_role','type','content_class']},sort_keys=True),
                            width=v['width'], height=v['height']))
    families = [dict(label=label, codec='evif:'+codec,
                     params=[json.dumps(p,sort_keys=True,separators=(',',':')) for p in params])
                for label,(codec,_,params) in specs().items() if not label.startswith('band-')]
    assert sum(len(f['params']) for f in families) == 76
    save(out, dict(sources=sources, families=families, implementation_shas=PINS,
                   peak_mem_bytes=256*1024*1024, diagnostics='none'))
    print(json.dumps(dict(spec=str(out), views=len(sources), cells=len(sources)*76)), flush=True)


def audit_cell(archive, cell, check_native):
    path=cas(archive,cell['evif_sha256']);data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==cell['evif_sha256']
    parts=chunks(data);jpeg=parts['BITS'][4:]
    assert hashlib.sha256(jpeg).hexdigest()==cell['jpeg_sha256']
    if check_native:assert cas(archive,cell['jpeg_sha256']).read_bytes()==jpeg
    target=json.loads(parts['scor'])['scores'];assert len(target)==1 and target[0]['metric']=='ssim2'
    assert target[0]['variant']=='safesyn-teacher' and target[0]['better']=='higher'
    assert target[0]['implementation']=='fast-ssim2@b7c2b4b33f3a26b753caabd5264a7fef9c4f4545;crate=0.8.2;source-helper-sha256='+PINS[1]
    assert struct.pack('>d',target[0]['value']).hex()==cell['teacher_bits']


def admit(ledger, archive, out, qualification=False, verify_threads=1):
    assert 1<=verify_threads<=32
    out.mkdir()
    views = {v['path']:v for v in json.loads((ROOT/'VIEWS.json').read_text())}
    rows = pq.read_table(ledger).to_pylist()
    if qualification:
        assert len(rows)==1 and out.name.startswith('qualification-'), 'one-view qualification stays separate from full data'
    else:
        assert len(rows)==len(views)==9162, 'full Stage B admission requires every declared view'
    assert all(r['status']=='done' for r in rows)
    assert len({r['image_path'] for r in rows})==len(rows) and all(r['image_path'] in views for r in rows)
    families = {k:(backend,scales,params) for k,(backend,scales,params) in specs().items() if not k.startswith('band-')}
    ladders=[]; selections=[]; buffer=[]; count=0; native_verified=set(); totals=Counter(); classes=defaultdict(Counter); roles=defaultdict(Counter)
    writer = None
    verifier=ThreadPoolExecutor(max_workers=verify_threads)
    try:
        for i, row in enumerate(sorted(rows,key=lambda r:r['image_path'])):
            view=views[row['image_path']]
            manifest_path=cas(archive,row['output_sha']); assert sha(manifest_path)==row['output_sha']
            manifest=json.loads(manifest_path.read_text())
            assert manifest['source_sha256']==view['sha256'] and manifest['reference_pixels_sha256']==view['pixels_sha256']
            assert manifest['image_path']==row['image_path'] and manifest['knob_tuple_json']==row['knob_tuple_json']
            assert manifest['runtime']['canonical_teacher_helper']==PINS[1] and manifest['runtime']['diagnostics']=='none'
            assert manifest['runtime']['batch_binaries']=={'evif-batch':PINS[3],'chroma-frozen-batch-teacher':PINS[4]}
            new_native={}
            for cell in manifest['cells']:
                if cell['jpeg_sha256'] not in native_verified:new_native.setdefault(cell['jpeg_sha256'],id(cell))
            list(verifier.map(lambda cell:audit_cell(archive,cell,new_native.get(cell['jpeg_sha256'])==id(cell)),manifest['cells']))
            native_verified.update(new_native)
            by_family=defaultdict(list)
            for cell in manifest['cells']:by_family[cell['family']].append(cell)
            assert set(by_family)==set(families)
            selected=[]
            for family,(backend,scales,params) in families.items():
                cells=sorted(by_family[family],key=lambda r:r['rung'])
                assert len(cells)==len(params)
                values=[c['teacher_ssim2'] for c in cells];gate=admission(values)
                endpoints={gate['kept_indices'][0],gate['kept_indices'][-1]} if gate['kept_indices'] else set()
                totals[family+'_ladders']+=1;totals[family+'_admitted']+=bool(gate['kept_indices']);totals[family+'_kept']+=len(gate['kept_indices'])
                classes[view['content_class']][family+'_ladders']+=1
                roles[view['mlic_role']][family+'_ladders']+=1
                roles[view['mlic_role']][family+'_admitted']+=bool(gate['kept_indices'])
                roles[view['mlic_role']][family+'_kept']+=len(gate['kept_indices'])
                classes[view['content_class']][family+'_admitted']+=bool(gate['kept_indices'])
                classes[view['content_class']][family+'_kept']+=len(gate['kept_indices'])
                ladders.append(dict(view_id=view['id'],source=view['source'],reference_group=view['reference_group'],
                                    mlic_role=view['mlic_role'],content_class=view['content_class'],family=family,
                                    scores=values,kept_indices=gate['kept_indices'],candidate_indices=gate['candidate_indices'],
                                    stop_index=gate['stop_index'],drop_reason=gate['drop_reason'],admitted=bool(gate['kept_indices'])))
                for rung,cell in enumerate(cells):
                    assert cell['rung']==rung and json.loads(cell['params_json'])==params[rung]
                    assert cell['teacher_bits']==struct.pack('>d',cell['teacher_ssim2']).hex()
                    assert cell['reference_pixels_sha256']==view['pixels_sha256']
                    reason=diagnostic_reason(view['id'],family,rung,endpoints)
                    if reason:
                        selected.append(cell['evif_sha256'])
                        selections.append(dict(view_id=view['id'],image_path=view['path'],family=family,rung=rung,
                                               evif_sha256=cell['evif_sha256'],jpeg_sha256=cell['jpeg_sha256'],
                                               reason=reason,
                                               content_class=view['content_class'],mlic_role=view['mlic_role']))
                    buffer.append(dict(**cell,view_id=view['id'],source=view['source'],image_path=view['path'],
                                       reference_group=view['reference_group'],mlic_role=view['mlic_role'],
                                       content_class=view['content_class'],source_type=view['type'],width=view['width'],height=view['height'],
                                       kept=rung in gate['kept_indices'],diagnostic_selected=bool(reason)))
                    count+=1
            if len(buffer)>=7600 or i+1==len(rows):
                table=pa.Table.from_pylist(buffer)
                if writer is None:writer=pq.ParquetWriter(out/'CELLS.parquet',table.schema,compression='zstd')
                writer.write_table(table);buffer=[]
                save(out/'PROGRESS.json',dict(views=i+1,required_views=len(rows),cells=count,stage='hash verification and admission'))
    finally:
        verifier.shutdown(wait=True)
        if writer:writer.close()
    persisted=pq.ParquetFile(out/'CELLS.parquet')
    assert persisted.schema_arrow.field('teacher_ssim2').type==pa.float64()
    readback=0
    for batch in persisted.iter_batches(columns=['teacher_ssim2','teacher_bits']):
        for value,bits in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist()):
            assert struct.pack('>d',value).hex()==bits,'Parquet target bits differ'
            readback+=1
    assert readback==count
    assert count==len(rows)*76
    if not qualification:assert count==696312
    pq.write_table(pa.Table.from_pylist(ladders),out/'LADDERS.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(selections),out/'DIAGNOSTICS_SELECTION.parquet',compression='zstd')
    save(out/'SUMMARY.json',dict(qualified_full_grid=not qualification,cells=count,views=len(rows),ladders=len(ladders),totals=dict(totals),
                                 per_class={k:dict(v) for k,v in classes.items()},per_role={k:dict(v) for k,v in roles.items()},native_objects_verified=len(native_verified),
                                 diagnostics_selected=len(selections),ledger_sha256=sha(ledger)))
    save(out/'PROGRESS.json',dict(stage='complete',views=len(rows),cells=count))


def verify_first(ledger, archive, evif, out):
    rows=pq.read_table(ledger).to_pylist()
    assert len(rows)==1 and rows[0]['status']=='done'
    manifest=cas(archive,rows[0]['output_sha'])
    assert sha(manifest)==rows[0]['output_sha']
    result=json.loads(manifest.read_text());assert len(result['cells'])==76
    with out.with_suffix('.verify.log').open('w') as log:
        for cell in result['cells']:
            path=cas(archive,cell['evif_sha256']);assert sha(path)==cell['evif_sha256']
            parts=chunks(path.read_bytes());native=parts['BITS'][4:]
            assert hashlib.sha256(native).hexdigest()==cell['jpeg_sha256']
            assert cas(archive,cell['jpeg_sha256']).read_bytes()==native
            value=json.loads(parts['scor'])['scores'][0]['value']
            assert struct.pack('>d',value).hex()==cell['teacher_bits']
            subprocess.run([str(evif),'verify',str(path)],stdout=log,stderr=subprocess.STDOUT,check=True)
    save(out,dict(passed=True,view=result['image_path'],cells=76,primary_sha256=rows[0]['output_sha'],
                  checks=['canonical launcher and native S3 claims/ledger','all76 native EVIF seal checks','separate JPEG CAS and embedded bytes exact','frozen target doubles exact on readback'],runtime=result['runtime']))


def diagnostic_proof(ctl, image, out, view):
    out.mkdir();archive=out/'cas';archive.mkdir();scratch=out/'scratch';scratch.mkdir()
    case=NAS/'b/walltime/i270-r9/batch-none'
    row=next(r for r in pq.read_table(case/'ledger.parquet').to_pylist() if r['image_path'].endswith(view))
    result=json.loads(cas(case/'cas',row['output_sha']).read_text())
    raw=Path(row['image_path']).read_bytes();w,h=struct.unpack('>II',raw[16:24])
    source=dict(image_path=row['image_path'],source_sha256=hashlib.sha256(raw).hexdigest(),pixels_sha256=result['reference_pixels_sha256'],identity='frozen144-diagnostic-proof',width=w,height=h)
    for c in result['cells']:
        dest=cas(archive,c['evif_sha256']);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(cas(case/'cas',c['evif_sha256']),dest);assert sha(dest)==c['evif_sha256']
    spec=out/'spec.json';manifest=out/'manifest.json'
    save(spec,dict(sources=[dict(source=source,evif_shas=[c['evif_sha256'] for c in result['cells']])],implementation_shas=PINS,peak_mem_bytes=256*1024*1024))
    subprocess.run([str(ctl),'declare-evif-diagnostics','--spec',str(spec),'--out',str(manifest)],check=True)
    endpoint=os.environ['CHROMA_S3_ENDPOINT']
    args=['docker','run','--name','chroma-'+out.name,'--cpu-shares=256','--cpuset-cpus=0','--memory=1g','--memory-swap=1g',
          '-e','AWS_ACCESS_KEY_ID','-e','AWS_SECRET_ACCESS_KEY','-e','ZEN_R2_ENDPOINT='+endpoint,'-e','ZEN_BUCKET=zentrain','-e','ZEN_RUN=jobs/chroma-b-diagnostic-proof-20261028',
          '-e','ZEN_ARCHIVE_BLOBS_DIR='+str(archive),'-e','TMPDIR='+str(scratch),'-e','ZEN_CHUNK_WALL_SEC=0',
          '-v',str(NAS)+':'+str(NAS)+':ro','-v',str(out)+':'+str(out),'-v',str(Path.home()/'tmp/chroma-run-heavy')+':/opt/private/run-heavy:ro',
          '--entrypoint','/opt/private/run-heavy',image,'--mem','1G','--jobs','1','--','zenfleet-worker','--manifest',str(manifest),'--ledger-out',str(out/'ledger.parquet'),
          '--blobs-r2-bucket','zentrain','--blobs-r2-prefix','unused-diagnostic-proof','--r2-endpoint',endpoint,'--exec','/usr/local/bin/evif-exec','--worker','i270-chroma-diagnostic-proof','--provider','lan']
    with (out/'worker.log').open('w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=pq.read_table(out/'ledger.parquet').to_pylist();assert len(rows)==1 and rows[0]['status']=='done'
    primary=cas(archive,rows[0]['output_sha']);assert sha(primary)==rows[0]['output_sha'];actual=json.loads(primary.read_text())
    assert actual['source_sha256']==source['source_sha256']
    assert actual['reference_pixels_sha256']==source['pixels_sha256']
    expected={(r['image_path'],r['family'],r['params_json']):r for r in json.loads((NAS/'b/walltime/EXPECTED.json').read_text())}
    recipes={c['evif_sha256']:c for c in result['cells']};maps=0
    assert len(actual['cells'])==12
    for cell in actual['cells']:
        original=recipes[cell['evif_sha256']];old=expected[(source['image_path'],original['family'],original['params_json'])]
        scores=json.loads(chunks(Path(old['path']).read_bytes())['scor'])['scores']
        for metric,value in cell['scores'].items():
            want=next(s['value'] for s in scores if s['metric']==metric)
            assert struct.pack('>d',value)==struct.pack('>d',want),'diagnostic scalar differs from frozen control'
        assert struct.pack('>d',cell['teacher']['value']).hex()==original['teacher_bits']
        assert set(cell['diffmaps'])=={'butteraugli','cvvdp@standard_4k'}
        for binding in cell['diffmaps'].values():
            path=cas(archive,binding['sha256']);assert sha(path)==binding['sha256']
            fields=gzip.decompress(path.read_bytes()).split(b'\n',3)
            assert fields[:3]==[b'Pf',f'{w} {h}'.encode(),b'-1.0'] and len(fields[3])==w*h*4
            maps+=1
        assert sha(cas(archive,cell['evif_sha256']))==cell['evif_sha256']
    for digest in actual['sidecars'].values():assert sha(cas(archive,digest))==digest
    save(out/'PROOF.json',dict(passed=True,cells=12,maps=maps,scalar_doubles='all3 diagnostics exact to unbatched144 control',targets_unchanged=True,image=image,primary_sha256=rows[0]['output_sha']))


def snapshot(ledger, out, previous):
    assert not out.exists(), 'preserve earlier snapshot observation'
    rows=pq.read_table(ledger).to_pylist();assert len({r['job_id'] for r in rows})==len(rows)
    counts=Counter();failed=[]
    for row in rows:
        if row['status']!='done':failed.append(dict(job_id=row['job_id'],error_class=row['error_class']));continue
        host=row['worker'].split('-chroma-',1)[0]
        assert host in ['i270','i265','r3800x','r3500']
        counts[host]+=1
    now=datetime.now(timezone.utc)
    result=dict(observed_utc=now.isoformat(),ledger_sha256=sha(ledger),done_views=sum(counts.values()),done_cells=sum(counts.values())*76,required_views=9162,required_cells=696312,by_host_views=dict(counts),failed=failed)
    if previous:
        old=json.loads(previous.read_text());elapsed=(now-datetime.fromisoformat(old['observed_utc'])).total_seconds();assert elapsed>0
        delta={h:counts[h]-old['by_host_views'].get(h,0) for h in ['i270','i265','r3800x','r3500']}
        result.update(previous=str(previous),elapsed_seconds=elapsed,delta_views=delta)
        if all(n>=0 for n in delta.values()):result['cells_per_second_by_host']={h:n*76/elapsed for h,n in delta.items()}
        else:result['rate_not_qualified']='native first-wins attribution changed between observations'
    save(out,result);print(json.dumps(result),flush=True)


def archive_binaries(out, build_commit):
    out.mkdir()
    base=Path('/var/tmp/byc-chroma-20261028')
    files={p.name:p for p in (base/'b/bin').iterdir() if p.is_file()}
    files.update({'frozen-safesyn-extractor-7c7f':base/'bin/teacher','frozen-view-prepare':base/'bin/prepare',
                  'zenfleet-worker-r10':base/'b/fleet-target/release/zenfleet-worker','zenfleet-ctl':base/'b/fleet-target/release/zenfleet-ctl'})
    records=[]
    for name,source in sorted(files.items()):
        digest=sha(source);destination=out/name
        shutil.copyfile(source,destination);assert sha(destination)==digest
        records.append(dict(name=name,source=str(source),sha256=digest,bytes=destination.stat().st_size))
    toolchain=subprocess.check_output([str(Path.home()/'.cargo/bin/rustc'),'--version'],text=True).strip()
    save(out/'_MANIFEST.json',dict(build_commit=build_commit,toolchain=toolchain,host='i270',files=records,
                                 target_image='ghcr.io/imazen/zenfleet-worker@sha256:9ad684d702187387e1f6a4f70933598da32aa213e84917fde8eb4939d3cd0200',
                                 diagnostic_image='ghcr.io/imazen/zenfleet-worker@sha256:b04a59d67b7962bb8914232817dc546de760a92e2c3e00588a90bad56b5b03cb'))


def audit_ledgers(directory, out):
    out.mkdir();groups=defaultdict(list);statuses=Counter();files={}
    for file in sorted(directory.glob('*.parquet')):
        files[file.name]=sha(file)
        for row in pq.read_table(file).to_pylist():
            statuses[row['status']]+=1
            if row['status']=='done':groups[row['job_id']].append(row)
    assert files and groups
    repeats=[];conflicts=[]
    for key,rows in groups.items():
        outputs={r['output_sha'] for r in rows}
        identities={(r['image_path'],r['codec'],r['q'],r['knob_tuple_json']) for r in rows}
        if len(outputs)!=1 or len(identities)!=1 or None in outputs:conflicts.append(dict(job_id=key,rows=rows))
        if len(rows)>1:repeats.append(dict(job_id=key,records=len(rows),outputs=sorted(outputs,key=lambda v:'' if v is None else v),workers=sorted({r['worker'] for r in rows}),cross_host=len({r['worker'].split('-chroma-',1)[0] for r in rows})>1))
    save(out/'REPEATS.json',repeats);save(out/'CONFLICTS.json',conflicts)
    result=dict(raw_files=files,statuses=dict(statuses),unique_done_jobs=len(groups),extra_done_records=sum(r['records']-1 for r in repeats),repeated_jobs=len(repeats),cross_host_repeated_jobs=sum(r['cross_host'] for r in repeats),conflicting_jobs=len(conflicts))
    save(out/'SUMMARY.json',result)
    assert not conflicts,'contradictory outputs: inspect retained CONFLICTS.json'
    print(json.dumps({k:v for k,v in result.items() if k!='raw_files'}),flush=True)


def source_accounting(out):
    assert not out.exists()
    sources=json.loads((ROOT/'SOURCES.json').read_text());views=json.loads((ROOT/'VIEWS.json').read_text())
    represented={v['source'] for v in views};by_id={r['source']['id']:r for r in sources}
    cid={(r['width'],r['height'],r['source_pixels_sha256']):r['source'] for r in sources if r['source']['type']=='cid22'}
    aliases=[]
    for key in sorted(set(by_id)-represented):
        row=by_id[key];source=row['source'];assert source['type']=='safesyn'
        winner=cid[(row['width'],row['height'],row['source_pixels_sha256'])]
        assert winner['id'] in represented
        aliases.append(dict(alias_source=key,alias_role=source['mlic_role'],retained_source=winner['id'],retained_reference_group=winner['reference_group'],retained_role=winner['mlic_role'],pixels_sha256=row['source_pixels_sha256']))
    save(out,dict(qualified_candidates=len(sources),qualified_candidate_types=dict(Counter(r['source']['type'] for r in sources)),
                  contributing_reference_groups=len({v['reference_group'] for v in views}),contributing_sources=len(represented),
                  contributing_source_types=dict(Counter(by_id[k]['source']['type'] for k in represented)),
                  wholly_deduplicated_sources=len(aliases),alias_retained_roles=dict(Counter(r['retained_role'] for r in aliases)),aliases=aliases,
                  note='geometry SUMMARY.sources counts qualified source records before exact-pixel dedup; source roles are never promoted'))


def diagnostic_spec(admission_dir, out):
    assert not out.exists(), 'preserve existing diagnostic specification'
    views={v['id']:v for v in json.loads((ROOT/'VIEWS.json').read_text())}
    selected=defaultdict(list)
    for row in pq.read_table(admission_dir/'DIAGNOSTICS_SELECTION.parquet').to_pylist():
        selected[row['view_id']].append(row['evif_sha256'])
    sources=[]
    for key,digests in sorted(selected.items()):
        v=views[key]
        source=dict(image_path=v['path'],source_sha256=v['sha256'],pixels_sha256=v['pixels_sha256'],
                    identity=json.dumps({k:v[k] for k in ['id','reference_group','mlic_role','type','content_class']},sort_keys=True),width=v['width'],height=v['height'])
        sources.append(dict(source=source,evif_shas=sorted(set(digests))))
    save(out,dict(sources=sources,implementation_shas=PINS,peak_mem_bytes=256*1024*1024))
    print(json.dumps(dict(views=len(sources),selected=sum(len(r['evif_shas']) for r in sources))),flush=True)


def prepare_diagnostic_selection(ledger, archive, out):
    """Declare from authenticated view receipts while full media readback continues.

    This is selection only, never a substitute for full artifact admission.
    Harvest subsequently requires exact coverage of the fully verified selection.
    """
    out.mkdir();views={v['path']:v for v in json.loads((ROOT/'VIEWS.json').read_text())}
    rows=pq.read_table(ledger).to_pylist()
    assert len(rows)==len(views)==9162 and all(r['status']=='done' for r in rows)
    assert {r['image_path'] for r in rows}==set(views)
    selections=[]
    families={k:v for k,v in specs().items() if not k.startswith('band-')}
    def load(row):
        data=cas(archive,row['output_sha']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['output_sha']
        return row,json.loads(data)
    def receipts():
        ordered=sorted(rows,key=lambda r:r['image_path'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(ordered),32):
                yield from list(pool.map(load,ordered[start:start+32]))
    for index,(row,result) in enumerate(receipts()):
        view=views[row['image_path']]
        assert result['source_sha256']==view['sha256'] and result['reference_pixels_sha256']==view['pixels_sha256']
        grouped=defaultdict(list)
        for c in result['cells']:grouped[c['family']].append(c)
        assert set(grouped)==set(families)
        for family,(_,_,params) in families.items():
            cells=sorted(grouped[family],key=lambda c:c['rung']);assert len(cells)==len(params)
            gate=admission([c['teacher_ssim2'] for c in cells])
            endpoints={gate['kept_indices'][0],gate['kept_indices'][-1]} if gate['kept_indices'] else set()
            for rung,cell in enumerate(cells):
                assert cell['rung']==rung and json.loads(cell['params_json'])==params[rung]
                assert struct.pack('>d',cell['teacher_ssim2']).hex()==cell['teacher_bits']
                reason=diagnostic_reason(view['id'],family,rung,endpoints)
                if reason:selections.append(dict(view_id=view['id'],image_path=view['path'],family=family,rung=rung,
                    evif_sha256=cell['evif_sha256'],jpeg_sha256=cell['jpeg_sha256'],reason=reason,
                    content_class=view['content_class'],mlic_role=view['mlic_role']))
        if (index+1)%100==0:save(out/'PROGRESS.json',dict(stage='diagnostic selection only',views=index+1,required_views=len(rows)))
    pq.write_table(pa.Table.from_pylist(selections),out/'DIAGNOSTICS_SELECTION.parquet',compression='zstd')
    save(out/'SUMMARY.json',dict(qualified_full_grid=False,purpose='diagnostic sample declaration only; full media admission and exact selection reconciliation required',views=len(rows),selected_cells=len(selections),ledger_sha256=sha(ledger)))
    diagnostic_spec(out,out/'spec.json')


def harvest_diagnostics(ledger, admission_dir, archive, out, qualification=False):
    """Read back the declared sample; diagnostics never change target admission."""
    import numpy as np
    out.mkdir()
    selected=defaultdict(dict)
    for row in pq.read_table(admission_dir/'DIAGNOSTICS_SELECTION.parquet').to_pylist():
        key=(row['family'],row['rung'])
        assert key not in selected[row['image_path']]
        selected[row['image_path']][key]=row
    views={v['path']:v for v in json.loads((ROOT/'VIEWS.json').read_text())}
    rows=pq.read_table(ledger).to_pylist()
    if qualification:
        assert len(rows)==1 and out.name.startswith('qualification-')
        assert rows[0]['image_path'] in selected
        selected={rows[0]['image_path']:selected[rows[0]['image_path']]}
    assert len(rows)==len(selected) and all(r['status']=='done' for r in rows)
    assert {r['image_path'] for r in rows}==set(selected)
    assert len({r['job_id'] for r in rows})==len(rows)
    verified=set();verified_maps=set();counts=Counter();classes=Counter();roles=Counter();reasons=Counter();buffer=[];writer=None
    implementation='zenmetrics@0a61830db89a768ab9462f166d1412e948e4efa9;display=standard_4k;exe-sha256='+PINS[2]
    def read_object(digest):
        data=cas(archive,digest).read_bytes()
        assert hashlib.sha256(data).hexdigest()==digest
        verified.add(digest)
        return data
    try:
        for index,row in enumerate(sorted(rows,key=lambda r:r['image_path'])):
            view=views[row['image_path']];expected=selected[row['image_path']]
            result=json.loads(read_object(row['output_sha']))
            assert result['schema']=='evif-diagnostics-v1' and result['implementation']==implementation
            assert result['image_path']==row['image_path'] and result['knob_tuple_json']==row['knob_tuple_json']
            assert result['source_sha256']==view['sha256'] and result['reference_pixels_sha256']==view['pixels_sha256']
            knobs=json.loads(row['knob_tuple_json'])
            assert knobs['width']==view['width'] and knobs['height']==view['height']
            actual={r['evif_sha256']:r for r in result['cells']}
            assert len(actual)==len(result['cells'])
            assert set(actual)==set(knobs['evif_shas'])=={r['evif_sha256'] for r in expected.values()}
            assert result['sidecars']['variant-index.tsv']==result['variant_index_sha256']
            map_digests={b['sha256'] for c in actual.values() for b in c['diffmaps'].values()}
            assert map_digests<=set(result['sidecars'].values())
            for digest in set(result['sidecars'].values())-map_digests:
                if digest not in verified:read_object(digest)
            for cell in actual.values():
                parts=chunks(read_object(cell['evif_sha256']))
                assert json.loads(parts['scor'])['scores']==[cell['teacher']]
                assert hashlib.sha256(parts['BITS'][4:]).hexdigest()==cell['jpeg_sha256']
                assert hashlib.sha256(parts['PNGF']).hexdigest()==cell['png_sha256']
                assert set(cell['scores'])=={'butteraugli_max','butteraugli_pnorm3','cvvdp_cpu_imazen_v0_1_0_standard_4k'}
                assert all(math.isfinite(v) for v in cell['scores'].values())
                assert set(cell['diffmaps'])=={'butteraugli','cvvdp@standard_4k'}
                for metric,binding in cell['diffmaps'].items():
                    w,h=view['width'],view['height']
                    assert (binding['width'],binding['height'],binding['format'])==(w,h,'gzip-PFM-f32le')
                    want=hashlib.sha256((view['pixels_sha256']+'|'+cell['png_sha256']+'|'+metric+'|'+PINS[2]).encode()).hexdigest()
                    assert binding['input_binding']==want
                    map_key=(binding['sha256'],w,h)
                    if map_key not in verified_maps:
                        fields=gzip.decompress(read_object(binding['sha256'])).split(b'\n',3)
                        assert fields[:3]==[b'Pf',f'{w} {h}'.encode(),b'-1.0'] and len(fields[3])==w*h*4
                        assert np.isfinite(np.frombuffer(fields[3],dtype='<f4')).all()
                        verified_maps.add(map_key)
                    counts['map_bindings']+=1
            for identity,selection in expected.items():
                cell=actual[selection['evif_sha256']]
                assert cell['jpeg_sha256']==selection['jpeg_sha256']
                buffer.append(dict(**selection,reference_group=view['reference_group'],width=view['width'],height=view['height'],
                                   teacher_ssim2=cell['teacher']['value'],teacher_bits=struct.pack('>d',cell['teacher']['value']).hex(),
                                   png_sha256=cell['png_sha256'],**cell['scores'],
                                   butteraugli_map_sha256=cell['diffmaps']['butteraugli']['sha256'],
                                   cvvdp_map_sha256=cell['diffmaps']['cvvdp@standard_4k']['sha256'],primary_sha256=row['output_sha']))
                counts['selected_cells']+=1;classes[view['content_class']]+=1;roles[view['mlic_role']]+=1;reasons[selection['reason']]+=1
            if len(buffer)>=1000 or index+1==len(rows):
                table=pa.Table.from_pylist(buffer)
                if writer is None:writer=pq.ParquetWriter(out/'DIAGNOSTICS.parquet',table.schema,compression='zstd')
                writer.write_table(table);buffer=[]
                save(out/'PROGRESS.json',dict(stage='diagnostic readback',views=index+1,required_views=len(rows),**counts))
    finally:
        if writer:writer.close()
    assert counts['selected_cells']==sum(len(v) for v in selected.values())
    save(out/'SUMMARY.json',dict(passed=True,qualified_full_grid=not qualification,views=len(rows),**counts,unique_objects_verified=len(verified),
                                 per_class=dict(classes),per_role=dict(roles),selection_reasons=dict(reasons),
                                 ledger_sha256=sha(ledger),selection_sha256=sha(admission_dir/'DIAGNOSTICS_SELECTION.parquet'),
                                 diagnostic_implementation=implementation,targets_unchanged=True,admission_unchanged=True))
    save(out/'PROGRESS.json',dict(stage='complete',views=len(rows),**counts))


def seal_stage_b(out, build_commit, diagnostic_ledger, diagnostic_result):
    """Freeze indices of the verified CAS graph, without rewriting its blobs."""
    assert not out.exists()
    admission_summary=json.loads((ROOT/'admission-r1/SUMMARY.json').read_text())
    diagnostic_summary=json.loads((diagnostic_result/'SUMMARY.json').read_text())
    audit=json.loads((ROOT/'ledger-audit-final/SUMMARY.json').read_text())
    assert admission_summary['qualified_full_grid'] and admission_summary['cells']==696312
    assert diagnostic_summary['passed'] and diagnostic_summary['qualified_full_grid'] and diagnostic_summary['targets_unchanged']
    assert diagnostic_summary['selected_cells']==admission_summary['diagnostics_selected']
    assert set(diagnostic_summary['per_class'])==set(admission_summary['per_class'])
    assert audit['unique_done_jobs']==9162 and audit['conflicting_jobs']==0
    files=[p for p in ROOT.glob('*.json') if p.name not in ['PROGRESS.json','SNAPSHOT-A.json','SNAPSHOT-T0.json','SNAPSHOT-T1.json']]
    for directory in [ROOT/'plan',ROOT/'admission-r1',ROOT/'ledger-audit-final',diagnostic_result]:
        files.extend(p for p in directory.iterdir() if p.is_file() and p.name!='PROGRESS.json')
    files.extend([ROOT/'final-ledger/snap_chroma-b-20261028.parquet',diagnostic_ledger,NAS/'b/binaries/_MANIFEST.json'])
    for name in ['ENCODES.json','LADDERS.json','SUMMARY.json','ORACLE.json','hdrvdp3.parquet','natural-scores.jsonl','pairs.tsv']:
        files.append(NAS/'b/hdr-a3'/name)
    identities={str(p.relative_to(NAS)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))}
    save(out,dict(schema='chroma-stage-b-verified-indices-v1',build_commit=build_commit,
                  observed_utc=datetime.now(timezone.utc).isoformat(),files=identities,
                  closure='Target ledger indexes all view manifests; CELLS indexes every verified EVIF/native JPEG. Diagnostic ledger and joined Parquet index all verified diagnostic maps/raw tables/logs. Raw-ledger audit lists every contributing ledger file hash.',
                  targets=admission_summary,diagnostics=diagnostic_summary,
                  duplicate_execution=dict(unique_jobs=audit['unique_done_jobs'],extra_done_records=audit['extra_done_records'],conflicting_jobs=0),
                  stage_c='not started',stage_d='not started'))


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('spec');a.add_argument('out',type=Path);a.add_argument('--one-view',action='store_true')
    a=s.add_parser('admit');a.add_argument('ledger',type=Path);a.add_argument('archive',type=Path);a.add_argument('out',type=Path);a.add_argument('--verify-threads',type=int,default=1)
    a=s.add_parser('qualify-admission');a.add_argument('ledger',type=Path);a.add_argument('archive',type=Path);a.add_argument('out',type=Path);a.add_argument('--verify-threads',type=int,default=1)
    a=s.add_parser('verify-first');a.add_argument('ledger',type=Path);a.add_argument('archive',type=Path);a.add_argument('evif',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('diagnostic-proof');a.add_argument('ctl',type=Path);a.add_argument('image');a.add_argument('out',type=Path);a.add_argument('--view',default='im5044_m32.png')
    a=s.add_parser('snapshot');a.add_argument('ledger',type=Path);a.add_argument('out',type=Path);a.add_argument('--previous',type=Path)
    a=s.add_parser('archive-binaries');a.add_argument('out',type=Path);a.add_argument('build_commit')
    a=s.add_parser('audit-ledgers');a.add_argument('directory',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('source-accounting');a.add_argument('out',type=Path)
    a=s.add_parser('diagnostics-spec');a.add_argument('admission_dir',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('prepare-diagnostics');a.add_argument('ledger',type=Path);a.add_argument('archive',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('harvest-diagnostics');a.add_argument('ledger',type=Path);a.add_argument('admission_dir',type=Path);a.add_argument('archive',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('qualify-diagnostics');a.add_argument('ledger',type=Path);a.add_argument('admission_dir',type=Path);a.add_argument('archive',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('seal');a.add_argument('out',type=Path);a.add_argument('build_commit');a.add_argument('diagnostic_ledger',type=Path);a.add_argument('diagnostic_result',type=Path)
    a=p.parse_args()
    if a.command=='spec':specification(a.out,a.one_view)
    elif a.command=='admit':admit(a.ledger,a.archive,a.out,verify_threads=a.verify_threads)
    elif a.command=='qualify-admission':admit(a.ledger,a.archive,a.out,qualification=True,verify_threads=a.verify_threads)
    elif a.command=='verify-first':verify_first(a.ledger,a.archive,a.evif,a.out)
    elif a.command=='diagnostic-proof':diagnostic_proof(a.ctl,a.image,a.out,a.view)
    elif a.command=='snapshot':snapshot(a.ledger,a.out,a.previous)
    elif a.command=='archive-binaries':archive_binaries(a.out,a.build_commit)
    elif a.command=='audit-ledgers':audit_ledgers(a.directory,a.out)
    elif a.command=='source-accounting':source_accounting(a.out)
    elif a.command=='diagnostics-spec':diagnostic_spec(a.admission_dir,a.out)
    elif a.command=='prepare-diagnostics':prepare_diagnostic_selection(a.ledger,a.archive,a.out)
    elif a.command=='harvest-diagnostics':harvest_diagnostics(a.ledger,a.admission_dir,a.archive,a.out)
    elif a.command=='qualify-diagnostics':harvest_diagnostics(a.ledger,a.admission_dir,a.archive,a.out,qualification=True)
    else:seal_stage_b(a.out,a.build_commit,a.diagnostic_ledger,a.diagnostic_result)
