#!/usr/bin/env python3
"""Scientific Stage B specification/admission; zenfleet owns execution and retries."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
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


def admit(ledger, archive, out):
    out.mkdir()
    views = {v['path']:v for v in json.loads((ROOT/'VIEWS.json').read_text())}
    rows = pq.read_table(ledger).to_pylist()
    assert len(rows) == len(views) == 9162 and all(r['status']=='done' for r in rows)
    assert len({r['image_path'] for r in rows}) == len(views)
    families = {k:(backend,scales,params) for k,(backend,scales,params) in specs().items() if not k.startswith('band-')}
    ladders=[]; selections=[]; buffer=[]; count=0; native_verified=set(); totals=Counter(); classes=defaultdict(Counter)
    writer = None
    try:
        for i, row in enumerate(sorted(rows,key=lambda r:r['image_path'])):
            view=views[row['image_path']]
            manifest_path=cas(archive,row['output_sha']); assert sha(manifest_path)==row['output_sha']
            manifest=json.loads(manifest_path.read_text())
            assert manifest['source_sha256']==view['sha256'] and manifest['reference_pixels_sha256']==view['pixels_sha256']
            assert manifest['image_path']==row['image_path'] and manifest['knob_tuple_json']==row['knob_tuple_json']
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
                classes[view['content_class']][family+'_admitted']+=bool(gate['kept_indices'])
                classes[view['content_class']][family+'_kept']+=len(gate['kept_indices'])
                ladders.append(dict(view_id=view['id'],source=view['source'],reference_group=view['reference_group'],
                                    mlic_role=view['mlic_role'],content_class=view['content_class'],family=family,
                                    scores=values,kept_indices=gate['kept_indices'],candidate_indices=gate['candidate_indices'],
                                    stop_index=gate['stop_index'],admitted=bool(gate['kept_indices'])))
                for rung,cell in enumerate(cells):
                    assert cell['rung']==rung and json.loads(cell['params_json'])==params[rung]
                    assert cell['teacher_bits']==struct.pack('>d',cell['teacher_ssim2']).hex()
                    assert cell['reference_pixels_sha256']==view['pixels_sha256']
                    path=cas(archive,cell['evif_sha256']);data=path.read_bytes()
                    assert hashlib.sha256(data).hexdigest()==cell['evif_sha256']
                    parts=chunks(data);jpeg=parts['BITS'][4:]
                    assert hashlib.sha256(jpeg).hexdigest()==cell['jpeg_sha256']
                    if cell['jpeg_sha256'] not in native_verified:
                        assert cas(archive,cell['jpeg_sha256']).read_bytes()==jpeg
                        native_verified.add(cell['jpeg_sha256'])
                    target=json.loads(parts['scor'])['scores'];assert len(target)==1 and target[0]['metric']=='ssim2'
                    assert struct.pack('>d',target[0]['value']).hex()==cell['teacher_bits']
                    sample=int(hashlib.sha256((view['id']+'|'+family+'|'+str(rung)).encode()).hexdigest(),16)%20==0
                    if sample or rung in endpoints:
                        selected.append(cell['evif_sha256'])
                        selections.append(dict(view_id=view['id'],image_path=view['path'],family=family,rung=rung,
                                               evif_sha256=cell['evif_sha256'],jpeg_sha256=cell['jpeg_sha256'],
                                               reason='endpoint+sample' if sample and rung in endpoints else 'endpoint' if rung in endpoints else 'sample',
                                               content_class=view['content_class'],mlic_role=view['mlic_role']))
                    buffer.append(dict(**cell,view_id=view['id'],source=view['source'],image_path=view['path'],
                                       reference_group=view['reference_group'],mlic_role=view['mlic_role'],
                                       content_class=view['content_class'],source_type=view['type'],width=view['width'],height=view['height'],
                                       kept=rung in gate['kept_indices'],diagnostic_selected=sample or rung in endpoints))
                    count+=1
            if len(buffer)>=7600 or i+1==len(rows):
                table=pa.Table.from_pylist(buffer)
                if writer is None:writer=pq.ParquetWriter(out/'CELLS.parquet',table.schema,compression='zstd')
                writer.write_table(table);buffer=[]
                save(out/'PROGRESS.json',dict(views=i+1,required_views=len(rows),cells=count,stage='hash verification and admission'))
    finally:
        if writer:writer.close()
    assert count==696312
    pq.write_table(pa.Table.from_pylist(ladders),out/'LADDERS.parquet',compression='zstd')
    pq.write_table(pa.Table.from_pylist(selections),out/'DIAGNOSTICS_SELECTION.parquet',compression='zstd')
    save(out/'SUMMARY.json',dict(cells=count,views=len(rows),ladders=len(ladders),totals=dict(totals),
                                 per_class={k:dict(v) for k,v in classes.items()},native_objects_verified=len(native_verified),
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


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('spec');a.add_argument('out',type=Path);a.add_argument('--one-view',action='store_true')
    a=s.add_parser('admit');a.add_argument('ledger',type=Path);a.add_argument('archive',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('verify-first');a.add_argument('ledger',type=Path);a.add_argument('archive',type=Path);a.add_argument('evif',type=Path);a.add_argument('out',type=Path)
    a=s.add_parser('diagnostics-spec');a.add_argument('admission_dir',type=Path);a.add_argument('out',type=Path)
    a=p.parse_args()
    if a.command=='spec':specification(a.out,a.one_view)
    elif a.command=='admit':admit(a.ledger,a.archive,a.out)
    elif a.command=='verify-first':verify_first(a.ledger,a.archive,a.evif,a.out)
    else:diagnostic_spec(a.admission_dir,a.out)
