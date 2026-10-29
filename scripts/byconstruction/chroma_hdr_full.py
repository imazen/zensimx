#!/usr/bin/env python3
"""HDR chroma scientific inputs and readback; zenfleet owns all job execution.

Only committed canonical manifests select sources. Native kernels own pixels;
all roles and links survive the transport. No feature extraction or training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request
from chroma_pilot import admission, save, sha

DELTAS = [0, 8, 16, 24, 32, 40, 48, 56, 63]
QUALITIES = [95, 85, 75, 60]
CONTROLS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(x).hexdigest()


def census(repo, out):
    # Repository HEAD is frozen once; no working-copy manifest is opened.
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    revision = git('rev-parse', 'HEAD').decode().strip()
    def committed(path):
        return git('show', revision+':'+path)
    train = {r['id']: r for r in csv.DictReader(io.StringIO(committed('manifests/train.tsv').decode()), delimiter='\t')}
    split = {r['id']: r for r in csv.DictReader(io.StringIO(committed('manifests/split_map.tsv').decode()), delimiter='\t')}
    paths = git('ls-tree', '-r', '--name-only', revision, 'png-v3').decode().splitlines()
    hdr = [p for p in paths if p.endswith('.hdr.png')]
    sources = []
    for path in hdr:
        identity = Path(path).name[:4]
        if identity not in train:
            continue
        if split[identity]['split'] != 'train' or int(identity)%10 not in [0, 2, 4, 6, 8]:
            raise ValueError('committed canonical split disagreement')
        pointer = committed(path).decode()
        if not pointer.startswith('version https://git-lfs.github.com/spec/v1\n'):
            raise ValueError('committed HDR source must carry its LFS identity')
        source_sha = pointer.split('oid sha256:')[1].split()[0]
        sources.append(dict(id=identity, path=path, source_sha256=source_sha,
                            bytes=int(pointer.split('size ')[1]), split='train',
                            content_class=train[identity]['content_class'], original=train[identity],
                            url='https://codec-corpus.r2.imazen.org/imazen-26-png-v3/'+path.removeprefix('png-v3/')))
    pins = {p:digest(committed(p)) for p in ['CORPUS-MANIFEST.tsv', 'manifests/train.tsv', 'manifests/split_map.tsv', 'VARIANT.md']}
    save(out, dict(schema='hdr-chroma-source-census-v1', corpus_commit=revision,
                   total_hdr_origins=len(hdr), train_hdr_origins=len(sources),
                   classes=dict(Counter(s['content_class'] for s in sources)),
                   source_bytes=sum(s['bytes'] for s in sources), manifest_sha256=pins, sources=sources))


def prepare(root, renderer):
    source_spec = json.loads((root/'SOURCE_CENSUS.json').read_text())
    for directory in ['raw', 'views', 'receipts', 'logs']:
        (root/directory).mkdir(exist_ok=True, parents=True)
    views = []
    for index, source in enumerate(source_spec['sources']):
        raw = root/'raw'/(source['id']+'.hdr.png')
        if not raw.exists():
            print('fetch', source['id'], source['url'], flush=True)
            with urllib.request.urlopen(urllib.request.Request(source['url'], headers={'User-Agent': 'chroma-hdr/1.0'}), timeout=180) as response, raw.open('xb') as out:
                shutil.copyfileobj(response, out, length=1024*1024)
        if sha(raw) != source['source_sha256'] or raw.stat().st_size != source['bytes']:
            raise ValueError('source differs from committed LFS identity: '+source['id'])
        dest = root/'views'/source['id']
        if not (dest/'views.json').exists():
            if dest.exists():
                raise ValueError('incomplete native preparation retained: '+str(dest))
            with (root/'logs'/('prepare-'+source['id']+'.log')).open('x') as log:
                subprocess.run([str(renderer), 'prepare', str(raw), str(dest)], stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = json.loads((dest/'views.json').read_text())
        for item in receipt:
            path = dest/item['file']
            if sha(path) != item['sha256'] or item['width']*item['height'] >= 1_000_000:
                raise ValueError('prepared-view identity or dimensions differ')
            views.append(dict(**item, id=source['id']+'_'+path.stem, source=source['id'],
                              content_class=source['content_class'], split='train',
                              path=str(path), source_sha256=source['source_sha256']))
        save(root/'PROGRESS.json', dict(stage='prepare', pid=os.getpid(), completed_sources=index+1,
                                       required_sources=len(source_spec['sources']), views=len(views)))
    # Dedup only within a source; every origin remains represented.
    kept, duplicates, seen = [], [], set()
    for view in views:
        key = (view['source'], view['width'], view['height'], view['pixels_sha256'])
        if key in seen:
            duplicates.append(view)
        else:
            seen.add(key)
            kept.append(view)
    tiny = sum(max(v['width'], v['height']) <= 64 for v in kept)
    if tiny*2 >= len(kept):
        raise ValueError('tiny HDR views must remain a minority')
    save(root/'VIEWS.json', kept)
    save(root/'DUPLICATE_VIEWS.json', duplicates)
    save(root/'GEOMETRY.json', dict(sources=len(source_spec['sources']), views=len(kept), tiny_views=tiny,
                                  native_crops=sum(v['kind']=='native-crop' for v in kept),
                                  max_pixels=max(v['width']*v['height'] for v in kept),
                                  duplicates=len(duplicates), renderer_sha256=sha(renderer),
                                  classes=dict(Counter(v['content_class'] for v in kept))))
    print((root/'GEOMETRY.json').read_text(), flush=True)


def mirror_inputs(root, mirror):
    """Read back every committed source/view identity before shared declarations."""
    mirror.mkdir(parents=True,exist_ok=True)
    records=[]
    paths=[root/name for name in ['SOURCE_CENSUS.json','VIEWS.json','GEOMETRY.json','DUPLICATE_VIEWS.json']]
    paths+=sorted((root/'raw').glob('*.png'))
    paths+=sorted((root/'views').glob('*/*'))
    for index,path in enumerate(paths):
        relative=path.relative_to(root); destination=mirror/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        expected=sha(path)
        if not destination.exists():shutil.copyfile(path,destination)
        if sha(destination)!=expected:raise ValueError('mirror bytes differ: '+str(relative))
        records.append(dict(path=str(relative),sha256=expected,bytes=path.stat().st_size))
        if index%50==0:print('mirrored',index+1,'/',len(paths),flush=True)
    receipt=dict(passed=True,files=len(records),bytes=sum(r['bytes'] for r in records),records=records)
    save(root/'INPUT_MIRROR.json',receipt);shutil.copyfile(root/'INPUT_MIRROR.json',mirror/'INPUT_MIRROR.json')
    print(canonical(dict(passed=True,files=receipt['files'],bytes=receipt['bytes'])),flush=True)


def specification(root, mirror, pins, executor, out):
    views = json.loads((root/'VIEWS.json').read_text())
    binaries = json.loads(pins.read_text())
    if set(binaries) != {'chroma-hdr-render', 'chroma-natural-hdr'}:
        raise ValueError('exact renderer/teacher binary pins required')
    # Freeze a comparison before declaration: first origin per class, m256/m512.
    first = {}
    for v in sorted(views, key=lambda x:x['id']):
        first.setdefault(v['content_class'], v['source'])
    selections = []
    jobs = []
    for v in views:
        control = v['source'] == first[v['content_class']] and v['file'] == 'm512.png'
        compare = v['source'] == first[v['content_class']] and v['file'] in ['m256.png', 'm512.png']
        if compare:
            selections.append(dict(view_id=v['id'], control=control, content_class=v['content_class']))
        path = mirror/Path(v['path']).relative_to(root)
        if sha(path) != v['sha256']:
            raise ValueError('mirror source bytes differ before declaration')
        identity = canonical({k:v[k] for k in ['id', 'source', 'content_class', 'split', 'source_sha256', 'pixels_sha256', 'kind']})
        knobs = dict(control=control, width=v['width'], height=v['height'], source_sha256=v['sha256'],
                     source_identity=identity, binaries=binaries, executor_sha256=sha(executor))
        inputs = [v['sha256'], digest(canonical(knobs).encode()), digest(identity.encode()), sha(executor), *binaries.values()]
        jobs.append(dict(view='hdr-chroma-view-v1',
                         cell=dict(image_path=str(path), codec='hdr-chroma:avif-svt', q=-1, knob_tuple_json=canonical(knobs)),
                         inputs=inputs, hint=dict(peak_mem_bytes=536870912, threads=1, vram_bytes=0),
                         requires=['hdr-chroma-view-v1', 'ssim2-natural-hdr-b7c2b4b3']))
    save(out, dict(schema=1, views=jobs))
    save(root/'COMPARISON_SELECTION.json', selections)
    save(root/'PLAN.json', dict(views=len(views), chroma_cells=36*len(views),
                               control_extra_cells=16*sum(j['cell']['knob_tuple_json'].find('"control":true')>=0 for j in jobs),
                               comparison_views=len(selections), sample_rule='first committed TRAIN origin per class, Mitchell long edges 256 and 512; luma control only at 512',
                               binaries=binaries, executor_sha256=sha(executor), features=False, training=False))
    print((root/'PLAN.json').read_text(), flush=True)


def cas(archive, value):
    return archive/value[:2]/value[2:4]/value


def png_contract(data, width, height, primaries):
    import zlib
    if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('native PNG signature differs')
    offset=8;chunks={}
    while offset<len(data):
        size=struct.unpack('>I',data[offset:offset+4])[0];tag=data[offset+4:offset+8]
        body=data[offset+8:offset+8+size]
        if offset+12+size>len(data):raise ValueError('truncated native PNG')
        crc=struct.unpack('>I',data[offset+8+size:offset+12+size])[0]
        if zlib.crc32(tag+body)&0xffffffff!=crc:raise ValueError('native PNG CRC differs')
        if tag!=b'IDAT':
            if tag in chunks:raise ValueError('duplicate native PNG metadata')
            chunks[tag]=body
        offset+=12+size
    if offset!=len(data) or b'IEND' not in chunks:raise ValueError('native PNG framing differs')
    if struct.unpack('>II',chunks[b'IHDR'][:8])!=(width,height) or chunks[b'IHDR'][8:10]!=bytes([16,2]):raise ValueError('native PNG geometry/precision differs')
    if chunks.get(b'cICP')!=bytes([primaries,16,0,1]):raise ValueError('native PQ signaling differs')
    if set(chunks)&{b'iCCP',b'sRGB',b'gAMA',b'cHRM'}:raise ValueError('ambiguous native HDR color metadata')


def admit(root, mirror, ledger, out, qualification=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir()
    archive=mirror/'cas'
    views={str(mirror/Path(v['path']).relative_to(root)):v for v in json.loads((root/'VIEWS.json').read_text())}
    rows=pq.read_table(ledger).to_pylist()
    if any(r['status']!='done' for r in rows):raise ValueError('admission requires an all-DONE reconciled ledger')
    if len({r['image_path'] for r in rows})!=len(rows):raise ValueError('duplicate view ledger rows')
    if not qualification and set(r['image_path'] for r in rows)!=set(views):raise ValueError('full admission requires every frozen view')
    if qualification and len(rows)!=16:raise ValueError('host-sample qualification requires its sixteen views')
    plan=json.loads((root/'PLAN.json').read_text());selection={r['view_id']:r for r in json.loads((root/'COMPARISON_SELECTION.json').read_text())}
    cells_out=[];ladders=[];classes=defaultdict(Counter);totals=Counter();outputs={};seen_native={}
    def check_cell(item):
        cell,view=item
        for key in ['encoded_sha256','rendition_sha256']:
            data=cas(archive,cell[key]).read_bytes()
            if digest(data)!=cell[key]:raise ValueError('CAS identity differs: '+cell[key])
            if key=='rendition_sha256':png_contract(data,view['width'],view['height'],9)
        if struct.pack('>d',cell['natural_ssim2']).hex()!=cell['value_bits']:raise ValueError('score double bits differ')
        if cell['cicp']!=[9,16,6,True] or cell['bit_depth']!=10:raise ValueError('AVIF declared coding differs')
        if cell['implementation']!='fast-ssim2@b7c2b4b33f3a26b753caabd5264a7fef9c4f4545;natural-cube-root;linear-BT709;nits/100;unclipped;E26-common-ingress@348bde5b':raise ValueError('natural target identity differs')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,row in enumerate(sorted(rows,key=lambda r:r['image_path'])):
            view=views[row['image_path']];receipt_path=cas(archive,row['output_sha'])
            if sha(receipt_path)!=row['output_sha']:raise ValueError('receipt hash differs')
            receipt=json.loads(receipt_path.read_text())
            if receipt['image_path']!=row['image_path'] or receipt['knob_tuple_json']!=row['knob_tuple_json'] or receipt['source_sha256']!=view['sha256']:raise ValueError('receipt source binding differs')
            knobs=json.loads(row['knob_tuple_json'])
            if receipt['binaries']!=plan['binaries'] or receipt['executor_sha256']!=plan['executor_sha256'] or knobs['binaries']!=plan['binaries'] or knobs['executor_sha256']!=plan['executor_sha256']:raise ValueError('runtime implementation differs')
            if receipt['source_identity']!=knobs['source_identity']:raise ValueError('source provenance binding differs')
            identity=json.loads(receipt['source_identity'])
            if identity['split']!='train' or identity['source']!=view['source'] or identity['content_class']!=view['content_class'] or identity['pixels_sha256']!=view['pixels_sha256']:raise ValueError('source role/class/pixels differs')
            source=cas(archive,view['sha256'])
            if sha(source)!=view['sha256']:raise ValueError('prepared source CAS missing')
            expected={(q,d) for q in QUALITIES for d in DELTAS}
            if receipt['control']:expected.update((q,0) for q in CONTROLS)
            recipe={(c['quality'],c['chroma_q'][0]):c for c in receipt['cells']}
            if set(recipe)!=expected or len(recipe)!=len(receipt['cells']):raise ValueError('view recipe coverage differs')
            list(pool.map(check_cell,((c,view) for c in receipt['cells'])))
            chroma_kept=set();control_kept=set()
            groups=[('chroma-q'+str(q),[(q,d) for d in DELTAS]) for q in QUALITIES]
            if receipt['control']:groups.append(('luma-control',[(q,0) for q in CONTROLS]))
            for family,keys in groups:
                values=[recipe[k]['natural_ssim2'] for k in keys];gate=admission(values)
                totals[family+'_ladders']+=1;totals[family+'_admitted']+=bool(gate['kept_indices']);totals[family+'_kept']+=len(gate['kept_indices'])
                classes[view['content_class']][family+'_ladders']+=1;classes[view['content_class']][family+'_admitted']+=bool(gate['kept_indices']);classes[view['content_class']][family+'_kept']+=len(gate['kept_indices'])
                for i in gate['kept_indices']:(control_kept if family=='luma-control' else chroma_kept).add(keys[i])
                ladders.append(dict(view_id=view['id'],source=view['source'],content_class=view['content_class'],split='train',family=family,scores=values,**gate))
            for cell in receipt['cells']:
                key=(cell['quality'],cell['chroma_q'][0]);native=cell['encoded_sha256'];rendition=cell['rendition_sha256']
                if native in seen_native and seen_native[native]!=rendition:raise ValueError('one AVIF maps to different PQ renditions')
                seen_native[native]=rendition
                cells_out.append(dict(**cell,view_id=view['id'],source=view['source'],source_sha256=view['source_sha256'],reference_sha256=view['sha256'],reference_pixels_sha256=view['pixels_sha256'],content_class=view['content_class'],split='train',width=view['width'],height=view['height'],kind=view['kind'],receipt_sha256=row['output_sha'],kept_chroma=key in chroma_kept,kept_control=key in control_kept,comparison_selected=view['id'] in selection))
            outputs[row['job_id']]=row['output_sha']
            save(out/'PROGRESS.json',dict(stage='media readback and admission',views=index+1,required_views=len(rows),cells=len(cells_out)))
    # Arrow lists are homogeneous: preserve signaling as typed columns rather
    # than coercing the bool full-range flag into the integer CICP triplet.
    for cell in cells_out:
        cp,tc,mc,full=cell.pop('cicp')
        cell.update(color_primaries=cp,transfer_characteristics=tc,matrix_coefficients=mc,full_range=full)
    for filename,data in [('CELLS.parquet',cells_out),('LADDERS.parquet',ladders)]:pq.write_table(pa.Table.from_pylist(data),out/filename,compression='zstd')
    actual=pq.read_table(out/'CELLS.parquet',columns=['natural_ssim2','value_bits'])
    if actual.schema.field('natural_ssim2').type!=pa.float64():raise ValueError('Parquet must preserve f64 targets')
    for value,bits in zip(actual.column(0).to_pylist(),actual.column(1).to_pylist()):
        if struct.pack('>d',value).hex()!=bits:raise ValueError('Parquet target bits differ')
    if not qualification and len(cells_out)!=plan['chroma_cells']+plan['control_extra_cells']:raise ValueError('complete cell census differs')
    # Human-inspectable adjacent pairs are hard links to the authenticated CAS.
    pairdir=mirror/'artifacts/zenavif';pairdir.mkdir(parents=True,exist_ok=True)
    def link_pair(item):
        native,rendition=item
        for filename,value in [(native+'.avif',native),(native+'.pq.png',rendition)]:
            dest=pairdir/filename
            if not dest.exists():os.link(cas(archive,value),dest)
            if sha(dest)!=value:raise ValueError('adjacent artifact pair differs')
    with ThreadPoolExecutor(max_workers=8) as links:
        list(links.map(link_pair,seen_native.items()))
    result=dict(passed=True,qualified_full_grid=not qualification,views=len(rows),cells=len(cells_out),unique_avif=len(seen_native),ladders=len(ladders),totals=dict(totals),per_class={k:dict(v) for k,v in classes.items()},outputs=outputs,ledger_sha256=sha(ledger),features=False,training=False)
    save(out/'SUMMARY.json',result);save(out/'PROGRESS.json',dict(stage='complete',views=len(rows),cells=len(cells_out)))
    print(canonical({k:v for k,v in result.items() if k not in ['outputs','per_class']}),flush=True)


def declare_production(root,mirror,ctl,build_commit,image):
    rates=[json.loads((mirror/'rates'/host/'RATE.json').read_text()) for host in ['i270','i265','r3500','r3800x']]
    if any(not r['passed'] or r['cells']!=640 or r['outputs']!=rates[0]['outputs'] for r in rates):raise ValueError('all host byte/score receipts must match before production')
    admission_proof=json.loads((mirror/'qualification/admission/SUMMARY.json').read_text())
    if not admission_proof['passed'] or admission_proof['cells']!=640 or admission_proof['qualified_full_grid']:raise ValueError('sample admission proof missing or mislabeled')
    replay=json.loads((mirror/'qualification/A3_REPLAY.json').read_text())
    if not replay['passed'] or [replay[k] for k in ['teacher_exact','encoded_byte_exact','rendered_score_exact']]!=[208]*3:raise ValueError('all A3 golden replays required')
    out=mirror/'plan';out.mkdir()
    shutil.copyfile(root/'SPEC.json',out/'SPEC.json')
    subprocess.run([str(ctl),'declare-views','--spec',str(out/'SPEC.json'),'--out',str(out/'manifest.json')],check=True)
    # Reuse the native verified sixteen-job ledger; never manufacture DONE rows.
    shutil.copyfile(mirror/'rates/i270/ledger.parquet',out/'seed-ledger.parquet')
    subprocess.run([str(ctl),'gap','--manifest',str(out/'manifest.json'),'--ledger',str(out/'seed-ledger.parquet'),'--out',str(out/'initial-gap.json')],check=True)
    manifest=json.loads((out/'manifest.json').read_text());gap=json.loads((out/'initial-gap.json').read_text())
    if len(manifest)!=456 or len(gap)!=440:raise ValueError('frozen declaration or reused sample census differs')
    record=dict(passed=True,build_commit=build_commit,image=image,views=len(manifest),reused_sample_views=16,new_views=len(gap),manifest_sha256=sha(out/'manifest.json'),seed_ledger_sha256=sha(out/'seed-ledger.parquet'),spec_sha256=sha(out/'SPEC.json'),rates=[{k:v for k,v in r.items() if k!='outputs'} for r in rates],all_host_outputs_exact=True,admission_unchanged=True,features=False,training=False)
    save(out/'DECLARATION.json',record);print(canonical(record),flush=True)


def sample_spec(root, mirror, ctl):
    spec=json.loads((root/'SPEC.json').read_text())
    first={}
    for item in spec['views']:
        identity=json.loads(json.loads(item['cell']['knob_tuple_json'])['source_identity'])
        first.setdefault(identity['content_class'],identity['source'])
    selected=[]
    for item in spec['views']:
        identity=json.loads(json.loads(item['cell']['knob_tuple_json'])['source_identity'])
        if identity['source']==first[identity['content_class']] and Path(item['cell']['image_path']).name in ['m64.png','m256.png','m512.png','m960.png']:
            selected.append(item)
    if len(selected)!=16:raise ValueError('host sample must cover four sizes and all four HDR classes')
    out=mirror/'qualification';out.mkdir()
    save(out/'host-sample.spec.json',dict(schema=1,views=selected))
    subprocess.run([str(ctl),'declare-views','--spec',str(out/'host-sample.spec.json'),'--out',str(out/'host-sample.manifest.json')],check=True)
    save(out/'HOST_SAMPLE.json',dict(views=len(selected),cells=640,classes=first,long_edges=[64,256,512,960],selection='first TRAIN origin per class; identical declaration on every host; diagnostic luma control only at512'))


def host_sample(root, mirror, image, host, cpuset, memory):
    import time
    out=root/'rates'/host;out.mkdir(parents=True)
    scratch=out/'scratch';scratch.mkdir()
    archive=mirror/'cas';archive.mkdir(exist_ok=True)
    wrapper=Path.home()/'tmp/chroma-run-heavy'
    manifest=mirror/'qualification/host-sample.manifest.json'
    args=['docker','run','--name','chromahdr-host-sample-'+host,'--network','none','--cpuset-cpus',cpuset,'--cpu-shares','2048','--memory',memory,'--memory-swap',memory,'-e','TMPDIR=/scratch','-e','ZEN_CHUNK_WALL_SEC=0','-e','ZEN_CORE_OVERSUBSCRIBE=1','-e','ZEN_ARCHIVE_BLOBS_DIR='+str(archive),'-v',str(mirror)+':'+str(mirror),'-v',str(out)+':/output','-v',str(scratch)+':/scratch','-v',str(wrapper)+':/opt/private/run-heavy:ro','--entrypoint','/opt/private/run-heavy',image,'--mem',memory,'--jobs','1','--','/usr/local/bin/zenfleet-worker','--manifest',str(manifest),'--ledger-out','/output/ledger.parquet','--blobs-r2-bucket','zentrain','--r2-endpoint',os.environ['EP'],'--exec','/usr/local/bin/hdr-chroma-exec','--worker',host+'-chromahdr-sample','--provider','lan']
    start=time.monotonic()
    with (out/'worker.log').open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
    seconds=time.monotonic()-start
    code="import pyarrow.parquet as pq,json;print(json.dumps(pq.read_table('/output/ledger.parquet').to_pylist()))"
    raw=subprocess.check_output(['docker','run','--rm','--network','none','--memory','512m','--memory-swap','512m','--cpus','1','-v',str(out)+':/output:ro','--entrypoint','python3',image,'-c',code])
    rows=json.loads(raw)
    if len(rows)!=16 or any(r['status']!='done' for r in rows):raise ValueError('host sample has incomplete/failed jobs')
    count=0
    for row in rows:
        path=archive/row['output_sha'][:2]/row['output_sha'][2:4]/row['output_sha']
        if sha(path)!=row['output_sha']:raise ValueError('sample receipt hash differs')
        receipt=json.loads(path.read_text());count+=len(receipt['cells'])
        for cell in receipt['cells']:
            for key in ['encoded_sha256','rendition_sha256']:
                value=cell[key];object_path=archive/value[:2]/value[2:4]/value
                if sha(object_path)!=value:raise ValueError('sample artifact readback differs')
    if count!=640:raise ValueError('sample cell count differs')
    result=dict(passed=True,host=host,views=len(rows),cells=count,seconds=seconds,cells_per_second=count/seconds,image=image,cpuset=cpuset,memory=memory,shares=2048,manifest_sha256=sha(manifest),outputs={r['job_id']:r['output_sha'] for r in rows})
    save(out/'RATE.json',result)
    mirror_out=mirror/'rates'/host;mirror_out.mkdir(parents=True)
    for path in out.iterdir():
        if path.is_file():shutil.copyfile(path,mirror_out/path.name)
    print(canonical({k:v for k,v in result.items() if k!='outputs'}),flush=True)


def smoke(base, bins, executor, image, out):
    """Real declared one-view job, persisted through the native worker transport."""
    import time
    import pyarrow.parquet as pq
    out.mkdir()
    reference=Path(json.loads((base/'b/hdr-a3/ENCODES.json').read_text())[0]['ref_path'])
    raw=reference.read_bytes(); width,height=struct.unpack('>II',raw[16:24])
    binaries={name:sha(bins/name) for name in ['chroma-hdr-render','chroma-natural-hdr']}
    identity=canonical(dict(id='A3-qualification',split='train',source='1066'))
    knobs=dict(control=True,width=width,height=height,source_sha256=digest(raw),source_identity=identity,binaries=binaries,executor_sha256=sha(executor))
    spec=dict(schema=1,views=[dict(view='hdr-chroma-view-v1',cell=dict(image_path=str(reference),codec='hdr-chroma:avif-svt',q=-1,knob_tuple_json=canonical(knobs)),inputs=[digest(raw),digest(canonical(knobs).encode()),digest(identity.encode()),sha(executor),*binaries.values()],hint=dict(peak_mem_bytes=536870912,threads=1,vram_bytes=0),requires=['hdr-chroma-view-v1','ssim2-natural-hdr-b7c2b4b3'])])
    save(out/'spec.json',spec)
    subprocess.run([str(bins/'zenfleet-ctl'),'declare-views','--spec',str(out/'spec.json'),'--out',str(out/'manifest.json')],check=True)
    scratch=out/'scratch';scratch.mkdir()
    wrapper=Path.home()/'tmp/chroma-run-heavy'
    args=['docker','run','--name','chromahdr-'+out.name,'--network','none','--cpuset-cpus','0','--cpu-shares','2048','--memory','2g','--memory-swap','2g','-e','TMPDIR=/scratch','-e','ZEN_CHUNK_WALL_SEC=0','-e','RAYON_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-v',str(base)+':'+str(base)+':ro','-v',str(out)+':/output','-v',str(scratch)+':/scratch','-v',str(wrapper)+':/opt/private/run-heavy:ro','--entrypoint','/opt/private/run-heavy',image,'--mem','2G','--jobs','1','--','/usr/local/bin/zenfleet-worker','--manifest','/output/manifest.json','--ledger-out','/output/ledger.parquet','--blobs','/output/blobs','--exec','/usr/local/bin/hdr-chroma-exec','--worker','i270-chromahdr-proof','--provider','lan']
    start=time.monotonic()
    with (out/'worker.log').open('x') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    elapsed=time.monotonic()-start
    rows=pq.read_table(out/'ledger.parquet').to_pylist()
    if len(rows)!=1 or rows[0]['status']!='done':raise ValueError('real HDR worker failed: inspect ledger/log')
    receipt=out/'blobs'/rows[0]['output_sha']
    if sha(receipt)!=rows[0]['output_sha']:raise ValueError('receipt CAS hash differs')
    manifest=json.loads(receipt.read_text())
    if manifest['source_sha256']!=digest(raw) or manifest['knob_tuple_json']!=canonical(knobs):raise ValueError('worker reference identity differs')
    golden=json.loads((base/'b/hdr-a3/ENCODES.json').read_text())
    targets=[json.loads(line) for line in (base/'b/hdr-a3/natural-scores.jsonl').read_text().splitlines()]
    byrecipe={(int(r['quality']),r['chroma_q'][0]):r for r in golden if r['ref_path']==str(reference)}
    for cell in manifest['cells']:
        old=byrecipe[(cell['quality'],cell['chroma_q'][0])]
        if cell['encoded_sha256']!=old['encode_sha256']:raise ValueError('worker AVIF differs from A3')
        expected=next(r['value_bits'] for r in targets if r['reference']==str(reference) and r['distorted']==old['dist_path'])
        if cell['value_bits']!=expected:raise ValueError('worker score bits differ from A3')
        for key in ['encoded_sha256','rendition_sha256']:
            if sha(out/'blobs'/cell[key])!=cell[key]:raise ValueError('worker native/rendition CAS missing or corrupt')
    result=dict(passed=True,cells=len(manifest['cells']),jobs=1,seconds=elapsed,image=image,output_sha256=rows[0]['output_sha'],binaries=binaries,executor_sha256=sha(executor),checks=['real declare-views','DONE only after native AVIF and PQ16 CAS readback','all AVIF hashes match HDR-A3','all target bits match HDR-A3'])
    save(out/'PROOF.json',result);print(canonical(result),flush=True)


def replay(base, bins, out):
    """Exact A3 doubles and native bitstreams; no tolerance or target changes."""
    import time
    out.mkdir()
    golden = base/'b/hdr-a3'
    expected = [json.loads(line) for line in (golden/'natural-scores.jsonl').read_text().splitlines()]
    pairs = (golden/'natural-pairs.jsonl').read_bytes()
    start = time.monotonic()
    with (out/'natural.jsonl').open('xb') as f, (out/'teacher.stderr').open('xb') as err:
        subprocess.run([str(bins/'chroma-natural-hdr')], input=pairs, stdout=f, stderr=err, check=True)
    rows = [json.loads(line) for line in (out/'natural.jsonl').read_text().splitlines()]
    if len(rows) != len(expected):
        raise ValueError('A3 replay row count differs')
    for a, b in zip(rows, expected):
        if (a['reference'],a['distorted'],a['value_bits']) != (b['reference'],b['distorted'],b['value_bits']):
            raise ValueError('A3 teacher bits or pair differs')
    records = json.loads((golden/'ENCODES.json').read_text())
    actual = []
    for source in sorted({r['source'] for r in records}):
        old = [r for r in records if r['source']==source]
        ref = Path(old[0]['ref_path'])
        dest = out/source
        with (out/(source+'.render.stderr')).open('xb') as err:
            subprocess.run([str(bins/'chroma-hdr-render'), 'a3', str(ref), str(dest)], stdout=err, stderr=subprocess.STDOUT, check=True)
        lookup = {(int(r['quality']),r['chroma_q'][0]):r for r in old}
        for r in json.loads((dest/'encodes.json').read_text()):
            key = (int(r['quality']),r['chroma_q'][0]); expected_row=lookup[key]
            encoded = dest/r['file']; candidate = encoded.with_suffix('.pq.png')
            if sha(encoded) != expected_row['encode_sha256']:
                raise ValueError('A3 AVIF byte parity differs: '+str(encoded))
            actual.append(dict(reference=str(ref), distorted=str(candidate), expected_bits=next(x['value_bits'] for x in expected if x['reference']==str(ref) and x['distorted']==expected_row['dist_path']), encoded_sha256=sha(encoded), rendition_sha256=sha(candidate)))
    input_bytes = ''.join(canonical({k:r[k] for k in ['reference','distorted']})+'\n' for r in actual).encode()
    with (out/'rendered-scores.jsonl').open('xb') as f, (out/'rendered-teacher.stderr').open('xb') as err:
        subprocess.run([str(bins/'chroma-natural-hdr')], input=input_bytes, stdout=f, stderr=err, check=True)
    scored = [json.loads(line) for line in (out/'rendered-scores.jsonl').read_text().splitlines()]
    if len(scored) != len(actual):
        raise ValueError('render replay identities lost')
    for a,b in zip(actual,scored):
        if (a['reference'],a['distorted'],a['expected_bits']) != (b['reference'],b['distorted'],b['value_bits']):
            raise ValueError('A3 rendition/score differs')
    result = dict(passed=True, teacher_exact=len(rows), encoded_byte_exact=len(actual), rendered_score_exact=len(scored),
                  teacher_sha256=sha(bins/'chroma-natural-hdr'), renderer_sha256=sha(bins/'chroma-hdr-render'),
                  golden_scores_sha256=sha(golden/'natural-scores.jsonl'), seconds=time.monotonic()-start,
                  stored_pin='4e34a176419a4bf38c2ba209a42ddc3e1bf2fcd06a38a42a50e0f485f1689e55')
    save(out/'REPLAY.json',result)
    print(canonical(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='command', required=True)
    a = s.add_parser('census'); a.add_argument('repo', type=Path); a.add_argument('out', type=Path)
    a = s.add_parser('prepare'); a.add_argument('root', type=Path); a.add_argument('renderer', type=Path)
    a = s.add_parser('spec'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path); a.add_argument('pins', type=Path); a.add_argument('executor', type=Path); a.add_argument('out', type=Path)
    a = s.add_parser('replay'); a.add_argument('base', type=Path); a.add_argument('bins', type=Path); a.add_argument('out', type=Path)
    a = s.add_parser('smoke'); a.add_argument('base', type=Path); a.add_argument('bins', type=Path); a.add_argument('executor', type=Path); a.add_argument('image'); a.add_argument('out', type=Path)
    a = s.add_parser('mirror-inputs'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path)
    a = s.add_parser('sample-spec'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path); a.add_argument('ctl', type=Path)
    a = s.add_parser('host-sample'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path); a.add_argument('image'); a.add_argument('host'); a.add_argument('cpuset'); a.add_argument('memory')
    a = s.add_parser('admit'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path); a.add_argument('ledger', type=Path); a.add_argument('out', type=Path); a.add_argument('--qualification',action='store_true')
    a = s.add_parser('declare-production'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path); a.add_argument('ctl', type=Path); a.add_argument('build_commit'); a.add_argument('image')
    a = p.parse_args()
    if a.command == 'census': census(a.repo, a.out)
    elif a.command == 'prepare': prepare(a.root, a.renderer)
    elif a.command == 'mirror-inputs': mirror_inputs(a.root, a.mirror)
    elif a.command == 'replay': replay(a.base, a.bins, a.out)
    elif a.command == 'smoke': smoke(a.base, a.bins, a.executor, a.image, a.out)
    elif a.command == 'sample-spec': sample_spec(a.root,a.mirror,a.ctl)
    elif a.command == 'declare-production': declare_production(a.root,a.mirror,a.ctl,a.build_commit,a.image)
    elif a.command == 'host-sample': host_sample(a.root,a.mirror,a.image,a.host,a.cpuset,a.memory)
    elif a.command == 'admit': admit(a.root,a.mirror,a.ledger,a.out,a.qualification)
    else: specification(a.root, a.mirror, a.pins, a.executor, a.out)
