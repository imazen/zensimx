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


def consumer_roles(root, mirror, build_commit):
    """Freeze source-only roles before joining already admitted targets."""
    import pyarrow.parquet as pq
    final=json.loads((root/'FINAL.json').read_text())
    if not final['passed'] or sha(root/'_MANIFEST.json')!=final['manifest_sha256']:
        raise ValueError('qualified parent archive required')
    if len(build_commit)!=40 or any(c not in '0123456789abcdef' for c in build_commit):
        raise ValueError('full role source commit required')
    sources=json.loads((root/'SOURCE_CENSUS.json').read_text())['sources']
    grouped=defaultdict(list)
    for source in sources:
        if source['split']!='train':raise ValueError('canonical role must remain TRAIN')
        grouped[source['content_class']].append(source)
    roles=[]
    for content_class,items in sorted(grouped.items()):
        def key(source):return digest(('hdr-chroma-consumer-origin-v1:'+source['id']).encode())
        ordered=sorted(items,key=key)
        count=max(1,min(len(items)-1,(3*len(items)+5)//10))
        for index,source in enumerate(ordered):
            roles.append(dict(source=source['id'],source_sha256=source['source_sha256'],content_class=content_class,
                              canonical_split='train',consumer_role='eval' if index<count else 'train',selection_sha256=key(source)))
    out=root/'consumer-roles-r1';out.mkdir()
    plan=dict(schema='hdr-chroma-consumer-origin-roles-v1',build_commit=build_commit,parent_manifest_sha256=final['manifest_sha256'],
              algorithm='Within each class sort sha256(UTF8(hdr-chroma-consumer-origin-v1:<four-digit-id>)); first round-half-up(0.3*n), bounded to [1,n-1], reserved for eval',
              scope='TRAIN-side holdout for future model fits; canonical corpus split remains train; all views and rungs follow source',
              prior_exposure='Generation, admission and declared metric comparison already observed; no feature extraction or model training; not an untouched external test',
              sources=roles)
    # This label-independent assignment is durable before any target join.
    save(out/'ORIGIN_ROLES.json',plan)
    by_source={r['source']:r['consumer_role'] for r in roles}
    if len(by_source)!=38:raise ValueError('origin role coverage differs')
    counts={role:Counter(origins=sum(r['consumer_role']==role for r in roles)) for role in ['train','eval']}
    ladders=pq.read_table(root/'admission-r1/LADDERS.parquet').to_pylist()
    for ladder in ladders:
        role=by_source[ladder['source']]
        if ladder['split']!='train':raise ValueError('canonical ladder role differs')
        if ladder['family'].startswith('chroma-'):
            counts[role]['chroma_ladders']+=1
            counts[role]['admitted_chroma_ladders']+=bool(ladder['kept_indices'])
            counts[role]['kept_chroma_rungs']+=len(ladder['kept_indices'])
    views={role:set() for role in counts}
    for cell in pq.read_table(root/'admission-r1/CELLS.parquet',columns=['source','view_id']).to_pylist():
        role=by_source[cell['source']];views[role].add(cell['view_id']);counts[role]['cells']+=1
    for role in counts:counts[role]['views']=len(views[role])
    if sum(x['kept_chroma_rungs'] for x in counts.values())!=10422:raise ValueError('role join lost admitted rungs')
    result=dict(passed=True,build_commit=build_commit,parent_manifest_sha256=final['manifest_sha256'],
                origin_roles_sha256=sha(out/'ORIGIN_ROLES.json'),counts=counts,features=False,training=False)
    save(out/'SUMMARY.json',result)
    destination=mirror/out.name;destination.mkdir()
    for source in sorted(out.iterdir()):
        shutil.copyfile(source,destination/source.name)
        if sha(source)!=sha(destination/source.name):raise ValueError('consumer role mirror differs')
    print(canonical(result),flush=True)


def seal(root,mirror,build_commit):
    import random
    import pyarrow.parquet as pq
    if len(build_commit)!=40 or any(c not in '0123456789abcdef' for c in build_commit):raise ValueError('full sealing source commit required')
    admission=json.loads((mirror/'admission-r1/SUMMARY.json').read_text())
    oracle=json.loads((mirror/'oracle-full-r1/SUMMARY.json').read_text())
    comparison_result=json.loads((root/'comparison-r2/SUMMARY.json').read_text())
    audit=json.loads((mirror/'ledger-audit-final/SUMMARY.json').read_text())
    copied=json.loads((root/'CAS_MIRROR.json').read_text())
    if not admission['passed'] or not admission['qualified_full_grid'] or admission['cells']!=16480 or admission['views']!=456:raise ValueError('full admission missing')
    if not oracle['passed'] or oracle['cells']!=16480 or oracle['ledger_sha256']!=admission['ledger_sha256']:raise ValueError('independent full decode proof missing')
    if not comparison_result['passed'] or comparison_result['cells']!=352 or not comparison_result['targets_unchanged']:raise ValueError('declared comparison missing')
    if not audit['passed'] or audit['unique_done_jobs']!=456 or audit['conflicting_jobs']!=0:raise ValueError('raw ledger audit missing')
    if not copied['passed']:raise ValueError('complete CAS copy missing')
    def copy_file(source,destination):
        if source.resolve()==destination.resolve():return
        destination.parent.mkdir(parents=True,exist_ok=True)
        expected=sha(source)
        if not destination.exists():shutil.copyfile(source,destination)
        if sha(destination)!=expected:raise ValueError('metadata mirror differs: '+str(destination))
    roots=['plan','admission-r1','oracle-full-r1','ledger-audit-final','ledger-raw-final','rates','qualification','ledgers','binaries']
    for directory in roots:
        for source in sorted((mirror/directory).rglob('*')):
            if source.is_file():copy_file(source,root/source.relative_to(mirror))
    # The sample is read from already verified media, not a new encode.
    for source in sorted((root/'comparison-r2').rglob('*')):
        if source.is_file():copy_file(source,mirror/source.relative_to(root))
    filenames=['SOURCE_CENSUS.json','VIEWS.json','GEOMETRY.json','DUPLICATE_VIEWS.json','PLAN.json','COMPARISON_SELECTION.json','SPEC.json','BINARY_PINS.json','INPUT_MIRROR.json','CAS_MIRROR.json','HOST_SAMPLE_PARITY.json','PRODUCTION_RATES.json']
    for name in filenames:
        source=root/name if (root/name).is_file() else mirror/name
        copy_file(source,root/name);copy_file(source,mirror/name)
    # Adjacent native/rendition paths on the primary share the authenticated CAS.
    cells=pq.read_table(root/'admission-r1/CELLS.parquet').to_pylist()
    pairdir=root/'artifacts/zenavif';pairdir.mkdir(parents=True,exist_ok=True)
    for cell in cells:
        for name,value in [(cell['encoded_sha256']+'.avif',cell['encoded_sha256']),(cell['encoded_sha256']+'.pq.png',cell['rendition_sha256'])]:
            source=cas(root/'cas',value);dest=pairdir/name
            if not source.exists():raise ValueError('primary canonical media object missing')
            if not dest.exists():os.link(source,dest)
    samples=[]
    for record in random.Random(20261029).sample(copied['records'],3):
        for base in [root,mirror]:
            if sha(base/record['path'])!=record['sha256']:raise ValueError('final random mirror readback differs')
        samples.append(record)
    metadata=[]
    for name in filenames:
        path=root/name;metadata.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
    for directory in roots+['comparison-r2']:
        for path in sorted((root/directory).rglob('*')):
            if path.is_file():metadata.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)))
    inputs=json.loads((root/'INPUT_MIRROR.json').read_text())
    record=dict(schema='hdr-chroma-complete-v1',build_commit=build_commit,postprocess_source_sha256=sha(Path(__file__)),qualified_generation_and_admission=True,features=False,training=False,
                corpus=json.loads((root/'SOURCE_CENSUS.json').read_text()),geometry=json.loads((root/'GEOMETRY.json').read_text()),
                admission={k:v for k,v in admission.items() if k!='outputs'},comparison=comparison_result,
                independent_oracle={k:v for k,v in oracle.items() if k!='records'},raw_ledger_audit={k:v for k,v in audit.items() if k!='raw_files'},
                production_rates=json.loads((root/'PRODUCTION_RATES.json').read_text()),
                cas=dict(objects=copied['objects'],bytes=copied['bytes'],index_sha256=sha(root/'CAS_MIRROR.json')),
                input_files=inputs['records'],metadata_files=metadata,random_readbacks=samples,
                color_contract=dict(encoded_cicp=[9,16,6,True],encoded_bit_depth=10,rendition_cicp=[9,16,0,True],rendition_bit_depth=16,natural_ingress='unclipped absolute linear BT709 divided by100; standard SSIMULACRA2 cube-root'),
                scalar_map_availability='frozen natural SSIMULACRA2 JSONL and E26 score-pairs interfaces expose scalars; no approximate map is substituted',
                source_identity='committed manifests and LFS OIDs; pre-existing corpus working files were never edited',
                retained_exclusions=['failed orchestration attempts','initial declaration before corrected source-commit binding','CAS staging cleanup residue','scratch'],
                canonical_directories=['cas','raw','views',*roots,'comparison-r2','artifacts/zenavif'])
    if (root/'_MANIFEST.json').exists() or (mirror/'_MANIFEST.json').exists():raise ValueError('preserve the existing final archive seal')
    save(root/'_MANIFEST.json',record);copy_file(root/'_MANIFEST.json',mirror/'_MANIFEST.json')
    result=dict(passed=True,build_commit=build_commit,manifest_sha256=sha(root/'_MANIFEST.json'),mirrored_manifest_sha256=sha(mirror/'_MANIFEST.json'),metadata_files=len(metadata),cas_objects=copied['objects'],cas_bytes=copied['bytes'],cells=16480,views=456,features=False,training=False)
    save(root/'FINAL.json',result);copy_file(root/'FINAL.json',mirror/'FINAL.json')
    print(canonical(result),flush=True)


def mirror_cas(root,mirror,receipt):
    """Copy every immutable hash object to the primary host; verify both copies."""
    from concurrent.futures import ThreadPoolExecutor
    archive=mirror/'cas';local=root/'cas';local.mkdir(exist_ok=True)
    objects=[]
    for first in sorted(archive.iterdir()):
        if not first.is_dir() or len(first.name)!=2:continue
        for second in sorted(first.iterdir()):
            if not second.is_dir() or len(second.name)!=2:continue
            for path in second.iterdir():
                if len(path.name)==64 and all(c in '0123456789abcdef' for c in path.name):objects.append(path)
    if not objects:raise ValueError('canonical CAS is empty')
    def copy(path):
        destination=local/path.relative_to(archive);destination.parent.mkdir(parents=True,exist_ok=True)
        expected=path.name
        if not destination.exists():
            check=hashlib.sha256()
            with path.open('rb') as source,destination.open('xb') as target:
                while chunk:=source.read(1024*1024):check.update(chunk);target.write(chunk)
            if check.hexdigest()!=expected:raise ValueError('NAS CAS source hash differs')
        if sha(destination)!=expected:raise ValueError('primary CAS copy differs')
        return dict(path=str(destination.relative_to(root)),sha256=expected,bytes=destination.stat().st_size)
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(copy,sorted(objects)):
            records.append(item)
            if len(records)%250==0:
                save(root/'CAS_MIRROR_PROGRESS.json',dict(stage='verified canonical CAS copy',completed=len(records),required=len(objects)))
                print('copied+verified',len(records),'/',len(objects),flush=True)
    result=dict(passed=True,objects=len(records),bytes=sum(r['bytes'] for r in records),records=records)
    save(receipt,result);save(root/'CAS_MIRROR_PROGRESS.json',dict(stage='complete',objects=len(records),bytes=result['bytes']))
    print(canonical({k:v for k,v in result.items() if k!='records'}),flush=True)


def audit_raw_ledgers(directory,out):
    import pyarrow.parquet as pq
    out.mkdir()
    groups=defaultdict(list);status=Counter();files=[]
    for path in sorted(directory.glob('*.parquet')):
        rows=pq.read_table(path).to_pylist();files.append(dict(path=path.name,sha256=sha(path),rows=len(rows)))
        for row in rows:
            status[row['status']]+=1
            if row['status']=='done':groups[row['job_id']].append(row)
    repeats=[];conflicts=[]
    for key,rows in groups.items():
        outputs={r['output_sha'] for r in rows}
        identities={(r['image_path'],r['codec'],r['q'],r['knob_tuple_json']) for r in rows}
        if len(outputs)!=1 or len(identities)!=1 or None in outputs:conflicts.append(dict(job_id=key,rows=rows))
        if len(rows)>1:repeats.append(dict(job_id=key,records=len(rows),outputs=sorted(outputs),workers=sorted({r['worker'] for r in rows}),cross_host=len({r['worker'].split('-chromahdr-',1)[0] for r in rows})>1))
    save(out/'REPEATS.json',repeats);save(out/'CONFLICTS.json',conflicts)
    result=dict(passed=not conflicts and len(groups)==456 and set(status)=={'done'},raw_files=files,statuses=dict(status),unique_done_jobs=len(groups),extra_done_records=sum(r['records']-1 for r in repeats),repeated_jobs=len(repeats),cross_host_repeated_jobs=sum(r['cross_host'] for r in repeats),conflicting_jobs=len(conflicts))
    save(out/'SUMMARY.json',result)
    if not result['passed']:raise ValueError('full raw ledger audit failed; inspect preserved conflict/coverage records')
    print(canonical({k:v for k,v in result.items() if k!='raw_files'}),flush=True)


def snapshot_rates(root,out):
    import pyarrow.parquet as pq
    from datetime import datetime,timezone
    snapshots=[]
    for name in ['t1','t2','t3']:
        path=root/'ledgers'/name/'snap_chroma-hdr-full-20261029.parquet'
        rows=pq.read_table(path).to_pylist();byhost=Counter()
        for row in rows:
            if row['status']!='done':raise ValueError('production rate snapshot contains failed row')
            n=52 if json.loads(row['knob_tuple_json'])['control'] else 36
            byhost[row['worker'].split('-chromahdr-',1)[0]]+=n
        snapshots.append(dict(stage=name,ledger_sha256=sha(path),modified_ns=path.stat().st_mtime_ns,snapshot_written_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),done_views=len(rows),done_cells=sum(byhost.values()),by_host_cells=dict(byhost)))
    before,after=snapshots[:2];seconds=(after['modified_ns']-before['modified_ns'])/1e9
    if seconds<=0:raise ValueError('snapshot observation timestamps not ordered')
    delta={host:after['by_host_cells'].get(host,0)-before['by_host_cells'].get(host,0) for host in ['i270','i265','r3500','r3800x']}
    if any(v<0 for v in delta.values()):raise ValueError('rate attribution regressed')
    result=dict(passed=True,measurement='actual unique first-wins completed cells between native ledger snapshot file-write timestamps',snapshots=snapshots,window_seconds=seconds,delta_cells=delta,cells_per_second={k:v/seconds for k,v in delta.items()})
    save(out,result);print(canonical({k:v for k,v in result.items() if k!='snapshots'}),flush=True)


def oracle_all(ledger,archive,oracle_base,golden,script,out,threads):
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir()
    rows=pq.read_table(ledger).to_pylist()
    if len(rows)!=456 or any(r['status']!='done' for r in rows):raise ValueError('independent full oracle requires all456 DONE views')
    if not 1<=threads<=4:raise ValueError('independent oracle concurrency must stay in1..4')
    def execute(row):
        receipt=cas(archive,row['output_sha'])
        identity=Path(row['image_path']).parent.name+'_'+Path(row['image_path']).stem
        dest=out/identity
        with (out/(identity+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-B',str(script),'--view-receipt',str(receipt),'--archive',str(archive),'--oracle-base',str(oracle_base),'--out',str(dest),'--golden',str(golden)],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((dest/'ORACLE.json').read_text())
        if not result['passed'] or result['receipt_sha256']!=row['output_sha']:raise ValueError('independent oracle receipt differs')
        return dict(view_id=identity,receipt_sha256=row['output_sha'],cells=result['cells'],oracle_sha256=sha(dest/'ORACLE.json'),headers_sha256=result['headers_sha256'],framemd5_sha256=result['framemd5_sha256'])
    tested=[]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for result in pool.map(execute,sorted(rows,key=lambda r:r['image_path'])):
            tested.append(result)
            save(out/'PROGRESS.json',dict(stage='independent AV1 decode and quantizer headers',completed_views=len(tested),required_views=len(rows),cells=sum(r['cells'] for r in tested)))
    result=dict(passed=True,views=len(tested),cells=sum(r['cells'] for r in tested),ledger_sha256=sha(ledger),oracle_script_sha256=sha(script),ffmpeg_sha256=sha(oracle_base/'bin/ffmpeg'),records=tested,checks=['every native AVIF independently decoded by libdav1d','PQ BT2020 full-range10-bit420 and explicit matrix6','luma base quantizer fixed at each quality','shared U/V deltas match every requested rung'])
    if result['cells']!=16480:raise ValueError('full independent oracle cell census differs')
    save(out/'SUMMARY.json',result);save(out/'PROGRESS.json',dict(stage='complete',views=len(tested),cells=result['cells']))
    print(canonical({k:v for k,v in result.items() if k!='records'}),flush=True)


def comparison(root,mirror,teacher,base,out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out.mkdir()
    if sha(teacher)!='1278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721':raise ValueError('E26 executable differs')
    conditions=['--hdr','--hdr-common-primaries','--gpu-runtime','cpu','--group-by-ref','--hdrvdp3-ppd','60','--hdrvdp3-task','quality','--hdrvdp3-input','rgb-bt709','--hdrvdp3-emission','led-lcd-srgb','--hdrvdp3-age','24','--hdrvdp3-surround','none']
    sources=json.loads((base/'a2/hdr/SOURCES.json').read_text())
    control=next(r for r in sources if r.get('pilot_control_dist'))
    def locate(path):
        p=Path(path)
        if p.exists():return p
        return base/str(p).split('/byc-chroma-20261028/',1)[1]
    ref,dist=locate(control['pilot_ref']),locate(control['pilot_control_dist'])
    if sha(ref)!=control['ref_sha256'] or sha(dist)!=control['dist_sha256']:raise ValueError('E26 control bytes differ')
    with (out/'control.tsv').open('x') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path']);w.writerow([ref,dist])
    with (out/'control.log').open('x') as f:subprocess.run([str(teacher),'score-pairs','--metric','hdrvdp3',*conditions,'--pairs-tsv',str(out/'control.tsv'),'--out-parquet',str(out/'control.parquet')],stdout=f,stderr=subprocess.STDOUT,check=True)
    control_rows=pq.read_table(out/'control.parquet').to_pylist();column='hdrvdp3_jod_imazen_v0_0_1'
    actual=control_rows[0][column]
    save(out/'CONTROL.json',dict(passed=actual==control['hdrvdp3_q_jod'],actual=actual,expected=control['hdrvdp3_q_jod'],teacher_sha256=sha(teacher),conditions=conditions))
    if actual!=control['hdrvdp3_q_jod']:raise ValueError('frozen E26 score replay differs')
    selected={r['view_id']:r for r in json.loads((root/'COMPARISON_SELECTION.json').read_text())}
    cells=[r for r in pq.read_table(mirror/'qualification/admission/CELLS.parquet').to_pylist() if r['view_id'] in selected]
    if len(cells)!=352 or len({r['view_id'] for r in cells})!=8:raise ValueError('declared comparison coverage differs')
    archive=mirror/'cas'
    named=out/'pq16';named.mkdir()
    def named_png(value):
        path=named/(value+'.pq.png')
        if not path.exists():shutil.copyfile(cas(archive,value),path)
        if sha(path)!=value:raise ValueError('E26 named native PNG differs from CAS')
        return path
    with (out/'pairs.tsv').open('x') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path','image_path','codec','q','knob_tuple_json'])
        for c in cells:
            reference,rendition=named_png(c['reference_sha256']),named_png(c['rendition_sha256'])
            if sha(reference)!=c['reference_sha256'] or sha(rendition)!=c['rendition_sha256']:raise ValueError('comparison pair bytes differ')
            w.writerow([reference,rendition,c['view_id']+'|'+c['encoded_sha256'],'zenavif-svt-hdr',c['quality'],canonical(dict(chroma_q=c['chroma_q']))])
    with (out/'teacher.log').open('x') as f:subprocess.run([str(teacher),'score-pairs','--metric','hdrvdp3',*conditions,'--pairs-tsv',str(out/'pairs.tsv'),'--out-parquet',str(out/'hdrvdp3.parquet')],stdout=f,stderr=subprocess.STDOUT,check=True)
    rows=pq.read_table(out/'hdrvdp3.parquet').to_pylist()
    bykey={(r['image_path'],float(r['q']),canonical(json.loads(r['knob_tuple_json']))):r for r in rows}
    if len(bykey)!=len(cells):raise ValueError('comparison lost/duplicated pair identities')
    for c in cells:
        key=(c['view_id']+'|'+c['encoded_sha256'],float(c['quality']),canonical(dict(chroma_q=c['chroma_q'])))
        c['hdrvdp3_jod']=bykey[key][column];c['hdrvdp3_score']=10*c['hdrvdp3_jod']
    ladders=[]
    for identity,selection in selected.items():
        subset=[c for c in cells if c['view_id']==identity]
        groups=[('chroma-q'+str(q),[(q,d) for d in DELTAS]) for q in QUALITIES]
        if selection['control']:groups.append(('luma-control',[(q,0) for q in CONTROLS]))
        recipes={(c['quality'],c['chroma_q'][0]):c for c in subset}
        for family,keys in groups:
            natural=[recipes[k]['natural_ssim2'] for k in keys];vdp=[recipes[k]['hdrvdp3_score'] for k in keys]
            ladders.append(dict(view_id=identity,content_class=selection['content_class'],family=family,natural_scores=natural,comparison_scores=vdp,natural_admission=admission(natural),comparison_admission=admission(vdp),natural_endpoint_drop=natural[0]-natural[-1],comparison_endpoint_drop=vdp[0]-vdp[-1]))
    pq.write_table(pa.Table.from_pylist(cells),out/'CELLS.parquet',compression='zstd');pq.write_table(pa.Table.from_pylist(ladders),out/'LADDERS.parquet',compression='zstd')
    result=dict(passed=True,views=8,cells=len(cells),chroma_ladders=sum(r['family']!='luma-control' for r in ladders),natural_chroma_admitted=sum(bool(r['natural_admission']['kept_indices']) for r in ladders if r['family']!='luma-control'),comparison_chroma_admitted=sum(bool(r['comparison_admission']['kept_indices']) for r in ladders if r['family']!='luma-control'),control_ladders=sum(r['family']=='luma-control' for r in ladders),natural_controls_admitted=sum(bool(r['natural_admission']['kept_indices']) for r in ladders if r['family']=='luma-control'),comparison_controls_admitted=sum(bool(r['comparison_admission']['kept_indices']) for r in ladders if r['family']=='luma-control'),teacher_sha256=sha(teacher),selection_sha256=sha(root/'COMPARISON_SELECTION.json'),conditions=conditions,targets_unchanged=True)
    save(out/'SUMMARY.json',result);print(canonical(result),flush=True)


def declare_production(root,mirror,ctl,build_commit,image):
    if len(build_commit)!=40 or any(c not in '0123456789abcdef' for c in build_commit):raise ValueError('full source commit hash required')
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
    a = s.add_parser('comparison'); a.add_argument('root', type=Path); a.add_argument('mirror', type=Path); a.add_argument('teacher', type=Path); a.add_argument('base', type=Path); a.add_argument('out', type=Path)
    a = s.add_parser('oracle-all'); a.add_argument('ledger', type=Path); a.add_argument('archive', type=Path); a.add_argument('oracle_base', type=Path); a.add_argument('golden', type=Path); a.add_argument('script', type=Path); a.add_argument('out', type=Path); a.add_argument('--threads',type=int,default=4)
    a = s.add_parser('audit-raw-ledgers'); a.add_argument('directory',type=Path); a.add_argument('out',type=Path)
    a = s.add_parser('snapshot-rates'); a.add_argument('root',type=Path); a.add_argument('out',type=Path)
    a = s.add_parser('mirror-cas'); a.add_argument('root',type=Path); a.add_argument('mirror',type=Path); a.add_argument('receipt',type=Path)
    a = s.add_parser('consumer-roles'); a.add_argument('root',type=Path); a.add_argument('mirror',type=Path); a.add_argument('build_commit')
    a = s.add_parser('seal'); a.add_argument('root',type=Path); a.add_argument('mirror',type=Path); a.add_argument('build_commit')
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
    elif a.command == 'comparison': comparison(a.root,a.mirror,a.teacher,a.base,a.out)
    elif a.command == 'audit-raw-ledgers': audit_raw_ledgers(a.directory,a.out)
    elif a.command == 'mirror-cas': mirror_cas(a.root,a.mirror,a.receipt)
    elif a.command == 'consumer-roles': consumer_roles(a.root,a.mirror,a.build_commit)
    elif a.command == 'seal': seal(a.root,a.mirror,a.build_commit)
    elif a.command == 'snapshot-rates': snapshot_rates(a.root,a.out)
    elif a.command == 'oracle-all': oracle_all(a.ledger,a.archive,a.oracle_base,a.golden,a.script,a.out,a.threads)
    else: specification(a.root, a.mirror, a.pins, a.executor, a.out)
