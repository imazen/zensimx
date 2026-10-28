#!/usr/bin/env python3
"""Independent sample/bitstream inspection plus exact EVIF score read-back."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
from chroma_pilot import save,sha
root=Path(sys.argv[1]); total=0;unique=set();roundtrips=[];ladders=0
for name in ['band-dc','band-ac','ycc444','ycc420','xyb-extended','xyb-capped']:
    out=root/name
    audit={r['distorted']:r for r in map(json.loads,(out/'teacher.jsonl').read_text().splitlines())}
    for ladder in json.loads((out/'LADDERS.json').read_text()):
        fixed=None;ladders+=1
        for cell in ladder['cells']:
            a=audit[cell['distorted']];digest=sha(cell['cas']);unique.add(digest)
            if digest!=cell['encode_sha256']:raise ValueError('CAS hash differs')
            with Image.open(cell['cas']) as im:
                im.load();w,h=im.size
                if w*h>=1_000_000:raise ValueError('view cap violated')
                comp=1 if out.name.startswith('xyb') else 0
                table=im.quantization[im.layer[comp][3]]
                if fixed is None:fixed=table
                if fixed!=table:raise ValueError('fixed luma DQT changed')
            with Image.open(cell['distorted']) as im:
                if im.size!=(w,h):raise ValueError('geometry differs')
                if hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()!=a['distorted_pixels_sha256']:raise ValueError('PNG pixels differ from teacher input')
            roundtrips.append(dict(original=cell['path'],scored=cell['scored_evif'],value=cell['teacher_ssim2'],implementation='frozen-safe-syn-executable-sha256:7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87'))
            total+=1
            if total%256==0:save(root/'ORACLE_PROGRESS.json',dict(pid=os.getpid(),stage='independent-main-decode',cells=total,required=12696))
save(root/'score-roundtrips.json',roundtrips)
with (root/'logs/score-roundtrip-readback.log').open('w') as f:subprocess.run([str(root/'bin/chroma_tools'),'verify-scores',str(root/'score-roundtrips.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
size_root=root/'size';size_views=json.loads((size_root/'VIEWS.json').read_text());crop_checks=0
for view in size_views:
    path=size_root/'src'/(view['source']+('_crop' if view['view'].startswith('native') else '_m')+str(view['maxside'])+'.png')
    with Image.open(path) as im:
        if im.width*im.height>=1_000_000:raise ValueError('size cap violated')
        if view['view'].startswith('native'):
            with Image.open(root.parent/'raw'/(view['source']+'.png')) as source:
                n=view['maxside'];x,y=((source.width-n,source.height-n) if view['source']=='im2406' else ((source.width-n)//2,(source.height-n)//2))
                if im.convert('RGB').tobytes()!=source.convert('RGB').crop((x,y,x+n,y+n)).tobytes():raise ValueError('independent crop pixels differ')
                crop_checks+=1
size_cells=0
for file in sorted(size_root.glob('*/*/*.evif')):
    dest=root/'size-oracle'/file.relative_to(size_root).parent
    if not (dest/(file.stem+'.digests.json')).exists():
        with (root/'logs/size-extract-oracle.log').open('a') as log:subprocess.run([str(root/'bin/evif'),'extract',str(file),str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
    bits=dest/(file.stem+'.jpeg')
    digest=sha(bits);blob=root/'cas'/(digest+'.jpg')
    if not blob.exists():os.link(bits,blob)
    with Image.open(bits) as im:
        im.load()
        if im.width*im.height>=1_000_000:raise ValueError('size encode cap violated')
    size_cells+=1
if len(size_views)!=36 or size_cells!=432 or crop_checks!=6:raise ValueError('size oracle lost views/cells')
save(root/'ORACLE.json',dict(passed=True,cells=total,ladders=ladders,size_cost_containers=size_cells,size_views=len(size_views),exact_native_crop_checks=crop_checks,unique_encoded_sha256=len(unique),oracle='Pillow '+Image.__version__,checks=['all JPEGs independently decode and obey cap','all CAS hashes match','all PNG RGB hashes match frozen teacher inputs','Y DQT fixed across each ladder','every score f64/pin roundtrips exactly; all media/recipes/digests unchanged']))
print(json.dumps(dict(cells=total,ladders=ladders,unique=len(unique))),flush=True)
