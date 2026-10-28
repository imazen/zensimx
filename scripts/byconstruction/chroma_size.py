#!/usr/bin/env python3
"""Measure Mitchell views and native integer crops below the registered cap."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from chroma_pilot import save,sha
from chroma_evif import BASE,specs
root=Path(sys.argv[1]);root.mkdir();(root/'src').mkdir();(root/'logs').mkdir()
records=[]
for source in ['im9040','im5044','im2406']:
    raw=BASE/'raw'/(source+'.png')
    for size in [32,64,128,192,256,384,512,640,768,960]:
        dest=root/'src'/f'{source}_m{size}.png';start=time.monotonic()
        with (root/'logs'/f'{dest.stem}.log').open('w') as f:subprocess.run([str(BASE/'bin/prepare'),'resize','--in',str(raw),'--out',str(dest),'--max',str(size)],stdout=f,stderr=subprocess.STDOUT,check=True)
        w,h=struct.unpack('>II',dest.read_bytes()[16:24]);assert w*h<1_000_000
        records.append(dict(source=source,view='mitchell',maxside=size,width=w,height=h,seconds=time.monotonic()-start,sha256=sha(dest)))
    for size in [256,512]:
        dest=root/'src'/f'{source}_crop{size}.png';start=time.monotonic()
        args=[str(BASE/'a2/bin/chroma_tools'),'crop',str(raw),str(dest),str(size),'0']
        if source=='im2406':args=[str(BASE/'a2/bin/chroma_tools'),'rect-crop',str(raw),str(dest),str(6190-size),str(8253-size),str(size),str(size)]
        with (root/'logs'/f'{dest.stem}.log').open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
        records.append(dict(source=source,view='native-bottomright-crop' if source=='im2406' else 'native-center-crop',maxside=size,width=size,height=size,seconds=time.monotonic()-start,sha256=sha(dest)))
save(root/'VIEWS.json',records)
evif=BASE/'a2/bin/evif';timings={}
for name,(family,scales,params) in specs().items():
    if name.startswith('band'):continue
    # Three registered table settings per family, fixed before observing scores.
    indices=[0,7,15] if name.startswith('ycc') else [0,7,len(params)-1]
    pf=root/(name+'.jsonl');pf.write_text(''.join(json.dumps(params[i])+'\n' for i in indices))
    timings[name]=[]
    for label in [f'm{s}' for s in [32,64,128,192,256,384,512,640,768,960]]+['crop256','crop512']:
        views=sorted((root/'src').glob('*_'+label+'.png'))
        assert len(views)==3
        start=time.monotonic()
        with (root/'logs'/(name+'-'+label+'.log')).open('w') as f:subprocess.run([str(evif),'render','--family',family,'--params',str(pf),'--format','evif','--rendition','png','--out',str(root/name/label),*map(str,views)],stdout=f,stderr=subprocess.STDOUT,check=True)
        files=list((root/name/label).glob('*.evif'))
        if len(files)!=9:raise ValueError('size pilot refused cells')
        timings[name].append(dict(view=label,seconds=time.monotonic()-start,cells=9,container_bytes=sum(p.stat().st_size for p in files)))
save(root/'TIMING.json',dict(views=len(records),tiny_views=sum(r['maxside']<=64 for r in records),max_pixels=max(r['width']*r['height'] for r in records),preparation_seconds=sum(r['seconds'] for r in records),families=timings,prepare_sha256=sha(BASE/'bin/prepare'),evif_sha256=sha(evif),crop_sha256=sha(BASE/'a2/bin/chroma_tools')))
