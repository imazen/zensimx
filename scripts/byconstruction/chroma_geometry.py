#!/usr/bin/env python3
"""Qualify every source and freeze nonupscaled Mitchell/native-crop views.

Imazen tools own generated pixels. Pillow is an independent hash/crop oracle.
Artifacts and per-source receipts are append-only, under the registered NAS run.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from collections import Counter
from PIL import Image
from chroma_pilot import save,sha
base=Path('/var/tmp/byc-chroma-20261028');root=Path(sys.argv[1]);root.mkdir(exist_ok=True,parents=True)
(root/'src').mkdir(exist_ok=True);(root/'receipts').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
candidates=json.loads((root.parent/'CANDIDATES.json').read_text());records=[];excluded=[];views=[]
prepare=base/'bin/prepare';crop=base/'a2/bin/chroma_tools'
assert sha(prepare)=='cb23096d02f5696d9517442ab3d8ce6dcafa6868390dd0f2fc50dcbd1de65947'
def pixel_sha(im):return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
def cmd(args):
    with (root/'logs/prepare.log').open('a') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)
for index,s in enumerate(candidates['sources']):
    receipt=root/'receipts'/(s['id']+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());records.append(r);views.extend(r['views']);continue
    raw=Path(s['worker_raw']);digest=sha(raw)
    if s.get('source_sha256') and digest!=s['source_sha256']:raise ValueError('canonical LFS bytes differ: '+s['id'])
    with Image.open(raw) as image:
        w,h=image.size
        if w*h>100_000_000:
            excluded.append(dict(source=s,reason='frozen preparation decoder 100M guard',width=w,height=h));continue
        original=image.convert('RGB');pixels=pixel_sha(original)
    if s.get('source_pixels_sha256') and pixels!=s['source_pixels_sha256']:raise ValueError('bank source pixels differ: '+s['id'])
    if s['split']!='train':raise ValueError('non-TRAIN source reached preparation')
    proposed=[]
    for cap in [32,64,128,192,256,384,512,640,768,960]:
        # A cap beyond native dimensions gives no additional view. Include the
        # native image once when it is smaller than the largest requested fit.
        actual=min(cap,max(w,h))
        if actual in [r['cap'] for r in proposed if r['kind']=='mitchell']:continue
        dest=root/'src'/f'{s["id"]}_m{actual}.png'
        if not dest.exists():
            if actual==max(w,h):shutil.copyfile(raw,dest)
            else:cmd([prepare,'resize','--in',raw,'--out',dest,'--max',actual])
        proposed.append(dict(id=dest.stem,path=str(dest),kind='mitchell',cap=actual))
    label=s.get('representative',{}).get('crop_label','full')
    if s.get('substitutes')=='im2402':label='c25_br'
    for size in [256,512]:
        if min(w,h)<size:continue
        x=0 if label.endswith('_tl') or label.endswith('_bl') else w-size if label.endswith('_tr') or label.endswith('_br') else (w-size)//2
        y=0 if label.endswith('_tl') or label.endswith('_tr') else h-size if label.endswith('_bl') or label.endswith('_br') else (h-size)//2
        dest=root/'src'/f'{s["id"]}_crop{size}.png'
        if not dest.exists():cmd([crop,'rect-crop',raw,dest,x,y,size,size])
        with Image.open(dest) as check:
            if check.convert('RGB').tobytes()!=original.crop((x,y,x+size,y+size)).tobytes():raise ValueError('native crop oracle differs: '+s['id'])
        proposed.append(dict(id=dest.stem,path=str(dest),kind='native-crop',cap=size,x=x,y=y,anchor=label))
    admitted=[];seen={}
    for v in proposed:
        with Image.open(v['path']) as im:
            vw,vh=im.size;pixel=pixel_sha(im)
        if vw*vh>=1_000_000 or vw>w or vh>h:raise ValueError('cap/upscale violation: '+v['id'])
        v.update(source=s['id'],reference_group=s['reference_group'],mlic_role=s['mlic_role'],content_class=s['content_class'],type=s['type'],width=vw,height=vh,sha256=sha(v['path']),pixels_sha256=pixel)
        key=(vw,vh,pixel)
        if key in seen:v['duplicate_of']=seen[key]
        else:seen[key]=v['id']
        admitted.append(v)
    r=dict(source=s,raw_sha256=digest,source_pixels_sha256=pixels,width=w,height=h,views=admitted)
    save(receipt,r);records.append(r);views.extend(admitted)
    save(root/'PROGRESS.json',dict(pid=os.getpid(),stage='source-preparation',completed_sources=index+1,required_sources=len(candidates['sources']),prepared_views=len(views)))
# Global exact-pixel dedup keeps the most restrictive inherited role first.
# Removed aliases never acquire a fitting role from another source.
rank={'calibration':0,'dev':1,'fit':2,'new-train':2}
unique={};duplicates=[]
for v in sorted(views,key=lambda v:(rank[v['mlic_role']],v['id'])):
    if v.get('duplicate_of'):duplicates.append(v);continue
    key=(v['width'],v['height'],v['pixels_sha256'])
    if key in unique:
        v['duplicate_of']=unique[key]['id'];duplicates.append(v)
    else:unique[key]=v
kept=sorted(unique.values(),key=lambda v:v['id'])
tiny=sum(max(v['width'],v['height'])<=64 for v in kept)
assert tiny*2<len(kept),'tiny views must remain a minority'
assert len({r['source']['content_class'] for r in records if r['source']['type']=='imazen26'})==21
summary=dict(schema='chroma-stage-b-geometry-v1',sources=len(records),candidate_sources=len(candidates['sources']),views=len(kept),tiny_views=tiny,native_crop_views=sum(v['kind']=='native-crop' for v in kept),max_pixels=max(v['width']*v['height'] for v in kept),duplicates=len(duplicates),excluded_sources=excluded,source_types=dict(Counter(r['source']['type'] for r in records)),content_classes=dict(Counter(v['content_class'] for v in kept)),mlic_roles=dict(Counter(v['mlic_role'] for v in kept)),prepare_sha256=sha(prepare),crop_sha256=sha(crop),substitution=candidates['substitution'])
save(root/'SOURCES.json',records);save(root/'VIEWS.json',kept);save(root/'DUPLICATE_VIEWS.json',duplicates);save(root/'SUMMARY.json',summary);save(root/'PROGRESS.json',dict(stage='complete',sources=len(records),views=len(kept)));print(json.dumps(summary),flush=True)
