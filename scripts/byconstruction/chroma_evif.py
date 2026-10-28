#!/usr/bin/env python3
"""A2 EVIF transport; frozen teacher and chroma_pilot.admission own labels/gates.

Run on an idle worker under run-heavy. All outputs are durable and append-only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from chroma_pilot import admission, save, sha

BASE = Path('/var/tmp/byc-chroma-20261028')
SCALES = [.1,.14142136,.2,.28284271,.4,.56568542,.8,1,1.13137085,1.6,2.2627417,3.2,4,5]
CHROMA = [0,.5,.70710678,1,1.41421356,2,2.82842712,4,5.65685425,8,11.3137085,16,22.627417,32,45.254834,64]


def specs():
    def expert(sub, scale, xyb=False):
        q = [[1]*4, [scale]*4, [scale]*4]
        if xyb: q = [[scale]*4, [1]*4, [scale]*4]
        return dict(quality=95, subsampling=sub, quant_scale=q, zero_bias_scale=[1]*3,
                    adaptive_quant=True, deringing=False, trellis=False, xyb=xyb)
    result = {f'ycc{sub}': ('zenjpeg-expert', CHROMA, [expert('s'+sub,s) for s in CHROMA]) for sub in ['444','420']}
    ext = SCALES + [6.4,8,11.3137085,16,22.627417,32,45.254834,64,90.509668,128,181.019336,256,362.038672,512,724.077344,1024]
    result['xyb-extended'] = ('zenjpeg-expert', ext, [expert('s420',s,True) for s in ext])
    result['xyb-capped'] = ('zenjpeg-color', SCALES, [dict(quality=95,color=dict(path='xyb',b_quarter=True),tables=dict(family='jpegli',chroma_scales=[s,s]),downsampling='box',deringing=False,adaptive_quant=True,auto_optimize=False) for s in SCALES])
    for variant in ['dc','ac']:
        params=[]
        for s in CHROMA:
            p=expert('s444',s)
            p['quant_scale'][1]=p['quant_scale'][2]=[s,1,1,1] if variant=='dc' else [1,s,s,s]
            params.append(p)
        result['band-'+variant]=('zenjpeg-expert',CHROMA,params)
    return result


def run(root, selected):
    evif = root/'bin/evif'; teacher=BASE/'bin/teacher'
    for d in ['logs','params','scratch','cas']: (root/d).mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(root/'scratch'), ZENSIM_FORMULA_REV='3',ZENSIM_ROOT_FORM='sqrt')
    progress=dict(pid=os.getpid(),root=str(root),evif_sha256=sha(evif),teacher_sha256=sha(teacher))
    sources=sorted((BASE/'pilot-run04/src').glob('*.png'))
    if len(sources)!=162: raise ValueError('frozen 162 views required')
    def cmd(label,args):
        start=time.monotonic()
        with (root/'logs'/f'{label}.log').open('a') as log:
            subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,check=True)
        return time.monotonic()-start
    for name,(family,scales,params) in specs().items():
        if selected and name not in selected: continue
        out=root/name; out.mkdir(exist_ok=True)
        if (out/'SUMMARY.json').exists(): continue
        views=sources[:12] if name.startswith('band-') else sources
        paramfile=root/'params'/f'{name}.jsonl'
        paramfile.write_text(''.join(json.dumps(p)+'\n' for p in params))
        progress.update(stage=name+'-render',required=len(views)*len(params)); save(root/'PROGRESS.json',progress)
        if not (out/'render.done').exists():
            if (out/'render.tsv').exists(): raise ValueError('partial render: preserve and explicitly recover')
            seconds=cmd(name+'-render',[evif,'render','--family',family,'--params',paramfile,'--format','evif','--rendition','png','--out',out,*views])
            (out/'render.done').write_text(str(seconds))
        with (out/'render.tsv').open() as f: cells=list(csv.DictReader(f,delimiter='\t'))
        if len(cells)!=len(views)*len(params): raise ValueError(f'{name}: render refused cells {len(cells)}')
        extracted=out/'extracted'
        progress.update(stage=name+'-extract',completed=0);save(root/'PROGRESS.json',progress)
        for i,c in enumerate(cells):
            c['image']=str(BASE/'pilot-run04/src'/(c['image']+'.png'))
            stem=Path(c['path']).stem
            if not (extracted/(stem+'.digests.json')).exists():cmd(name+'-extract',[evif,'extract',c['path'],extracted])
            bits=extracted/(stem+'.jpeg'); blob=root/'cas'/(sha(bits)+'.jpg')
            if not blob.exists():os.link(bits,blob)
            c.update(distorted=str(extracted/(stem+'.png')),cas=str(blob),encode_sha256=sha(bits),rung=int(c['tag'][1:]))
            if i%64==0:progress.update(completed=i);save(root/'PROGRESS.json',progress)
        pairs=out/'pairs.tsv'
        with pairs.open('w') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['ref_path','dist_path','human_score'])
            for c in cells:w.writerow([c['image'],c['distorted'],0])
        progress.update(stage=name+'-teacher');save(root/'PROGRESS.json',progress)
        if not (out/'teacher.done').exists():
            seconds=cmd(name+'-teacher',[teacher,'--corpus','pairs-tsv','--path',pairs,'--out',out/'teacher.csv','--full-944','--audit-jsonl',out/'teacher.jsonl','--audit-ssim2'])
            (out/'teacher.done').write_text(str(seconds))
        audit={r['distorted']:r for r in map(json.loads,(out/'teacher.jsonl').read_text().splitlines())}
        scored=out/'scored';scored.mkdir(exist_ok=True)
        ladders={}
        for c in cells:
            a=audit[c['distorted']];s=a['peer_ssim2']['score'];c['teacher_ssim2']=s
            if a['reference']!=c['image']:raise ValueError('reference mismatch')
            stem=Path(c['path']).stem
            envelope=json.loads((extracted/(stem+'.digests.json')).read_text())
            envelope['scores']=[dict(metric='ssim2',variant='safesyn-teacher',implementation='frozen-safe-syn-executable-sha256:'+progress['teacher_sha256'],value=s,better='higher')]
            scorefile=extracted/(stem+'.import.json');scorefile.write_text(json.dumps(envelope))
            dest=scored/(stem+'.evif')
            if not dest.exists():cmd(name+'-import',[evif,'score-import',c['path'],scorefile,dest])
            c['scored_evif']=str(dest)
            ladders.setdefault(c['image'],[]).append(c)
        result=[]
        for image,cc in ladders.items():
            cc.sort(key=lambda c:c['rung'])
            gate=admission([c['teacher_ssim2'] for c in cc]);result.append(dict(image=image,admission=gate,cells=cc))
        save(out/'LADDERS.json',result)
        summary=dict(family=name,views=len(views),cells=len(cells),passed=sum(bool(l['admission']['kept_indices']) for l in result),kept=sum(len(l['admission']['kept_indices']) for l in result),scales=scales,render_seconds=float((out/'render.done').read_text()),teacher_seconds=float((out/'teacher.done').read_text()),binary_pins=progress)
        save(out/'SUMMARY.json',summary);print(json.dumps(summary),flush=True)
    progress.update(stage='complete');save(root/'PROGRESS.json',progress)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--families',nargs='*');a=p.parse_args();run(a.root,a.families)
