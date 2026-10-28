#!/usr/bin/env python3
"""Freeze an A2 artifact inventory; independently read three NAS blobs back."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from chroma_pilot import save,sha

def manifest(root,encoder,aux,owner):
    if (root/'_MANIFEST.json').exists():raise ValueError('archive already frozen')
    if len(list(root.glob('*/scored/*.evif')))!=12696:raise ValueError('scored sample census differs')
    files=[]
    for directory,dirs,names in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in ['target','scratch','__pycache__','.git','.jj'])
        for name in sorted(names):
            if name in ['_MANIFEST.json','MIRROR_VERIFY.json','.workongoing']:continue
            p=Path(directory)/name
            if not p.resolve().is_relative_to(root.resolve()):raise ValueError(f'external symlink: {p}')
            files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    save(root/'_MANIFEST.json',dict(schema='chroma-a2-archive-v1',build_commit=encoder,build_repository='https://github.com/imazen/evif',build_role='SDR generation binary',auxiliary_build_commit=aux,orchestration_commit=owner,files=files,total_file_bytes=sum(r['bytes'] for r in files)))
    print(json.dumps(dict(files=len(files),file_bytes=sum(r['bytes'] for r in files),manifest_sha256=sha(root/'_MANIFEST.json'))),flush=True)

def verify(root):
    m=json.loads((root/'_MANIFEST.json').read_text());rng=random.Random(20261028);tested=[]
    for predicate in [lambda p:'/scored/' in p and p.endswith('.evif'),lambda p:p.startswith('hdr/') and p.endswith('.avif') and '/first-encodes/' not in p,lambda p:p.startswith('cas/') and p.endswith('.jpg')]:
        r=rng.choice([r for r in m['files'] if predicate(r['path'])]);actual=sha(root/r['path'])
        if actual!=r['sha256']:raise ValueError('NAS readback differs')
        tested.append(dict(path=r['path'],sha256=actual,bytes=r['bytes']))
    # The transport is tar over SSH; each listed member must exist at its frozen size.
    for r in m['files']:
        if (root/r['path']).stat().st_size!=r['bytes']:raise ValueError(f"NAS inventory differs: {r['path']}")
    result=dict(passed=True,manifest_sha256=sha(root/'_MANIFEST.json'),inventory_files=len(m['files']),random_seed=20261028,random_readbacks=tested)
    if (root/'MIRROR_VERIFY.json').exists():raise ValueError('readback receipt already exists')
    save(root/'MIRROR_VERIFY.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['manifest','verify']);p.add_argument('root',type=Path);p.add_argument('--encoder');p.add_argument('--auxiliary');p.add_argument('--owner');a=p.parse_args()
    if a.action=='manifest':
        if not all([a.encoder,a.auxiliary,a.owner]):p.error('all build identities required')
        manifest(a.root,a.encoder,a.auxiliary,a.owner)
    else:verify(a.root)
