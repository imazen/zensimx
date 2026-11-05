#!/usr/bin/env python3
"""Assemble one IMIJ-set and code-only Pages artifact; no implicit publication."""
import argparse, gzip, hashlib, json, re, shutil, subprocess, urllib.request
from pathlib import Path

BOOTSTRAP=65536
A_SHA='d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a'
DATA_BASE='https://pub-0cc70fdd54ac4a4a837122a2b951ba1a.r2.dev'
ROOT=Path(__file__).resolve().parent
FORBIDDEN=re.compile(r'/home/|/mnt/|/var/|source_paths|zenpapers|\b(?:dev|tower|r7900x|r5900xt)\.lan\b|aic[-_ ]?2026',re.I)


def check_data(value):
    text=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    if FORBIDDEN.search(text):raise ValueError('private provenance remains in public data')
    return text.encode()


def sanitize(value):
    if isinstance(value,dict):return {k:sanitize(v) for k,v in value.items() if k not in {'bank_path','source_paths','path','bake','bank_provenance','certificate_binary_sha256'}}
    if isinstance(value,list):return [sanitize(v) for v in value]
    return value


def compact(source,destination,compressed=False,fields=None):
    value=sanitize(json.loads(source.read_text()))
    if fields:value={k:value[k] for k in fields}
    data=check_data(value)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(gzip.compress(data,mtime=0) if compressed else data)


def assemble(a):
    if not re.fullmatch('[0-9a-f]{40}',a.source):raise ValueError('full public source commit required')
    a.out.mkdir(parents=True,exist_ok=False)
    payload=a.out/'payload';payload.mkdir()
    for name in ['sdr','hdr']:
        src=a.exports/name;dest=payload/name;dest.mkdir()
        compact(src/'bank.json',dest/'bank.json')
        for file in ['sample.f32le','quantiles.f32le','model-quantiles.f32le']:shutil.copyfile(src/file,dest/file)
    for name in ['audit','function']:
        if json.loads((a.exports/(name+'.json')).read_text())['bake_sha256']!=A_SHA:raise ValueError('inspection bake identity differs')
    fields={
        'weights':['inputs','row_order','column_order','per_input_columns','top_inputs_by_unit','nonzero_counts','input_l1','capacity','scaler_mean','scaler_scale','layers'],
        'audit':['bake_sha256','bank_sha256','bank_rows','weight_count','counts','thresholds','flags'],
        'function':['schema','bake_sha256','output_bias','certificate','pairs','linear_background','other_hinge_count','global_affine_forms_upper_bound','certified_line_segments_upper_bound','preprocessing','memorization','algebra_scope','global_bound_scope','certified_line_bound_scope','output']}
    for name in fields:compact(a.exports/(name+'.json'),payload/(name+'.json.gz'),True,fields[name])
    reference=json.loads(a.reference.read_text())
    # Only registry equations and public implementation citations. No paper field,
    # prose, excerpt, inspiration text or private link enters this artifact.
    refs={key:{'formula':entry.get('formula'),'inspiration':[{k:c[k] for k in ['author_year','title','url']} for c in entry.get('inspiration',[])],'code':entry.get('code').replace('/blob/main/','/blob/'+a.source+'/') if entry.get('code') else None} for key,entry in reference['entries'].items()}
    (payload/'reference.json').write_bytes(check_data(refs))
    bake=a.bake.read_bytes()
    if hashlib.sha256(bake).hexdigest()!=A_SHA:raise ValueError('A bake SHA differs')
    if FORBIDDEN.search(bake.decode('latin1')):raise ValueError('bake contains private path/provenance')
    (payload/'A.bin').write_bytes(bake)
    files=sorted(p for p in payload.rglob('*') if p.is_file())
    entries=[];offset=BOOTSTRAP
    for p in files:
        entries.append(dict(name=p.relative_to(payload).as_posix(),offset=offset,length=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        offset+=p.stat().st_size
    catalog=dict(schema='zensim-imijset-v1',generation=1,bootstrap_bytes=BOOTSTRAP,source_commit=a.source,bytes=0,entries=entries)
    bootstrap=a.out/'catalog.json'
    def write_catalog():
        data=check_data(catalog)
        if len(data)>BOOTSTRAP:raise ValueError('bootstrap catalog overflow')
        bootstrap.write_bytes(data+b' '*(BOOTSTRAP-len(data)))
    def pack(name):
        args=[str(a.packer),str(a.out/name),'catalog.json='+str(bootstrap)]+[p.relative_to(payload).as_posix()+'='+str(p) for p in files]
        record=json.loads(subprocess.check_output(args))
        if not record['native_roundtrip_verified']:raise ValueError('native archive roundtrip not verified')
        return record
    write_catalog();first=pack('preflight.imijset');catalog['bytes']=first['bytes'];write_catalog();record=pack('bundle.imijset')
    actual=record['entries'][1:]
    if actual!=entries or record['bytes']!=catalog['bytes']:raise ValueError('native writer offsets differ from bootstrap')
    total=(a.out/'bundle.imijset').stat().st_size
    if total>10000000:raise ValueError(f'archive exceeds 10 MB: {total}')
    (a.out/'archive-identity.json').write_text(json.dumps(dict(bytes=total,sha256=hashlib.sha256((a.out/'bundle.imijset').read_bytes()).hexdigest(),entries=len(record['entries']),source_commit=a.source,native_roundtrip_verified=True),indent=2)+'\n')
    print(json.dumps(dict(objects=1,bytes=total,entries=len(record['entries']),native_roundtrip_verified=True)))


def site(a):
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['index.html','style.css','app.js','archive.js','worker.js']:shutil.copyfile(ROOT/'pages'/name,a.out/name)
    (a.out/'design').mkdir(exist_ok=True);shutil.copyfile(ROOT/'web/design/tokens.css',a.out/'design/tokens.css')
    url=a.archive_url or DATA_BASE+'/viz/'+a.source+'/bundle.imijset'
    if FORBIDDEN.search(url):raise ValueError('private deployment URL')
    (a.out/'config.js').write_text('export const DATA_BASE='+json.dumps(DATA_BASE)+';\nexport const SOURCE_COMMIT='+json.dumps(a.source)+';\nexport const ARCHIVE_URL='+('DATA_BASE+'+json.dumps('/viz/'+a.source+'/bundle.imijset') if not a.archive_url else json.dumps(url))+';\n')
    print(json.dumps(dict(site_code_bytes=sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file() and p.suffix in {'.js','.css','.html'}))))


def pull_a(a):
    def get(start,length):
        request=urllib.request.Request(a.url,headers={'Range':f'bytes={start}-{start+length-1}'})
        with urllib.request.urlopen(request) as response:
            if response.status!=206:raise ValueError('archive server must support Range')
            data=response.read()
        if len(data)!=length:raise ValueError('truncated range')
        return data
    catalog=json.loads(get(0,BOOTSTRAP))
    if catalog['schema']!='zensim-imijset-v1' or catalog['source_commit']!=a.source:raise ValueError('archive/source mismatch')
    entry=next(e for e in catalog['entries'] if e['name']=='A.bin')
    bake=get(entry['offset'],entry['length'])
    if hashlib.sha256(bake).hexdigest()!=A_SHA or entry['sha256']!=A_SHA:raise ValueError('A bake SHA differs')
    a.out.write_bytes(bake)


def workflow(a):
    if not re.fullmatch('[0-9a-f]{40}',a.source):raise ValueError('full public source commit required')
    text=(ROOT/'pages/pages.yml.in').read_text().replace('@SOURCE_COMMIT@',a.source)
    lines=text.splitlines()
    patch='diff --git a/.github/workflows/pages.yml b/.github/workflows/pages.yml\nnew file mode 100644\n--- /dev/null\n+++ b/.github/workflows/pages.yml\n@@ -0,0 +1,'+str(len(lines))+' @@\n'+''.join('+'+line+'\n' for line in lines)
    a.out.write_text(patch)
    print(json.dumps(dict(source_commit=a.source,workflow_bytes=len(text.encode()),diff_bytes=len(patch.encode()),destination='imazen/zensim:.github/workflows/pages.yml',written_locally_only=True)))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('assemble')
    for name in ['exports','reference','bake','packer','out']:a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--source',required=True);a.set_defaults(run=assemble)
    a=sub.add_parser('site');a.add_argument('--source',required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--archive-url');a.set_defaults(run=site)
    a=sub.add_parser('pull-a');a.add_argument('--source',required=True);a.add_argument('--url',required=True);a.add_argument('--out',type=Path,required=True);a.set_defaults(run=pull_a)
    a=sub.add_parser('workflow');a.add_argument('--source',required=True);a.add_argument('--out',type=Path,required=True);a.set_defaults(run=workflow)
    a=p.parse_args();a.run(a)

if __name__=='__main__':main()
