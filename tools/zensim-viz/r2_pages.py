#!/usr/bin/env python3
"""One-object publication to the authorized immutable viz prefix; never bucket mutations."""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path


def environment():
    credential_file = Path.home()/'.config/cloudflare/r2-credentials'
    raw = subprocess.check_output(['bash','-c','set -a; source "$1"; env -0','r2-pages',str(credential_file)])
    values = dict(item.split(b'=',1) for item in raw.split(b'\0') if b'=' in item)
    get = lambda key: values[key.encode()].decode()
    env = os.environ.copy()
    env.pop('AWS_SESSION_TOKEN',None)
    env.update(AWS_ACCESS_KEY_ID=get('R2_ACCESS_KEY_ID'),AWS_SECRET_ACCESS_KEY=get('R2_SECRET_ACCESS_KEY'),AWS_DEFAULT_REGION='auto',AWS_MAX_ATTEMPTS='1',AWS_RETRY_MODE='standard')
    return env, 'https://'+get('R2_ACCOUNT_ID')+'.r2.cloudflarestorage.com'


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inventory','upload']);p.add_argument('--source');p.add_argument('--archive',type=Path);p.add_argument('--evidence',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
    env,endpoint=environment()
    def aws(*args):
        return json.loads(subprocess.check_output(['aws','s3api',*args,'--endpoint-url',endpoint,'--output','json'],env=env))
    count=total=0;token=None
    while True:
        args=['list-objects-v2','--bucket','zensim-a','--no-paginate','--max-keys','1000']
        if token:args+=['--continuation-token',token]
        page=aws(*args)
        for item in page.get('Contents',[]):count+=1;total+=item['Size']
        if not page.get('IsTruncated'):break
        token=page['NextContinuationToken']
    result={'bucket_objects_before':count,'bucket_bytes_before':total,'write_calls':0,'uploaded_objects':0,'uploaded_bytes':0}
    if a.mode=='upload':
        if not a.source or not re.fullmatch('[0-9a-f]{40}',a.source):raise SystemExit('Expected full public source commit')
        if not a.archive or a.archive.suffix!='.imijset' or not a.evidence:raise SystemExit('Archive and browser evidence required')
        size=a.archive.stat().st_size;sha=hashlib.sha256(a.archive.read_bytes()).hexdigest();e=json.loads(a.evidence.read_text())
        if e.get('passed') is not True or e.get('archive_sha256')!=sha or e.get('code_bytes',500000)>=500000 or e.get('data_bytes')!=size or e.get('browser_range_verified') is not True:raise SystemExit('Local browser/range/budget qualification is incomplete')
        if e.get('source_commit')!=a.source or e.get('map_parity_passed') is not True or e.get('lens_parity_passed') is not True or e.get('native_pooled_bit_mismatches')!=0:raise SystemExit('Source identity or scientific qualification is incomplete')
        if not a.receipt or a.receipt.exists():raise SystemExit('Fresh durable receipt path required')
        if size>10000000:raise SystemExit('10 MB data budget exceeded')
        if total+size>100000000000:raise SystemExit(f'Owner permission required: {total}+{size} exceeds 100 GB')
        key='viz/'+a.source+'/bundle.imijset'
        existing=aws('list-objects-v2','--bucket','zensim-a','--prefix',key,'--no-paginate')
        if existing.get('Contents'):raise SystemExit('Immutable object already exists; inspect instead of overwriting')
        result.update(write_calls=1,key=key,sha256=sha,status='PUT attempted; resolve remote state before any retry')
        a.receipt.write_text(json.dumps(result,indent=2)+'\n')
        aws('put-object','--bucket','zensim-a','--key',key,'--body',str(a.archive),'--content-type','application/octet-stream','--cache-control','public,max-age=31536000,immutable','--metadata','sha256='+sha)
        result.update(uploaded_objects=1,uploaded_bytes=size,status='PUT returned successfully; verifying')
        a.receipt.write_text(json.dumps(result,indent=2)+'\n')
        head=aws('head-object','--bucket','zensim-a','--key',key)
        if head['ContentLength']!=size or head.get('Metadata',{}).get('sha256')!=sha:raise SystemExit('Uploaded object identity differs')
        readback=a.archive.with_name('bundle.r2-readback.imijset')
        if readback.exists():raise SystemExit('Preserve existing readback; choose a fresh local artifact directory')
        aws('get-object','--bucket','zensim-a','--key',key,str(readback))
        if hashlib.sha256(readback.read_bytes()).hexdigest()!=sha:raise SystemExit('R2 full-object readback differs')
        result.update(status='HEAD and full-object readback verified')
        a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
