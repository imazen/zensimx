"""Remove only imported worker transport duplicates after durable cold readback."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shlex,subprocess
from pathlib import Path
from shaping_completion import encoded,sha

REMOTE=r'''
import hashlib,json,pathlib,os,re,shutil,subprocess,sys
p=json.loads(sys.argv[1]);root=pathlib.Path(p['root']);directory=root/'worker-bundles'
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
if p['mode']=='inventory':
 print(json.dumps({q.name:q.stat().st_size for q in directory.glob('*.tar') if not q.is_symlink()}));sys.exit(0)
receipt=root/p['receipt']
if receipt.exists():raise ValueError('preserve existing remote retention receipt')
paths=[];identities={}
for row in p['rows']:
 name=row['name']
 if not re.fullmatch(r'[a-f0-9]{20}-attempt-001.tar',name):raise ValueError('unregistered bundle name')
 path=directory/name
 if path.resolve()!=path or path in paths:raise ValueError('symlink or duplicate bundle')
 paths.append(path)
def no_users():
 result=subprocess.run(['sudo','lsof','-nP','-t','--',*[str(q) for q in paths]],capture_output=True,text=True)
 if result.returncode not in (0,1) or result.stderr or result.stdout:raise ValueError('bundle in use or open-file inspection failed: '+result.stderr+result.stdout)
no_users()
for row,path in zip(p['rows'],paths):
 stat=path.stat();identities[str(path)]=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
 if stat.st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('worker bundle drift')
 sidecar=pathlib.Path(str(path)+'.json')
 if digest(sidecar)!=row['producer_sha256']:raise ValueError('worker bundle receipt drift')
no_users()
for path in paths:
 stat=path.stat()
 if identities[str(path)]!=(stat.st_ino,stat.st_size,stat.st_mtime_ns):raise ValueError('bundle changed during verification')
record=dict(stage='verified_before_removal',rows=p['rows'],free_before_bytes=shutil.disk_usage(root).free)
with receipt.open('x') as stream:json.dump(record,stream,indent=2)
for path in paths:path.unlink()
record.update(stage='duplicates_removed',free_after_bytes=shutil.disk_usage(root).free,receipt=str(receipt))
receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''


def remote(host,payload):
    if host not in ('i265','i270','r3800x','i134','r5600g','r3500'):raise ValueError('unregistered worker')
    command='TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- '+shlex.join(['python3','-c',REMOTE,json.dumps(payload)])
    return json.loads(subprocess.check_output(['ssh',host,command],text=True))


def cold_sha(path):
    with path.open('rb') as stream:
        os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        return hashlib.file_digest(stream,'sha256').hexdigest()


def plan(study,archive,host,inventory):
    rows=[]
    for name,size in sorted(inventory.items()):
        if not re.fullmatch(r'[a-f0-9]{20}-attempt-001.tar',name):continue
        cell=name[:20];bundle=archive/(host+'-'+name[:-4])/name
        if not bundle.exists():continue
        sidecar=Path(str(bundle)+'.json');verified=Path(str(bundle)+'.verified.json')
        collection=study/'runs'/cell/'attempt-001/WORKER_COLLECTION.json'
        if not verified.exists() or not collection.exists():continue
        v=json.loads(verified.read_text());c=json.loads(collection.read_text());producer=json.loads(sidecar.read_text())
        if (v['bundle']!=str(bundle) or v['bundle_bytes']!=size or bundle.stat().st_size!=size
            or c['source_host']!=host or producer['cell']['id']!=cell or producer['attempt']!='attempt-001'
            or not v['bundle_sha256']==c['bundle_sha256']==producer['bundle_sha256']
            or not v['assignment_sha256']==c['assignment_sha256']==producer['assignment_sha256']
            or v['producer_receipt_sha256']!=sha(sidecar)):
            raise ValueError('worker archive/import provenance differs')
        result=json.loads(collection.with_name('RESULT.json').read_text())
        if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('imported fit is not complete')
        if cold_sha(bundle)!=v['bundle_sha256']:raise ValueError('durable worker bundle readback differs')
        rows.append(dict(name=name,bytes=size,sha256=v['bundle_sha256'],producer_sha256=sha(sidecar),
                         durable_path=str(bundle),import_receipt=str(collection),import_receipt_sha256=sha(collection)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True)
    for key in ('study','archive','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing retention receipt')
    root=a.study.parent;inventory=remote(a.host,dict(mode='inventory',root=str(root)))
    rows=plan(a.study,a.archive,a.host,inventory)
    if not rows:raise ValueError('no imported durable bundle duplicates')
    receipt='RECLAIMED_TRANSPORT_'+a.output.stem+'.json'
    record=dict(host=a.host,rows=rows,stage='durable_readback_verified',driver_sha256=sha(__file__),original_fit_directories_preserved=True)
    with a.output.open('xb') as stream:stream.write(encoded(record))
    if a.apply:
        result=remote(a.host,dict(mode='apply',root=str(root),receipt=receipt,rows=rows))
        record.update(stage='worker_duplicates_removed',remote_result=result);a.output.write_bytes(encoded(record))
    print(json.dumps(dict(stage=record['stage'],files=len(rows),bytes=sum(r['bytes'] for r in rows))),flush=True)


if __name__=='__main__':main()
