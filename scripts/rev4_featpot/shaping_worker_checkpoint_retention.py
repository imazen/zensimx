"""Reclaim imported worker checkpoints only after durable bundle readback."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def selected(files,epochs,keep):
    if epochs not in (120,240):raise ValueError('unregistered worker phase')
    expected={f'checkpoints/ckpt_epoch{i:03}.bin' for i in range(epochs)}
    actual={n for n in files if n.startswith('checkpoints/') and n.endswith('.bin')}
    if actual!=expected:raise ValueError('incomplete checkpoint inventory')
    required={0,119,epochs-1,*[e for e in (49,99,119,149,199,219,239) if e<epochs]}
    if not required<=set(keep):raise ValueError('required epochs missing')
    return {n:files[n] for i in range(epochs) if i not in keep
            for n in [f'checkpoints/ckpt_epoch{i:03}.bin']}


def no_users(paths):
    # Exact filenames avoid unrelated stale NFS mounts and host-specific portals.
    # Bound argv size independently of the number of completed fits.
    for start in range(0,len(paths),256):
        result=subprocess.run(['sudo','lsof','-nP','-t','--',*[str(p) for p in paths[start:start+256]]],capture_output=True,text=True)
        if result.returncode not in (0,1) or result.stderr:raise ValueError('open-file inspection failed: '+result.stderr)
        if result.stdout:raise ValueError('checkpoint in use')


def remove_remote(payload):
    root=Path(payload['study']);output=Path(payload['receipt'])
    if output.exists():raise ValueError('preserve retention receipt')
    if root.resolve()!=root or output.parent!=root.parent:raise ValueError('unsafe retention root')
    plan=json.loads((root/'PLAN.json').read_text());pins=plan['input_pins']
    if sha(root/'PLAN.json')!=payload['plan_sha256']:raise ValueError('plan drift')
    paths=[];identities={}
    for row in payload['cells']:
        if not re.fullmatch('[a-f0-9]{20}',row['cell']):raise ValueError('unsafe cell')
        run=root/'runs'/row['cell']/'attempt-001'
        if list(run.parent.glob('attempt-*'))!=[run]:raise ValueError('multiple attempts')
        if set(row['metadata'])!={'RUN.json','RESULT.json'}:raise ValueError('missing metadata pins')
        for name,pin in row['metadata'].items():
            if name not in ('RUN.json','RESULT.json') or sha(run/name)!=pin:raise ValueError('metadata drift')
        result=json.loads((run/'RESULT.json').read_text())
        if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('unfinished fit')
        if result['cell']['id']!=row['cell'] or result['epochs']!=row['epochs']:raise ValueError('fit identity differs')
        files=selected(row['files'],row['epochs'],row['keep'])
        if not files:raise ValueError('empty retention')
        for name,entry in files.items():
            path=run/name
            if path.resolve()!=path or path.is_symlink() or str(path) in pins:raise ValueError('protected path')
            st=path.stat()
            if st.st_nlink!=1 or st.st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('checkpoint drift')
            if path in identities:raise ValueError('duplicate removal')
            identities[path]=(st.st_ino,st.st_size,st.st_mtime_ns);paths.append(path)
    no_users(paths)
    for path,identity in identities.items():
        st=path.stat()
        if (st.st_ino,st.st_size,st.st_mtime_ns)!=identity:raise ValueError('checkpoint changed')
    record=dict(payload,stage='verified_before_removal',free_before_bytes=shutil.disk_usage(root).free)
    with output.open('x') as stream:json.dump(record,stream,indent=2)
    no_users(paths)
    for path in paths:path.unlink()
    record.update(stage='duplicates_removed',removed_files=len(paths),removed_bytes=sum(v[1] for v in identities.values()),free_after_bytes=shutil.disk_usage(root).free)
    output.write_text(json.dumps(record,indent=2)+'\n')
    return record


def prior_cells(study,host,receipts):
    excluded=set();pins=[]
    for path in receipts:
        record=json.loads(path.read_text())
        if record['stage']!='worker_duplicates_removed' or record['host']!=host:raise ValueError('completed matching retention receipt required')
        payload=record['payload'];result=record['result']
        if payload['study']!=str(study) or payload['plan_sha256']!=sha(study/'PLAN.json'):
            raise ValueError('prior retention scope differs')
        if result['stage']!='duplicates_removed' or any(result.get(k)!=v for k,v in payload.items()):
            raise ValueError('prior worker receipt differs')
        removed={};expected_bytes=0
        for row in payload['cells']:
            if row['cell'] in removed:raise ValueError('duplicate prior cell')
            files=selected(row['files'],row['epochs'],row['keep'])
            removed[row['cell']]=len(files);expected_bytes+=sum(f['bytes'] for f in files.values())
        if result['removed_files']!=sum(removed.values()) or result['removed_bytes']!=expected_bytes:
            raise ValueError('prior removal totals differ')
        excluded.update(removed)
        pins.append(dict(path=str(path),sha256=sha(path),cells=sorted(removed)))
    return excluded,pins


def make_plan(study,archive,host,remote_receipt,excluded=()):
    from shaping_checkpoint_retention import retained_epochs
    rows=[]
    for collection in sorted(study.glob('runs/*/attempt-001/WORKER_COLLECTION.json')):
        c=json.loads(collection.read_text())
        if c['source_host']!=host:continue
        run=collection.parent;cell=run.parent.name
        if cell in excluded:continue
        bundle=archive/f'{host}-{cell}-attempt-001'/f'{cell}-attempt-001.tar'
        verified=Path(str(bundle)+'.verified.json')
        if not verified.exists():continue
        v=json.loads(verified.read_text())
        if v['bundle_sha256']!=c['bundle_sha256'] or v['bundle']!=str(bundle):raise ValueError('archive/import drift')
        with bundle.open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            if hashlib.file_digest(stream,'sha256').hexdigest()!=c['bundle_sha256']:raise ValueError('durable bundle drift')
        with tarfile.open(bundle) as tar:
            manifest=json.load(tar.extractfile('COLLECTION.json'))
        if manifest['files']!=c['files'] or manifest['cell']['id']!=cell:raise ValueError('bundle inventory drift')
        result=json.loads((run/'RESULT.json').read_text())
        if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('unfinished imported fit')
        metadata={n:c['files'][n]['sha256'] for n in ('RUN.json','RESULT.json')}
        if any(sha(run/n)!=pin for n,pin in metadata.items()):raise ValueError('import metadata drift')
        epochs=result['epochs'];keep=sorted(retained_epochs(epochs));selected(c['files'],epochs,keep)
        rows.append(dict(cell=cell,epochs=epochs,keep=keep,files=c['files'],metadata=metadata,
                         durable_bundle=str(bundle),bundle_sha256=c['bundle_sha256'],import_receipt_sha256=sha(collection)))
    return dict(study=str(study),receipt=str(remote_receipt),plan_sha256=sha(study/'PLAN.json'),cells=rows)


def main():
    if sys.argv[1:]==['--remote']:
        print(json.dumps(remove_remote(json.load(sys.stdin))));return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['i265','i270','r3800x','i134','r5600g','r3500'],required=True)
    for key in ['study','archive','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--prior-receipt',type=Path,action='append',default=[])
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing output')
    excluded,prior=prior_cells(a.study,a.host,a.prior_receipt)
    payload=make_plan(a.study,a.archive,a.host,a.study.parent/('WORKER_'+a.output.name),excluded)
    record=dict(stage='durable_readback_verified',host=a.host,payload=payload,driver_sha256=sha(__file__),prior_receipts=prior)
    if any(sha(row['path'])!=row['sha256'] for row in prior):raise ValueError('prior receipt changed')
    with a.output.open('x') as stream:json.dump(record,stream,indent=2)
    if a.apply:
        command='TMPDIR="$HOME/tmp" "$HOME/work/claudehints/scripts/run-heavy" --mem 1G --jobs 1 -- '+shlex.join(['python3','-c',Path(__file__).read_text(),'--remote'])
        result=subprocess.run(['ssh',a.host,command],input=json.dumps(payload),text=True,capture_output=True)
        Path(str(a.output)+'.remote.log').write_text(result.stderr)
        if result.returncode:raise RuntimeError('worker verification failed; see '+str(a.output)+'.remote.log')
        record.update(stage='worker_duplicates_removed',result=json.loads(result.stdout))
        a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(stage=record['stage'],cells=len(payload['cells']),removed_files=record.get('result',{}).get('removed_files'))))


if __name__=='__main__':main()
