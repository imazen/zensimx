"""Bind live native snapshot stores and retire only verified completed copies."""
import argparse,fcntl,json,os,re,shutil
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_native_observe import observe

STATES={'state-0.snap','state-1.snap'}


def regular(path):
    if path.resolve()!=path or path.is_symlink() or not path.is_file():raise ValueError('regular canonical file required: '+str(path))


@contextmanager
def closed_store(directory):
    if (directory.resolve()!=directory or directory.parent.name!='shared-training-snapshots'
        or not re.fullmatch('[0-9a-f]{64}',directory.name)):raise ValueError('canonical shared snapshot identity required')
    global_path=directory.parent/'.lock';regular(global_path)
    with global_path.open('r+b') as global_lock:
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        local_path=directory/'.lock';regular(local_path)
        with local_path.open('r+b') as local_lock:
            fcntl.flock(local_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if {p.name for p in directory.iterdir()}-STATES-{'.lock'}:
                raise ValueError('unfinished write or unknown snapshot file; preserve store')
            yield


def bind(scope,run,output):
    if output!=run/'SNAPSHOT_BINDING.json' or output.exists():raise ValueError('one preserved binding beside the run required')
    rows=[r for r in observe([scope])['native_jobs'] if r['run']==str(run)]
    if len(rows)!=1 or not rows[0]['snapshot_directory']:raise ValueError('one live native snapshot-lock owner required')
    row=rows[0];directory=Path(row['snapshot_directory']);regular(run/'RUN.json')
    identity=json.loads((run/'RUN.json').read_text())
    if identity['stage']!='running' or identity['epochs'] not in (120,240):raise ValueError('active registered native exposure required')
    record=dict(schema='native-snapshot-run-binding-v1',run=str(run),run_sha256=sha(run/'RUN.json'),
        snapshot_directory=str(directory),native_observation=row,host=__import__('socket').gethostname(),
        checked_utc=datetime.now(timezone.utc).isoformat(),driver_sha256=sha(__file__))
    with output.open('xb') as stream:stream.write(encoded(record))
    return record


def wait_and_bind(process,scope,run):
    """Observe the owned native child while waiting; never signal or replay it."""
    import subprocess
    output=run/'SNAPSHOT_BINDING.json'
    while process.poll() is None and not output.exists():
        try:bind(scope,run,output)
        except ValueError as error:
            if str(error)!='one live native snapshot-lock owner required':
                (run/'SNAPSHOT_BINDING_FAILURE.json').write_bytes(encoded(dict(reason=str(error),
                    native_process_preserved=True,driver_sha256=sha(__file__))))
                break
        if not output.exists():
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
    code=process.wait()
    if not output.exists() and not (run/'SNAPSHOT_BINDING_FAILURE.json').exists():
        (run/'SNAPSHOT_BINDING_FAILURE.json').write_bytes(encoded(dict(
            reason='native process ended before snapshot-lock binding',native_exit_code=code,
            native_process_preserved=True,driver_sha256=sha(__file__))))
    return code


def completed(binding):
    record=json.loads(binding.read_text());run=Path(record['run']);directory=Path(record['snapshot_directory'])
    if (record['schema']!='native-snapshot-run-binding-v1' or sha(run/'RUN.json')!=record['run_sha256']
        or record['native_observation']['run']!=str(run)
        or record['native_observation']['snapshot_directory']!=str(directory)
        or record['host']!=__import__('socket').gethostname()):raise ValueError('snapshot binding identity differs')
    result=json.loads((run/'RESULT.json').read_text());identity=json.loads((run/'RUN.json').read_text())
    if (result.get('stage') not in ('trained','qualified') or result.get('exit_code')!=0
        or result['cell']!=identity['cell'] or result['epochs']!=identity['epochs']
        or result['build_sha256']!=identity['build_sha256']):raise ValueError('successful completed native run required; interrupted states stay protected')
    return record,run,directory


def stage(binding,output):
    record,run,directory=completed(binding)
    if output.exists():raise ValueError('preserve existing snapshot archive stage')
    if output.parent.stat().st_dev!=directory.stat().st_dev:raise ValueError('stage snapshots on the active SSD')
    with closed_store(directory):
        states=sorted(p for p in directory.iterdir() if p.name in STATES)
        if not states:raise ValueError('no retained snapshots')
        output.mkdir();pins={}
        for source in [binding,run/'RUN.json',run/'RESULT.json',*states]:
            regular(source);pin=sha(source);destination=output/source.name
            shutil.copyfile(source,destination)
            if sha(source)!=pin or sha(destination)!=pin:raise ValueError('snapshot stage identity changed')
            pins[source.name]=pin
        manifest=dict(schema='completed-native-snapshot-stage-v1',binding_sha256=sha(binding),
            snapshot_directory=str(directory),files=pins,driver_sha256=sha(__file__))
        (output/'SNAPSHOTS.json').write_bytes(encoded(manifest));return manifest


def retire(binding,stage_root,receipt):
    from shaping_checkpoint_retention import archive_inventory,no_users
    record,run,directory=completed(binding);manifest=json.loads((stage_root/'SNAPSHOTS.json').read_text())
    if (manifest['schema']!='completed-native-snapshot-stage-v1' or manifest['binding_sha256']!=sha(binding)
        or manifest['snapshot_directory']!=str(directory)):raise ValueError('archived snapshot binding differs')
    mirror,hashes=archive_inventory(stage_root)
    if mirror.resolve()==stage_root.resolve():raise ValueError('distinct durable snapshot archive required')
    if receipt.exists():raise ValueError('preserve previous retirement receipt; reconcile partial journal explicitly')
    with closed_store(directory):
        files=sorted(p for p in directory.iterdir() if p.name in STATES)
        if {p.name for p in files}!=STATES&set(manifest['files']):raise ValueError('snapshot inventory changed after archival')
        rows=[]
        for path in files:
            regular(path);remote=mirror/path.name;regular(remote)
            pin=manifest['files'][path.name]
            if hashes.get(path.name)!=pin or sha(path)!=pin:raise ValueError('local or archived snapshot manifest drift')
            with remote.open('rb') as stream:
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                cold=__import__('hashlib').file_digest(stream,'sha256').hexdigest()
            if cold!=pin:raise ValueError('durable snapshot cold readback differs')
            rows.append(dict(path=str(path),durable_path=str(remote),sha256=pin,bytes=path.stat().st_size))
        if not rows:raise ValueError('no completed snapshots to retire')
        no_users(rows)
        journal=dict(schema='completed-native-snapshot-retirement-v1',stage='verified_before_removal',
            snapshot_directory=str(directory),binding_sha256=sha(binding),archive_manifest_sha256=sha(stage_root/'SHA256SUMS'),
            rows=rows,driver_sha256=sha(__file__),checked_utc=datetime.now(timezone.utc).isoformat())
        with receipt.open('xb') as stream:stream.write(encoded(journal));stream.flush();os.fsync(stream.fileno())
        for row in rows:Path(row['path']).unlink()
        (directory/'.lock').unlink();directory.rmdir()
        # Removing only state files would retain the native 64-MiB reservation.
        parent_fd=os.open(directory.parent,os.O_RDONLY)
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
        journal['stage']='completed_store_retired';receipt.write_bytes(encoded(journal));return journal


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('bind','stage','retire'))
    for name in ('scope','run','binding','stage-root','output'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.mode=='bind':result=bind(args.scope,args.run,args.output)
    elif args.mode=='stage':result=stage(args.binding,args.output)
    else:result=retire(args.binding,args.stage_root,args.output)
    print(json.dumps(result),flush=True)
