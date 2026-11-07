"""Transfer terminal worker fits with a complete inventory and atomic publication."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import socket
import tarfile
from shaping_completion import encoded, sha


def inventory(root):
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('symlink in worker result')
        if path.is_dir():continue
        if not path.is_file():raise ValueError('nonregular worker result')
        result[path.relative_to(root).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path))
    return result


def validate(root, assignment, cells):
    run=json.loads((root/'RUN.json').read_text())
    result=json.loads((root/'RESULT.json').read_text())
    cell=run['cell']
    if cell not in cells or cell['id'] not in assignment['cells']:raise ValueError('unassigned cell')
    for key in ('cell','plan_sha256','build_sha256','execution_host','epochs','pairs_per_epoch'):
        if run[key]!=result[key]:raise ValueError('terminal receipt differs: '+key)
    for key in ('plan_sha256','build_sha256'):
        if run[key]!=assignment[key]:raise ValueError('worker scientific identity differs')
    if run['execution_host']!=assignment['host']:raise ValueError('worker host differs')
    if result['stage']!='trained' or result['exit_code']!=0:raise ValueError('worker fit not successful')
    epochs=120 if '_pretrained_' in cell['family'] or cell['stage']=='fresh_control_inner' else 240
    if run['epochs']!=epochs or run['pairs_per_epoch']!=50000:raise ValueError('exposure differs')
    expected={f'ckpt_epoch{i:03}.bin' for i in range(epochs)}
    if {p.name for p in (root/'checkpoints').glob('ckpt_epoch*.bin')}!=expected:raise ValueError('checkpoint coverage differs')
    if sha(root/'checkpoints'/f'ckpt_epoch{epochs-1:03}.bin')!=result['bake_sha256']:raise ValueError('final bake drift')
    return run


def pack(study,assignment_path,cell_id,attempt,bundle):
    from shaping_execution import load_registration
    _,cells,_=load_registration(study)
    assignment=json.loads(assignment_path.read_text())
    if assignment['host']!=socket.gethostname():raise ValueError('pack must run on assigned host')
    if assignment['plan_sha256']!=sha(study/'PLAN.json'):raise ValueError('assignment plan drift')
    if cell_id not in assignment['cells'] or not attempt.startswith('attempt-') or not attempt[8:].isdigit():
        raise ValueError('invalid cell or attempt')
    root=study/'runs'/cell_id/attempt
    run=validate(root,assignment,cells)
    if run['cell']['id']!=cell_id:raise ValueError('cell directory differs')
    # RESULT is written after wait() and closing train.log. No active process
    # or partial attempt is copied; downstream evaluation runs after collection.
    files=inventory(root)
    if sum(r['bytes'] for r in files.values())>1024**3:raise ValueError('bundle exceeds one-job storage allowance')
    receipt=dict(cell=run['cell'],attempt=attempt,assignment_sha256=sha(assignment_path),files=files)
    with tarfile.open(bundle,'x') as archive:
        import io
        data=encoded(receipt);info=tarfile.TarInfo('COLLECTION.json');info.size=len(data)
        archive.addfile(info,io.BytesIO(data))
        for name in files:archive.add(root/name,arcname='payload/'+name,recursive=False)
    if inventory(root)!=files:raise ValueError('worker output changed during packaging; preserve rejected bundle')
    receipt['bundle_sha256']=sha(bundle)
    with Path(str(bundle)+'.json').open('xb') as out:out.write(encoded(receipt))
    return receipt


def unpack(bundle,destination,expected):
    """No extraction API: reject links, traversal, duplicates and extra files."""
    if bundle.stat().st_size>2*1024**3:raise ValueError('bundle exceeds transport allowance')
    if sha(bundle)!=expected:raise ValueError('bundle checksum differs')
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(bundle,'r') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(set(names))!=len(names) or not members or names[0]!='COLLECTION.json':raise ValueError('ambiguous bundle')
        if any(not m.isfile() for m in members):raise ValueError('nonregular bundle member')
        receipt=json.load(archive.extractfile(members[0]));files=receipt['files']
        if sum(r['bytes'] for r in files.values())>1024**3:raise ValueError('bundle exceeds one-job storage allowance')
        if set(names)!={'COLLECTION.json'}|{'payload/'+p for p in files}:raise ValueError('bundle inventory differs')
        for name,row in files.items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or path.as_posix()!=name:raise ValueError('unsafe bundle path')
            member=archive.getmember('payload/'+name)
            if member.size!=row['bytes']:raise ValueError('member size differs')
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src,target.open('xb') as out:
                import shutil
                shutil.copyfileobj(src,out,1024*1024)
                out.flush();os.fsync(out.fileno())
            if sha(target)!=row['sha256']:raise ValueError('member checksum differs')
    return receipt


def collect(study,assignment_path,bundle,pin):
    from shaping_execution import load_registration
    from shaping_storage import reservation
    plan,cells,_=load_registration(study)
    assignment=json.loads(assignment_path.read_text())
    if assignment['plan_sha256']!=sha(study/'PLAN.json'):raise ValueError('assignment plan drift')
    # A hidden staging tree keeps the live evaluator from seeing partial files.
    staging=study/'worker_collection'/pin
    with reservation(study,staging,plan['minimum_ssd_free_bytes']):
        receipt=unpack(bundle,staging,pin)
        if receipt['assignment_sha256']!=sha(assignment_path):raise ValueError('assignment drift')
        run=validate(staging,assignment,cells)
        if run['cell']!=receipt['cell']:raise ValueError('collection cell differs')
        attempt=receipt['attempt']
        if not attempt.startswith('attempt-') or not attempt[8:].isdigit():raise ValueError('invalid attempt name')
        target=study/'runs'/run['cell']['id']
        # mkdir is an exclusive claim; an existing cell is never overwritten.
        target.mkdir(exist_ok=False)
        (staging/'WORKER_COLLECTION.json').write_bytes(encoded(dict(
            bundle_sha256=pin,assignment_sha256=sha(assignment_path),
            files=receipt['files'],source_host=assignment['host'],
            verified_bytes=sum(r['bytes'] for r in receipt['files'].values()))))
        staging.rename(target/attempt)
    return str(target/attempt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('pack','collect'))
    for name in ('study','assignment','bundle'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cell');p.add_argument('--attempt',default='attempt-001');p.add_argument('--sha256')
    a=p.parse_args()
    if a.action=='pack':
        if not a.cell:p.error('--cell is required')
        result=pack(a.study,a.assignment,a.cell,a.attempt,a.bundle)
    else:
        if not a.sha256:p.error('--sha256 is required')
        result=collect(a.study,a.assignment,a.bundle,a.sha256)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
