"""Retire only byte-identical, archived canonical comparison duplicates."""
from __future__ import annotations
import argparse,fcntl,json,os,socket,subprocess,time
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_partial_restart import CELLS,authorize

RECEIPT='d6fa13e6a927ce474d7a5a247af0d508e511337470f40ab98f9ec7a5c8a87271'


def atomic(path,value):
    pending=path.with_suffix('.writing')
    with pending.open('wb') as out:
        out.write(encoded(value));out.flush();os.fsync(out.fileno())
    pending.replace(path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare(study,archive,output):
    filesystem=subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(archive)],text=True).strip()
    if filesystem not in ('nfs','nfs4'):raise ValueError('mounted durable archive required')
    if sha(archive/'MIRROR_VERIFIED.json')!=RECEIPT:raise ValueError('original archive receipt differs')
    rows={};total=0
    for cell in CELLS:
        row,pin=authorize(study,cell,'attempt-002');files={}
        for name,expected in row['checkpoint_sha256'].items():
            path=archive/'runs'/cell/'attempt-001/checkpoints'/name
            if path.is_symlink() or not path.is_file():raise ValueError('archive checkpoint is not regular')
            with path.open('rb') as stream:
                os.fsync(stream.fileno());os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            if sha(path)!=expected:raise ValueError('archive checkpoint cold readback differs')
            files[name]=dict(sha256=expected,bytes=path.stat().st_size,archive_path=str(path))
            total+=path.stat().st_size
        rows[cell]=dict(authorization_sha256=pin,files=files)
    record=dict(schema='shaping-prefix-archive-proof-v1',archive_receipt_sha256=RECEIPT,
                archive_manifest_sha256=sha(archive/'SHA256SUMS'),cells=rows,
                verified_bytes=total,readback='fsync, POSIX_FADV_DONTNEED, complete SHA-256')
    with output.open('xb') as out:out.write(encoded(record));out.flush();os.fsync(out.fileno())
    return record


def no_open_users(paths):
    if not paths:return
    result=subprocess.run(['sudo','-n','lsof','-t','--',*map(str,paths)],capture_output=True,text=True)
    if result.returncode!=1 or result.stdout.strip():
        raise ValueError('canonical duplicates have open users or lsof failed: '+result.stdout+result.stderr)


def retire(study,cell,proof_path):
    if socket.gethostname()!='i134':raise ValueError('approved worker required')
    root=study/'runs'/cell/'attempt-002'
    with (root/'.prefix-retirement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return retire_locked(study,cell,proof_path)


def retire_locked(study,cell,proof_path):
    row,pin=authorize(study,cell,'attempt-002');proof=json.loads(proof_path.read_text())
    evidence=proof['cells'][cell]
    if (proof['schema']!='shaping-prefix-archive-proof-v1' or proof['archive_receipt_sha256']!=RECEIPT
        or evidence['authorization_sha256']!=pin
        or {k:v['sha256'] for k,v in evidence['files'].items()}!=row['checkpoint_sha256']):
        raise ValueError('archive proof does not cover the approved original checkpoints')
    root=study/'runs'/cell/'attempt-002';comparison=root/'PREFIX_COMPARISON_V2.json'
    local_proof=root/'PREFIX_ARCHIVE_PROOF.json'
    if local_proof.exists():
        if sha(local_proof)!=sha(proof_path):raise ValueError('local archive proof drift')
    else:
        with local_proof.open('xb') as out:out.write(proof_path.read_bytes());out.flush();os.fsync(out.fileno())
    report=json.loads(comparison.read_text());path=root/'PREFIX_DUPLICATE_RETIREMENT.json'
    record=json.loads(path.read_text()) if path.exists() else dict(
        schema='shaping-prefix-duplicate-retirement-v1',proof_sha256=sha(proof_path),files={})
    if record['proof_sha256']!=sha(proof_path):raise ValueError('retirement proof drift')
    candidates=[]
    for name,values in report['compared'].items():
        expected=evidence['files'][name]
        if not values['equal'] or any(values[k]!=expected['sha256'] for k in ('original_sha256','restart_sha256')):
            raise ValueError('canonical checkpoint is not identical to the archived original')
        for label in ('original','restart'):
            relative='prefix_comparison_v2/'+label+'-'+name;target=root/relative
            if target.is_symlink():raise ValueError('symlink in canonical duplicates')
            if target.exists():
                if sha(target)!=expected['sha256'] or target.stat().st_size!=expected['bytes']:
                    raise ValueError('canonical duplicate drift')
                candidates.append(target)
            elif relative not in record['files']:raise ValueError('unrecorded missing canonical duplicate')
            record['files'][relative]=expected
    no_open_users(candidates)
    # The comparison index publishes a name only after both native processes
    # have closed their files. Its owner never revisits completed names.
    # Persist retirement intent first: a crash can safely finish these unlinks.
    atomic(path,record)
    for target in candidates:target.unlink()
    return dict(compared=len(report['compared']),expected=report['expected'],
                retired_files=len(record['files']),retired_bytes=sum(v['bytes'] for v in record['files'].values()),
                proof_sha256=record['proof_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','watch'))
    p.add_argument('--study',type=Path,required=True);p.add_argument('--proof',type=Path,required=True)
    p.add_argument('--archive',type=Path);p.add_argument('--cell',choices=CELLS)
    args=p.parse_args()
    if args.mode=='prepare':print(json.dumps(prepare(args.study,args.archive,args.proof)));return
    if not args.cell:p.error('--cell required for watch')
    while True:
        try:record=retire(args.study,args.cell,args.proof)
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
        print(json.dumps(record),flush=True)
        if record['compared']==record['expected']:break
        time.sleep(30)


if __name__=='__main__':main()
