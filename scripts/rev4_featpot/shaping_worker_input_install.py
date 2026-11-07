"""Install one checksummed partition capsule without overwriting worker inputs."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,tarfile
from pathlib import Path,PurePosixPath


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def install(study,partition,assignment,bundle,manifest,bundle_pin,manifest_pin,receipt):
    if receipt.exists():raise ValueError('preserve existing installation receipt')
    if sha(bundle)!=bundle_pin or sha(manifest)!=manifest_pin:raise ValueError('capsule checksum differs')
    rows=json.loads(manifest.read_text());expected={r['path'].lstrip('/'):r for r in rows}
    if len(expected)!=len(rows):raise ValueError('duplicate manifest path')
    if len(PurePosixPath(partition).parts)!=1 or partition in ('.','..'):raise ValueError('invalid partition')
    if assignment.parent!=study/'worker_assignments':raise ValueError('assignment outside study')
    with tarfile.open(bundle) as archive:
        members=archive.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise ValueError('capsule inventory differs')
        for member in members:
            row=expected[member.name];path=Path(row['path']);parts=PurePosixPath(member.name)
            if not path.is_absolute() or '..' in parts.parts or parts.as_posix()!=member.name:
                raise ValueError('unsafe input path')
            if not (path.is_relative_to(study/'comparators'/partition) or path.is_relative_to(study/'shapes'/partition) or path==assignment):
                raise ValueError('input outside assigned partition')
            if not member.isfile() or member.size!=row['bytes']:raise ValueError('nonregular or wrong-sized input')
            if path.exists() or path.is_symlink():raise ValueError('existing input preserved')
            if any(p.is_symlink() for p in path.parents):raise ValueError('symlink parent refused')
        for member in members:
            row=expected[member.name];path=Path(row['path']);path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src,path.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
            if sha(path)!=row['sha256']:raise ValueError('installed input checksum differs')
    result=dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),archive_sha256=bundle_pin,
                manifest_sha256=manifest_pin,all_inputs_verified=True,scientific_inputs_unchanged=True,
                driver_sha256=sha(__file__))
    with receipt.open('xb') as stream:stream.write((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','assignment','bundle','manifest','receipt'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('partition','bundle-pin','manifest-pin'):p.add_argument('--'+name,required=True)
    a=p.parse_args();install(a.study,a.partition,a.assignment,a.bundle,a.manifest,a.bundle_pin,a.manifest_pin,a.receipt)
