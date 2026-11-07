"""Install one checksummed partition capsule without overwriting worker inputs."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,tarfile
from pathlib import Path,PurePosixPath


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def install(study,partition,assignment,bundle,manifest,bundle_pin,manifest_pin,receipt,*,reuse_identical=False):
    if len(PurePosixPath(partition).parts)!=1 or partition in ('.','..'):raise ValueError('invalid partition')
    if assignment.parent!=study/'worker_assignments':raise ValueError('assignment outside study')
    allowed=lambda path:(path.is_relative_to(study/'comparators'/partition) or path.is_relative_to(study/'shapes'/partition) or path==assignment)
    return install_checked(bundle,manifest,bundle_pin,manifest_pin,receipt,allowed,reuse_identical=reuse_identical)


def install_shared(bundle,manifest,bundle_pin,manifest_pin,receipt):
    roots=(Path('/mnt/v/output/zensim'),Path('/srv/zen-research/human-ssim2-depth-20261106'),
           Path('/srv/zen-research/human-ssim2-trial-20261106'))
    return install_checked(bundle,manifest,bundle_pin,manifest_pin,receipt,
                           lambda path:any(path.is_relative_to(root) for root in roots))


def install_checked(bundle,manifest,bundle_pin,manifest_pin,receipt,allowed,*,reuse_identical=False):
    if receipt.exists():raise ValueError('preserve existing installation receipt')
    if sha(bundle)!=bundle_pin or sha(manifest)!=manifest_pin:raise ValueError('capsule checksum differs')
    rows=json.loads(manifest.read_text());expected={r['path'].lstrip('/'):r for r in rows}
    if len(expected)!=len(rows):raise ValueError('duplicate manifest path')
    reused=set()
    with tarfile.open(bundle) as archive:
        members=archive.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected):raise ValueError('capsule inventory differs')
        for member in members:
            row=expected[member.name];path=Path(row['path']);parts=PurePosixPath(member.name)
            if not path.is_absolute() or '..' in parts.parts or parts.as_posix()!=member.name:
                raise ValueError('unsafe input path')
            if not allowed(path):
                raise ValueError('input outside assigned partition')
            if not member.isfile() or member.size!=row['bytes']:raise ValueError('nonregular or wrong-sized input')
            if any(p.is_symlink() for p in path.parents):raise ValueError('symlink parent refused')
            if path.exists() or path.is_symlink():
                if not reuse_identical:raise ValueError('existing input preserved')
                if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                    raise ValueError('existing input differs; preserved')
                reused.add(str(path))
        for member in members:
            row=expected[member.name];path=Path(row['path'])
            if str(path) in reused:continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src,path.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
            if sha(path)!=row['sha256']:raise ValueError('installed input checksum differs')
            os.chmod(path,member.mode&0o777)
    for row in rows:
        path=Path(row['path'])
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('input changed during installation')
    result=dict(reused_identical_files=sorted(reused),files=len(rows),bytes=sum(r['bytes'] for r in rows),archive_sha256=bundle_pin,
                manifest_sha256=manifest_pin,all_inputs_verified=True,scientific_inputs_unchanged=True,
                driver_sha256=sha(__file__))
    with receipt.open('xb') as stream:stream.write((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','assignment'):p.add_argument('--'+name,type=Path)
    for name in ('bundle','manifest','receipt'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse-identical',action='store_true')
    p.add_argument('--partition');p.add_argument('--shared',action='store_true')
    for name in ('bundle-pin','manifest-pin'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.shared and a.reuse_identical:p.error('reuse-identical is restricted to partition inputs')
    if a.shared:install_shared(a.bundle,a.manifest,a.bundle_pin,a.manifest_pin,a.receipt)
    else:
        if not (a.study and a.partition and a.assignment):p.error('partition install requires study, partition and assignment')
        install(a.study,a.partition,a.assignment,a.bundle,a.manifest,a.bundle_pin,a.manifest_pin,a.receipt,reuse_identical=a.reuse_identical)
