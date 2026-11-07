"""Narrow, authorized reclamation of old completed quadratic Gram outputs only."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

ROOT=Path('/mnt/v/output/zensim/shaped-linear-generalization-20261106')
MIRROR=Path('/mnt/tower/output/zensim/shaped-linear-generalization-20261106')
RECEIPT=Path('/mnt/v/output/zensim/shaping-completion-20261107/RECLAIMED_OLD_GRAMS.json')


def digest(path,cold=False):
    with path.open('rb') as f:
        if cold:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        return hashlib.file_digest(f,'sha256').hexdigest()


def no_users(paths):
    ancestors=set();pid=os.getpid()
    while pid>1:
        ancestors.add(str(pid))
        pid=int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('PPid:')))
    targets={str(p) for p in paths}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or proc.name in ancestors:continue
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if str(ROOT/'fits') in cmd:raise ValueError('old fit process is active: '+proc.name)
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target in targets:raise ValueError('open Gram file: '+target)
            maps=(proc/'maps').read_text()
            if any(path in maps for path in targets):raise ValueError('mapped Gram file: '+proc.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue


def main():
    if RECEIPT.exists():raise ValueError('preserve existing reclamation receipt')
    manifest=ROOT/'SHA256SUMS';pin=digest(manifest)
    if pin!='642428868aada79b69ee4358b14e01a279b7c068e6dd1f0bdea2e478de7b8882' or digest(MIRROR/'SHA256SUMS')!=pin:
        raise ValueError('historical archive manifest differs')
    hashes={name:value for value,name in (line.split('  ',1) for line in manifest.read_text().splitlines())}
    paths=sorted((ROOT/'fits').glob('*/gram.npz'))
    if len(paths)!=264:raise ValueError('old Gram inventory differs')
    plan=json.loads((RECEIPT.parent/'study/PLAN.json').read_text())
    if set(map(str,paths))&set(plan['input_pins']):raise ValueError('active study pins a Gram')
    no_users(paths);entries=[]
    for i,path in enumerate(paths):
        rel=str(path.relative_to(ROOT));expected=hashes[rel];mirror=MIRROR/rel
        if path.is_symlink() or mirror.is_symlink():raise ValueError('symlink refused')
        if digest(path)!=expected or digest(mirror,True)!=expected:raise ValueError('Gram mismatch: '+rel)
        entries.append(dict(relative_path=rel,sha256=expected,bytes=path.stat().st_size))
        if i%32==0:print(json.dumps(dict(verified=i+1,total=len(paths))),flush=True)
    no_users(paths)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),stage='verified_before_removal',
                 local_root=str(ROOT),durable_root=str(MIRROR),archive_manifest_sha256=pin,
                 readback='client SHA-256 after POSIX_FADV_DONTNEED',entries=entries,
                 total_bytes=sum(e['bytes'] for e in entries),scope='only fits/*/gram.npz; original corpus and all other results retained')
    RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')
    for path in paths:path.unlink()
    receipt['stage']='local_grams_removed';RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'LOCAL_ARCHIVE_STATUS.json').write_text(json.dumps(dict(partial_local_copy=True,
        retained='reports, models, predictions, fit specifications and active input arrays',
        removed_pattern='fits/*/gram.npz',restoration_root=str(MIRROR),receipt=str(RECEIPT)),indent=2)+'\n')
    print(json.dumps(dict(removed_files=len(paths),removed_bytes=receipt['total_bytes'],receipt=str(RECEIPT))),flush=True)


if __name__=='__main__':main()
