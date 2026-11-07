"""Reclaim only checksum-verified historical tree parity input NPZ copies."""
from datetime import datetime,timezone
import json
from pathlib import Path
from shaping_reclaim_grams import digest,no_users


def main():
    root=Path('/mnt/v/output/zensim/tree-study-20261106')
    mirror=Path('/mnt/tower/output/zensim/tree-study-20261106')
    output=Path('/mnt/v/output/zensim/shaping-completion-20261107/RECLAIMED_TREE_PARITY.json')
    if output.exists():raise ValueError('preserve existing reclamation receipt')
    expected='c98912ff5e927496899a5f6d4bd5aaa8a60f4b331d5b1fa483966ce1ac35ecad'
    if digest(root/'SHA256SUMS')!=expected or digest(mirror/'SHA256SUMS')!=expected:
        raise ValueError('historical archive identity differs')
    hashes={name:value for value,name in (line.split('  ',1) for line in (root/'SHA256SUMS').read_text().splitlines())}
    paths=sorted(root.rglob('*.parity.npz'))
    if len(paths)!=60 or any(p.relative_to(root).parts[0] not in ('search','peer-search','constraint-controls') for p in paths):
        raise ValueError('historical parity inventory differs')
    plan=json.loads((output.parent/'study/PLAN.json').read_text())
    if set(map(str,paths))&set(plan['input_pins']):raise ValueError('active study pins a parity input')
    no_users(paths,root);entries=[]
    for i,path in enumerate(paths):
        rel=str(path.relative_to(root));remote=mirror/rel
        if path.is_symlink() or remote.is_symlink():raise ValueError('symlink refused')
        if digest(path)!=hashes[rel] or digest(remote,True)!=hashes[rel]:raise ValueError('parity input mismatch: '+rel)
        entries.append(dict(relative_path=rel,sha256=hashes[rel],bytes=path.stat().st_size))
        print(json.dumps(dict(verified=i+1,total=len(paths))),flush=True)
    no_users(paths,root)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),stage='verified_before_removal',
                 local_root=str(root),durable_root=str(mirror),archive_manifest_sha256=expected,
                 readback='client SHA-256 after POSIX_FADV_DONTNEED',entries=entries,
                 total_bytes=sum(e['bytes'] for e in entries))
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    for path in paths:path.unlink()
    receipt['stage']='local_parity_inputs_removed';output.write_text(json.dumps(receipt,indent=2)+'\n')
    (root/'LOCAL_ARCHIVE_STATUS.json').write_text(json.dumps(dict(partial_local_copy=True,
        retained='models, reports, predictions, specifications, and all active input arrays',
        removed_pattern='*.parity.npz',restoration_root=str(mirror),receipt=str(output)),indent=2)+'\n')
    print(json.dumps(dict(removed_files=len(paths),removed_bytes=receipt['total_bytes'])),flush=True)


if __name__=='__main__':main()
