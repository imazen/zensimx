"""Reclaim duplicate transport parity arrays only after three-copy verification."""
from datetime import datetime,timezone
import json
from pathlib import Path
from shaping_completion import encoded
from shaping_reclaim_grams import digest,no_users


def main():
    base=Path('/mnt/v/output/zensim/shaping-completion-20261107')
    root=base/'tree-inner-archive'
    mirror=Path('/mnt/tower/output/zensim/shaping-completion-20261107/tree-inner-archive')
    output=base/'RECLAIMED_TREE_TRANSPORT.json'
    if output.exists():raise ValueError('preserve reclamation receipt')
    pin='2e7685d7746262d40a42a242358a1084f39c4d839e9944c3b5a72c71acb0213f'
    if digest(root/'SHA256SUMS')!=pin or digest(mirror/'SHA256SUMS')!=pin:
        raise ValueError('transport archive manifest differs')
    hashes={name:value for value,name in (line.split('  ',1) for line in (root/'SHA256SUMS').read_text().splitlines())}
    paths=sorted(root.glob('runs/*/attempt-001/model.parity.npz'))
    if len(paths)!=72:raise ValueError('duplicate parity inventory differs')
    plan=json.loads((base/'study/PLAN.json').read_text())
    if set(map(str,paths))&set(plan['input_pins']):raise ValueError('study pins transport copy')
    no_users(paths,root);entries=[]
    for index,path in enumerate(paths):
        rel=str(path.relative_to(root));copies=(path,mirror/rel,base/'study'/rel)
        if any(p.is_symlink() for p in copies):raise ValueError('symlink refused')
        if any(digest(p,p==mirror/rel)!=hashes[rel] for p in copies):raise ValueError('three-copy checksum differs: '+rel)
        entries.append(dict(relative_path=rel,sha256=hashes[rel],bytes=path.stat().st_size))
        print(json.dumps(dict(verified=index+1,total=len(paths))),flush=True)
    no_users(paths,root)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),stage='verified_before_removal',
                 local_root=str(root),durable_root=str(mirror),retained_original_root=str(base/'study'),
                 archive_manifest_sha256=pin,entries=entries,total_bytes=sum(e['bytes'] for e in entries),
                 scope='duplicate transport model.parity.npz only; original scientific runs retained',
                 readback='all three SHA-256 copies; durable client readback after POSIX_FADV_DONTNEED')
    output.write_bytes(encoded(receipt))
    for path in paths:path.unlink()
    receipt['stage']='duplicate_transport_arrays_removed';output.write_bytes(encoded(receipt))
    (root/'LOCAL_ARCHIVE_STATUS.json').write_bytes(encoded(dict(partial_local_copy=True,
        receipt=str(output),restoration_root=str(mirror),retained='all models, reports and original study arrays')))
    print(json.dumps(dict(removed_files=len(paths),removed_bytes=receipt['total_bytes'])),flush=True)


if __name__=='__main__':main()
