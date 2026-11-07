"""Reclaim duplicate transport parity arrays only after three-copy verification."""
from datetime import datetime,timezone
import argparse,json,shutil
from pathlib import Path
from shaping_completion import encoded
from shaping_reclaim_grams import digest,no_users


def main(originals=False):
    base=Path('/mnt/v/output/zensim/shaping-completion-20261107')
    archive=base/'tree-inner-archive'
    root=base/'study' if originals else archive
    mirror=Path('/mnt/tower/output/zensim/shaping-completion-20261107/tree-inner-archive')
    output=base/('RECLAIMED_COMPLETED_TREE_PARITY.json' if originals else 'RECLAIMED_TREE_TRANSPORT.json')
    if output.exists():raise ValueError('preserve reclamation receipt')
    pin='2e7685d7746262d40a42a242358a1084f39c4d839e9944c3b5a72c71acb0213f'
    if digest(archive/'SHA256SUMS')!=pin or digest(mirror/'SHA256SUMS')!=pin:
        raise ValueError('transport archive manifest differs')
    hashes={name:value for value,name in (line.split('  ',1) for line in (archive/'SHA256SUMS').read_text().splitlines())}
    paths=sorted(root.glob('runs/*/attempt-001/model.parity.npz'))
    if len(paths)!=72:raise ValueError('duplicate parity inventory differs')
    plan=json.loads((base/'study/PLAN.json').read_text())
    if set(map(str,paths))&set(plan['input_pins']):raise ValueError('study pins transport copy')
    no_users(paths,archive);entries=[]
    for index,path in enumerate(paths):
        rel=str(path.relative_to(root));copies=(path,mirror/rel) if originals else (path,mirror/rel,base/'study'/rel)
        if originals:
            result=json.loads((path.parent/'RESULT.json').read_text())
            if result['cell']['family']!='extra_trees' or result['stage']!='complete' or result['exit_code']!=0:
                raise ValueError('tree fit is not terminal')
            report=path.parent/'evaluation/REPORT.json';marker=report.with_name('COMPLETE.json')
            if json.loads(marker.read_text())['report_sha256']!=digest(report):raise ValueError('tree report drift')
        if any(p.is_symlink() for p in copies):raise ValueError('symlink refused')
        if any(digest(p,p==mirror/rel)!=hashes[rel] for p in copies):raise ValueError('three-copy checksum differs: '+rel)
        entries.append(dict(relative_path=rel,sha256=hashes[rel],bytes=path.stat().st_size))
        print(json.dumps(dict(verified=index+1,total=len(paths))),flush=True)
    no_users(paths,archive)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),stage='verified_before_removal',
                 local_root=str(root),durable_root=str(mirror),retained_original_root=str(base/'study'),
                 archive_manifest_sha256=pin,entries=entries,total_bytes=sum(e['bytes'] for e in entries),
                 scope='completed tree model.parity.npz only; models, reports and predictions retained' if originals else 'duplicate transport model.parity.npz only; original scientific runs retained',
                 readback='all retained copies SHA-256; durable client readback after POSIX_FADV_DONTNEED',
                 filesystem_free_before_bytes=shutil.disk_usage(root).free)
    output.write_bytes(encoded(receipt))
    for path in paths:path.unlink()
    receipt['stage']='completed_tree_parity_removed' if originals else 'duplicate_transport_arrays_removed'
    receipt['filesystem_free_after_bytes']=shutil.disk_usage(root).free
    output.write_bytes(encoded(receipt))
    if not originals:
        (root/'LOCAL_ARCHIVE_STATUS.json').write_bytes(encoded(dict(partial_local_copy=True,
            receipt=str(output),restoration_root=str(mirror),retained='all models, reports and original study arrays')))
    print(json.dumps(dict(removed_files=len(paths),removed_bytes=receipt['total_bytes'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--completed-originals',action='store_true')
    main(p.parse_args().completed_originals)
