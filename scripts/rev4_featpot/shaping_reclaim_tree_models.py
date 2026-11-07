"""Reclaim archived historical Python tree models; preserve their durable copies."""
from datetime import datetime,timezone
import json,shutil
from collections import Counter
from pathlib import Path
from shaping_completion import encoded,sha
from shaping_reclaim_grams import digest,no_users as no_fit_processes
from shaping_checkpoint_retention import no_users


def main():
    root=Path('/mnt/v/output/zensim/tree-study-20261106')
    mirror=Path('/mnt/tower/output/zensim/tree-study-20261106')
    study=Path('/mnt/v/output/zensim/shaping-completion-20261107')
    output=study/'RECLAIMED_TREE_MODELS.json'
    if output.exists():raise ValueError('preserve existing retention receipt')
    expected='c98912ff5e927496899a5f6d4bd5aaa8a60f4b331d5b1fa483966ce1ac35ecad'
    if sha(root/'SHA256SUMS')!=expected or sha(mirror/'SHA256SUMS')!=expected:
        raise ValueError('historical archive identity differs')
    hashes={name:pin for pin,name in (line.split('  ',1) for line in (root/'SHA256SUMS').read_text().splitlines())}
    paths=sorted(root.rglob('model.pickle'))
    inventory=Counter(p.relative_to(root).parts[0] for p in paths)
    expected_inventory={'search':129,'peer-search':54,'constraint-controls':6}
    if inventory!=expected_inventory or any(p.relative_to(root).parts[1]!='cells' for p in paths):
        raise ValueError('historical model inventory differs')
    pins=json.loads((study/'study/PLAN.json').read_text())['input_pins']
    if set(map(str,paths))&set(pins):raise ValueError('active study pins a historical model')
    if root.stat().st_dev!=study.stat().st_dev:raise ValueError('retention is not on experiment SSD')
    rows=[dict(path=str(p)) for p in paths];no_fit_processes(paths,root);no_users(rows);entries=[];identities={}
    for path in paths:
        relative=str(path.relative_to(root));remote=mirror/relative
        if path.is_symlink() or remote.is_symlink():raise ValueError('symlink refused')
        if digest(path)!=hashes[relative] or digest(remote,True)!=hashes[relative]:raise ValueError('model readback differs')
        stat=path.stat();identities[path]=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        entries.append(dict(relative_path=relative,sha256=hashes[relative],bytes=stat.st_size))
    no_fit_processes(paths,root);no_users(rows)
    for path,identity in identities.items():
        stat=path.stat()
        if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=identity:raise ValueError('historical model changed during verification')
    note_path=root/'LOCAL_ARCHIVE_STATUS.json';previous=note_path.read_bytes();note=json.loads(previous)
    record=dict(stage='verified_before_removal',checked_utc=datetime.now(timezone.utc).isoformat(),
        local_root=str(root),durable_root=str(mirror),archive_manifest_sha256=expected,entries=entries,
        total_bytes=sum(e['bytes'] for e in entries),previous_local_archive_status=note,
        readback='all 189 local SHA-256 and durable SHA-256 after POSIX_FADV_DONTNEED; two lsof inspections',
        filesystem_free_before_bytes=shutil.disk_usage(study).free)
    with output.open('xb') as stream:stream.write(encoded(record))
    for path in paths:path.unlink()
    record.update(stage='local_model_duplicates_removed',filesystem_free_after_bytes=shutil.disk_usage(study).free)
    output.write_bytes(encoded(record))
    if note_path.read_bytes()!=previous:raise ValueError('local archive note changed; receipt preserved')
    note['retained']='native exports, reports, predictions, specifications, and active input arrays; Python models available at restoration_root'
    note.setdefault('additional_retention_receipts',[]).append(str(output));note_path.write_bytes(encoded(note))
    print(json.dumps(dict(files=len(entries),bytes=record['total_bytes'],free_before=record['filesystem_free_before_bytes'],free_after=record['filesystem_free_after_bytes'])),flush=True)


if __name__=='__main__':main()
