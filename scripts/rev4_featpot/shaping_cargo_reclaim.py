"""Reclaim verified inactive Cargo trees after reference archive cold readback."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
from shaping_cargo_preserve import inactive, sha
from shaping_temp_storage import size


def witness(root, receipt_path, manifest_path, receipt_pin, manifest_pin):
    if sha(receipt_path) != receipt_pin or sha(manifest_path) != manifest_pin:
        raise ValueError('archive witness checksum differs')
    receipt = json.loads(receipt_path.read_text())
    if (receipt['all_artifact_sha256_verified'] is not True
            or receipt['readback_method'] != 'fsync, POSIX_FADV_DONTNEED and client SHA-256'):
        raise ValueError('complete destination cold readback required')
    destination = Path(receipt['destination']).resolve()
    if (root != root.resolve() or root.is_relative_to(destination)
            or destination.is_relative_to(root)):
        raise ValueError('local reference copy must be separate from durable archive')
    entries = {}
    for line in manifest_path.read_text().splitlines():
        pin, name = line.split('  ', 1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or name in entries:
            raise ValueError('unsafe archive manifest')
        entries[name] = pin
    if entries.get('MIRROR_VERIFIED.json') != receipt_pin:
        raise ValueError('manifest and cold-readback receipt differ')
    path = root / 'PRESERVATION.json'
    if entries.get(path.name) != sha(path):
        raise ValueError('preservation inventory differs from durable archive')
    record = json.loads(path.read_text())
    if record['schema'] != 'zensim-cargo-reference-preservation-v1' or record['data_deleted'] is not False:
        raise ValueError('completed original preservation required')
    for target in record['targets']:
        for row in target['preserved']:
            if (row['archive_path'] != 'objects/' + row['sha256']
                    or entries.get(row['archive_path']) != row['sha256']):
                raise ValueError('preserved reference lacks verified durable object')
    return record, receipt


def metadata(target):
    result = {}
    for path in sorted(target.rglob('*')):
        if any(part in ('.git', '.jj', '.workongoing') for part in path.relative_to(target).parts):
            raise ValueError('nested repository or workspace protects Cargo output')
        if path.is_symlink():
            raise ValueError('new symlink protects Cargo output')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('nonregular Cargo output')
        stat = path.stat()
        result[path.relative_to(target).as_posix()] = dict(
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino,
            device=stat.st_dev, mode=stat.st_mode)
    return result


def no_nested_mounts(target):
    # Include same-device bind mounts, which st_dev and ismount can miss.
    import re
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        escaped = line.split()[4]
        decoded = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), escaped)
        mount = Path(decoded)
        if mount == target or mount.is_relative_to(target):
            raise ValueError('mount point protects Cargo output')


def prune_objects(root, record):
    objects = root / 'objects'
    if objects.is_symlink() or not objects.is_dir():
        raise ValueError('regular reference object directory required')
    expected = {row['sha256'] for target in record['targets'] for row in target['preserved']}
    paths = list(objects.iterdir())
    if {p.name for p in paths} != expected:
        raise ValueError('reference object inventory changed after archival')
    for path in paths:
        if path.is_symlink() or not path.is_file() or sha(path) != path.name:
            raise ValueError('reference object changed after archival')
    inactive(objects)
    no_nested_mounts(objects)
    before = size(objects)
    shutil.rmtree(objects)
    return before


def reclaim(root, receipt_path, manifest_path, receipt_pin, manifest_pin, home,
            *, prune=False, references_only=False):
    record, receipt = witness(root, receipt_path, manifest_path, receipt_pin, manifest_pin)
    name = 'REFERENCE_COPY_RECLAIM.json' if references_only else 'CARGO_RECLAIM.json'
    path = root / name
    if path.exists():
        raise ValueError('preserve previous reclamation receipt')
    if not references_only:
        if record['host'] != socket.gethostname():
            raise ValueError('Cargo reclamation must run on its original host')
        for row in record['targets']:
            target = Path(row['path'])
            if (target != target.resolve() or not target.is_relative_to((home / 'tmp').resolve())
                    or target.is_symlink() or not (target / '.rustc_info.json').is_file()
                    or root.is_relative_to(target)):
                raise ValueError('recognized original scratch Cargo output required')
            inactive(target)
            no_nested_mounts(target)
            if metadata(target) != row['file_metadata']:
                raise ValueError('Cargo output changed after reference preservation')
            for saved in row['preserved']:
                relative = Path(saved['path'])
                if relative.is_absolute() or '..' in relative.parts or sha(target / relative) != saved['sha256']:
                    raise ValueError('original reference changed after preservation')
    result = dict(schema='zensim-cargo-reclamation-v1', host=socket.gethostname(),
                  started_utc=datetime.now(timezone.utc).isoformat(), stage='reclaiming',
                  archive=receipt['destination'], archive_receipt_sha256=receipt_pin,
                  archive_manifest_sha256=manifest_pin, preservation_sha256=sha(root / 'PRESERVATION.json'),
                  completed_targets=[], reference_object_allocated_bytes_removed=0)
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    if not references_only:
        for row in record['targets']:
            target = Path(row['path'])
            inactive(target)
            no_nested_mounts(target)
            if metadata(target) != row['file_metadata']:
                raise ValueError('Cargo output changed before reclamation')
            before = size(target)
            shutil.rmtree(target)
            result['completed_targets'].append(dict(path=str(target), allocated_bytes_removed=before))
            path.write_text(json.dumps(result, indent=2)+'\n')
    if prune or references_only:
        result['reference_object_allocated_bytes_removed'] = prune_objects(root, record)
    result.update(stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preservation', 'archive-receipt', 'archive-manifest', 'home'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('receipt-pin', 'manifest-pin'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--prune-reference-objects', action='store_true')
    parser.add_argument('--references-only', action='store_true')
    args = parser.parse_args()
    reclaim(args.preservation, args.archive_receipt, args.archive_manifest,
            args.receipt_pin, args.manifest_pin, args.home,
            prune=args.prune_reference_objects, references_only=args.references_only)
