"""Preserve reference files before reclaiming inactive Cargo output; never delete."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
from shaping_temp_storage import process_uses


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compiler_object(path):
    """Recognize native compiler objects conservatively; keep everything else."""
    with path.open('rb') as stream:
        header = stream.read(20)
    if path.suffix == '.rlib' and path.name.startswith('lib'):
        return header.startswith(b'!<arch>\n')
    if path.suffix == '.rmeta':
        return header.startswith(b'rust')
    if path.suffix == '.o' and header.startswith(b'\x7fELF') and len(header) >= 18:
        byteorder = {1: 'little', 2: 'big'}.get(header[5])
        return byteorder is not None and int.from_bytes(header[16:18], byteorder) == 1
    return False


def inactive(target):
    uses, unreadable = process_uses([target])
    if unreadable or uses[str(target)]:
        raise ValueError('complete visibility and no active Cargo users required')


def preserve(inventory, output, home):
    record = json.loads(inventory.read_text())
    if record['host'] != socket.gethostname():
        raise ValueError('inventory host differs')
    selected = [Path(row['path']) for row in record['cargo_targets']
                if not row['active_processes'] and row['active_use_check_complete']]
    if not selected:
        raise ValueError('no inactive Cargo target selected')
    if output.exists():
        raise ValueError('preserve existing reference archive')
    for target in selected:
        if (target != target.resolve() or not target.is_relative_to((home / 'tmp').resolve()) or target.is_symlink()
                or not (target / '.rustc_info.json').is_file()
                or any(p.is_symlink() for p in target.parents)):
            raise ValueError('recognized Cargo output under scratch required')
        if output.is_relative_to(target):
            raise ValueError('reference archive must be outside Cargo output')
        inactive(target)
    output.mkdir(parents=True)
    result = dict(schema='zensim-cargo-reference-preservation-v1', host=record['host'],
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  inventory_sha256=sha(inventory), targets=[], data_deleted=False)
    for index, target in enumerate(selected):
        files = {}
        saved = []
        for path in sorted(target.rglob('*')):
            if path.is_symlink():
                raise ValueError('symlink in Cargo output; target remains protected')
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError('nonregular Cargo output')
            stat = path.stat()
            relative = path.relative_to(target).as_posix()
            files[relative] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                   inode=stat.st_ino, device=stat.st_dev, mode=stat.st_mode)
            parts = Path(relative).parts
            compiler_directory = len(parts) > 2 and parts[1] in ('deps', 'build', 'incremental')
            if compiler_directory and compiler_object(path):
                continue
            pin = sha(path)
            destination = output / 'objects' / pin
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(path, destination)
            if sha(destination) != pin or sha(path) != pin:
                raise ValueError('reference changed during preservation')
            saved.append(dict(path=relative, bytes=stat.st_size, sha256=pin,
                              archive_path=destination.relative_to(output).as_posix()))
        inactive(target)
        result['targets'].append(dict(path=str(target), file_metadata=files,
                                      preserved=saved, recognized_compiler_objects=len(files)-len(saved)))
    (output / 'PRESERVATION.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(inventory, output / 'INVENTORY.json')
    print(json.dumps(dict(host=record['host'], targets=len(selected),
                         preserved_files=sum(len(t['preserved']) for t in result['targets']),
                         preserved_bytes=sum(p['bytes'] for t in result['targets'] for p in t['preserved']),
                         data_deleted=False)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    args = parser.parse_args()
    preserve(args.inventory, args.output, args.home)
