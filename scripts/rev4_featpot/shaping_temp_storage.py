"""Inventory zensim temporary trees and Cargo output; never delete data."""
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone


def size(path):
    result = subprocess.run(['du', '-sx', '--block-size=1', str(path)],
                            capture_output=True, text=True, check=True)
    return int(result.stdout.split()[0])


def targets(root, depth=0):
    if depth > 3 or not root.is_dir() or root.is_symlink():
        return
    # CARGO_TARGET_DIR is often named *-build or *-target rather than target.
    if (root/'.rustc_info.json').is_file() or (root.name == 'target' and (root/'CACHEDIR.TAG').is_file()):
        yield root
        return
    if root.name == 'target':
        return
    for path in root.iterdir():
        if path.name not in ('runs', 'data', 'comparators', 'cargo-home', 'objects'):
            yield from targets(path, depth+1)


def process_uses(candidates, proc=Path('/proc')):
    matches = {str(path): [] for path in candidates}
    unreadable = []
    for pid in proc.glob('[0-9]*'):
        try:
            paths = []
            for name in ('exe', 'cwd'):
                try: paths.append(os.readlink(pid/name))
                except FileNotFoundError: pass
            for fd in (pid/'fd').iterdir():
                try: paths.append(os.readlink(fd))
                except FileNotFoundError: pass
            paths.extend(line.split(maxsplit=5)[-1] for line in (pid/'maps').read_text().splitlines()
                         if len(line.split(maxsplit=5)) == 6)
            # Record only matching paths, never full environment or command lines.
            command = (pid/'cmdline').read_bytes().decode(errors='replace').split('\0')
            paths.extend(command)
            paths.extend(v.split('=', 1)[1] for v in (pid/'environ').read_bytes().decode(errors='replace').split('\0')
                         if v.startswith('CARGO_TARGET_DIR='))
            name = (pid/'comm').read_text().strip()
            for candidate in matches:
                hits = [p for p in paths if p == candidate or p.startswith(candidate+'/')]
                if hits: matches[candidate].append(dict(pid=int(pid.name), name=name,
                                                       matching_paths=sorted(set(hits))))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            unreadable.append(int(pid.name))
    return matches, unreadable


def inventory(home, output):
    scratch = home/'tmp'
    roots = sorted(p for p in scratch.iterdir() if not p.is_symlink() and
                   ('zensim' in p.name.lower() or p.name.startswith(('shaping-', 'fitv2')))
                   and p != output and not output.is_relative_to(p))
    cargo = sorted(set(t for root in roots for t in targets(root)))
    uses, unreadable = process_uses(cargo)
    record = dict(schema='zensim-temp-storage-inventory-v1', host=os.uname().nodename,
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  roots=[dict(path=str(p), allocated_bytes=size(p), directory=p.is_dir()) for p in roots],
                  cargo_targets=[], unreadable_processes=unreadable, data_deleted=False)
    for target in cargo:
        incremental = sorted(p for p in target.glob('*/incremental') if p.is_dir())
        binaries = sorted(p for mode in ('debug', 'release') for p in (target/mode).glob('*')
                          if p.is_file() and os.access(p, os.X_OK))
        record['cargo_targets'].append(dict(path=str(target), allocated_bytes=size(target),
            active_processes=uses[str(target)], active_use_check_complete=not unreadable,
            incremental=[dict(path=str(p), allocated_bytes=size(p)) for p in incremental],
            top_level_executables=[dict(path=str(p), bytes=p.stat().st_size) for p in binaries],
            disposition='preserve active build output' if uses[str(target)] else
                        'review reference binaries and non-Cargo artifacts before reclamation'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'INVENTORY.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(host=record['host'], roots=len(roots), cargo_targets=len(cargo),
                         allocated_bytes=sum(row['allocated_bytes'] for row in record['roots']),
                         active_targets=sum(bool(row['active_processes']) for row in record['cargo_targets']),
                         unreadable_processes=len(unreadable), output=str(output), data_deleted=False)), flush=True)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    inventory(args.home, args.output)
