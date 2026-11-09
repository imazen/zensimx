"""Deploy the exact existing evaluator binaries before admitting any prediction."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys

HOSTS = ('r7900x', 'i270', 'i134', 'r3800x', 'r5600g', 'r3500')
MAX_BYTES = 2 * 1024**3


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def register(study, source, binaries, inspector, output):
    files = set()
    for directory in binaries:
        build = directory / 'BUILD.json'
        record = json.loads(build.read_text()); files.add(build)
        for name, pin in record['binary_sha256'].items():
            path = directory / name
            if sha(path) != pin:
                raise ValueError('evaluator native binary differs')
            files.add(path)
    files.add(inspector)
    rows = {}
    for path in sorted(files):
        if not path.is_file() or path.is_symlink() or study.parent not in path.parents:
            raise ValueError('canonical study runtime path required')
        rows[str(path)] = dict(sha256=sha(path), bytes=path.stat().st_size, executable=path.stat().st_mode & 0o111)
    if sum(v['bytes'] for v in rows.values()) > MAX_BYTES or len(rows) > 128:
        raise ValueError('bounded evaluator runtime closure exceeded')
    if not rows[str(inspector)]['executable']:
        raise ValueError('executable evaluator inspector required')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='shaping-evaluator-runtime-v1', study=str(study), source=str(source),
                  plan_sha256=sha(study / 'PLAN.json'), driver_sha256=sha(__file__), files=rows,
                  predictions_generated=0, models_fitted=0)
    write(output / 'REGISTRATION.json', record)
    return record


def verify(manifest, allow_missing=False):
    q = json.loads(manifest.read_text())
    if q['schema'] != 'shaping-evaluator-runtime-v1' or sha(__file__) != q['driver_sha256'] or socket.gethostname() not in HOSTS:
        raise ValueError('evaluator runtime identity differs')
    if sha(Path(q['study']) / 'PLAN.json') != q['plan_sha256']:
        raise ValueError('evaluator runtime study differs')
    if len(q['files']) > 128 or sum(r['bytes'] for r in q['files'].values()) > MAX_BYTES:
        raise ValueError('bounded evaluator runtime closure exceeded')
    missing = 0
    for name, row in q['files'].items():
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts or path.is_symlink():
            raise ValueError('noncanonical evaluator runtime input')
        if not path.exists() and allow_missing:
            missing += row['bytes']; continue
        if (path.stat().st_size != row['bytes'] or sha(path) != row['sha256']
                or path.stat().st_mode & 0o111 != row['executable']):
            raise ValueError('preserve different existing evaluator runtime: ' + name)
    return q, missing


def receive(manifest, arguments):
    q, missing = verify(manifest, True)
    sys.path.insert(0, str(Path(q['source']) / 'scripts/rev4_featpot'))
    from shaping_storage import reservation
    if arguments[:1] != ['--server']:
        raise ValueError('rsync receiver arguments required')
    with ExitStack() as stack:
        remaining = missing + 16 * 1024**2 if missing else 0
        part = 0
        while remaining:
            amount = min(remaining, 1024**3)
            stack.enter_context(reservation(Path(q['study']), manifest.parent / f'runtime-admission-{part}',
                                            30 * 1024**3, bytes_required=amount))
            part += 1; remaining -= amount
        subprocess.run(['rsync', *arguments], check=True)
        verify(manifest)


def transfer(manifest, host):
    if host not in HOSTS:
        raise ValueError('approved evaluator endpoint required')
    q = json.loads(manifest.read_text())
    helper = manifest.with_name('runtime-owner.py')
    if not helper.exists():
        helper.write_bytes(Path(__file__).read_bytes())
    if sha(helper) != q['driver_sha256']:
        raise ValueError('runtime deployment helper changed')
    subprocess.run(['ssh', host, shlex.join(['mkdir', '-p', str(manifest.parent)])], check=True)
    subprocess.run(['rsync', '-a', '--ignore-existing', str(helper), str(manifest), host + ':' + str(manifest.parent) + '/'], check=True)
    identity='from pathlib import Path\nimport hashlib\n' + '\n'.join(
        'assert hashlib.sha256(Path('+repr(str(path))+').read_bytes()).hexdigest()=='+repr(pin)
        for path,pin in [(helper,q['driver_sha256']),(manifest,sha(manifest))])
    subprocess.run(['ssh',host,'python3 -B -'],input=identity,text=True,check=True)
    base = ['python3', '-B', str(helper), 'verify', '--manifest', str(manifest)]
    subprocess.run(['ssh', host, shlex.join(base + ['--allow-missing'])], check=True)
    paths = manifest.with_name('INPUT_PATHS.txt')
    paths.write_bytes(b''.join(n.lstrip('/').encode() + b'\0' for n in q['files']))
    receiver = ['/home/lilith/work/claudehints/scripts/run-heavy', '--mem', '512M', '--jobs', '1', '--',
                'python3', '-B', str(helper), 'receive', '--manifest', str(manifest), '--']
    subprocess.run(['rsync', '-rlptR', '--omit-dir-times', '--no-implied-dirs', '--ignore-existing', '--from0',
                    '--files-from=' + str(paths), '--rsync-path=' + shlex.join(receiver), '/', host + ':/'], check=True)
    subprocess.run(['ssh', host, shlex.join(base)], check=True)
    write(manifest.with_name(host + '-VERIFIED.json'), dict(host=host, manifest_sha256=sha(manifest),
         all_runtime_bytes_and_modes_verified=True, predictions_generated=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('register', 'verify', 'receive', 'transfer'))
    for name in ('study', 'source', 'inspector', 'output', 'manifest'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--binary-dir', type=Path, action='append')
    p.add_argument('--host', choices=HOSTS)
    p.add_argument('--allow-missing', action='store_true')
    args, rest = p.parse_known_args()
    if args.mode == 'register':
        register(args.study, args.source, args.binary_dir, args.inspector, args.output)
    elif args.mode == 'transfer':
        transfer(args.manifest, args.host)
    elif args.mode == 'receive':
        receive(args.manifest, rest[1:] if rest[:1] == ['--'] else rest)
    else:
        _, missing = verify(args.manifest, args.allow_missing)
        print(json.dumps(dict(missing_bytes=missing)), flush=True)
