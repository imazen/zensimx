"""Prepare admitted outer TRAIN inputs on a bounded worker, preserving all gates."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time

from shaping_completion import REPO, encoded, sha

MAX_BYTES = 2 * 1024**3


def register(study, outer, host, output):
    from shaping_rescue_worker import HOSTS, reads
    from shaping_outer_admission import resolve
    from shaping_execution import load_registration
    from shaping_reduced_scope import admit_compute
    if host not in HOSTS:
        raise ValueError('approved native preparation worker required')
    admit_compute(study, host=host)
    with reads() as paths:
        admission, path = resolve(study, outer)
        plan, _, _ = load_registration(study)
    # Preparation reads the original registered feature/label tables. It never
    # fits a model or opens an outer response for selection.
    paths.update(map(Path, plan['input_pins']))
    for name, digest in plan['input_pins'].items():
        if sha(name) != digest:
            raise ValueError('registered preparation input drift')
    paths.update(p for p in REPO.rglob('*.py') if p.is_file() and '.git' not in p.parts and '.jj' not in p.parts)
    paths.update((Path(__file__).absolute(), path))
    files = {}
    for p in sorted(paths):
        if not p.is_file() or p.is_symlink():
            raise ValueError('canonical preparation dependency required: ' + str(p))
        files[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
    if len(files) > 16000 or sum(r['bytes'] for r in files.values()) > MAX_BYTES:
        raise ValueError('bounded outer preparation capsule exceeded')
    record = dict(schema='shaping-outer-inputs-worker-v1', study=str(study), outer=outer,
                  host=host, source=str(REPO), driver_sha256=sha(__file__), files=files,
                  admission_sha256=sha(path), plan_sha256=sha(study / 'PLAN.json'), models_fitted=0)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'REQUEST.json').write_bytes(encoded(record))
    return record


def verify(request, allow_missing=False):
    import socket
    q = json.loads(request.read_text())
    if (q['schema'] != 'shaping-outer-inputs-worker-v1' or q['host'] != socket.gethostname()
            or q['driver_sha256'] != sha(__file__) or q['models_fitted'] != 0):
        raise ValueError('outer preparation request identity differs')
    if len(q['files']) > 16000 or sum(r['bytes'] for r in q['files'].values()) > MAX_BYTES:
        raise ValueError('bounded outer preparation capsule exceeded')
    for name, row in q['files'].items():
        p = Path(name)
        if not p.is_absolute() or '..' in p.parts or p.is_symlink():
            raise ValueError('canonical outer preparation path required')
        if not p.exists() and allow_missing:
            continue
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('preserve different outer preparation input: ' + name)
    return q


def transfer(request):
    q = json.loads(request.read_text())
    script = Path(q['source']) / 'scripts/rev4_featpot' / Path(__file__).name
    subprocess.run(['ssh', q['host'], 'mkdir -p ' + shlex.quote(str(request.parent))], check=True)
    subprocess.run(['scp', str(request), q['host'] + ':' + str(request)], check=True)
    base = ['python3', '-B', str(script), 'verify', '--request', str(request)]
    subprocess.run(['ssh', q['host'], shlex.join(base + ['--allow-missing'])], check=True)
    paths = request.with_name('INPUT_PATHS.txt')
    paths.write_bytes(b''.join(Path(p).as_posix().lstrip('/').encode() + b'\0' for p in q['files']))
    receive = ['/home/lilith/work/claudehints/scripts/run-heavy', '--mem', '512M', '--jobs', '1', '--',
               'python3', '-B', str(script), 'receive', '--request', str(request), '--']
    subprocess.run(['rsync', '-rlptR', '--omit-dir-times', '--no-implied-dirs', '--ignore-existing',
                    '--from0', '--files-from=' + str(paths), '--rsync-path=' + shlex.join(receive),
                    '/', q['host'] + ':/'], check=True)
    subprocess.run(['ssh', q['host'], shlex.join(base)], check=True)


def receive(request, argv):
    from shaping_storage import reservation
    q = verify(request, allow_missing=True)
    if argv[:1] != ['--server']:
        raise ValueError('rsync receiver arguments required')
    with ExitStack() as stack:
        for part in range(2):
            stack.enter_context(reservation(Path(q['study']), request.parent / f'input-admission-{part}',
                                            30 * 1024**3, bytes_required=1024**3))
        subprocess.run(['rsync', *argv], check=True)
        verify(request)


def prepare(request):
    from shaping_rescue_dispatch import reserve_worker, release_worker
    from shaping_outer_admission import resolve
    from shaping_outer_execution import prepare as owner
    q = verify(request)
    study = Path(q['study'])
    _, admission = resolve(study, q['outer'])
    if sha(admission) != q['admission_sha256'] or sha(study / 'PLAN.json') != q['plan_sha256']:
        raise ValueError('outer preparation admission changed')
    result = request.with_name('PREPARED_RESULT.json')
    if result.exists() or request.with_name('STARTED.json').exists():
        raise ValueError('preserve existing preparation attempt')
    key = 'outer-input-preparation-' + str(q['outer'])
    while not reserve_worker(study, key)['can_start']:
        time.sleep(10)
    try:
        request.with_name('STARTED.json').write_bytes(encoded(dict(request_sha256=sha(request), models_fitted=0)))
        owner(study, q['outer'])
        partition = 'outer-' + str(q['outer'])
        roots = [study / name / partition for name in ('comparators', 'shapes', 'outer_inputs')]
        files = {}
        for root in roots:
            for p in sorted(root.rglob('*')):
                if p.is_symlink():
                    raise ValueError('noncanonical prepared output')
                if p.is_file():
                    files[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
        if not files or sum(r['bytes'] for r in files.values()) > MAX_BYTES:
            raise ValueError('bounded prepared outputs required')
        record = dict(stage='prepared', request_sha256=sha(request), admission_sha256=sha(admission),
                      files=files, models_fitted=0, checked_utc=datetime.now(timezone.utc).isoformat())
        result.write_bytes(encoded(record))
        return record
    finally:
        release_worker(study, key)


def collect(request):
    q = json.loads(request.read_text())
    result = request.with_name('PREPARED_RESULT.json')
    if result.exists():
        raise ValueError('preserve existing preparation collection')
    raw = subprocess.check_output(['ssh', q['host'], 'cat ' + shlex.quote(str(result))])
    row = json.loads(raw)
    if row['stage'] != 'prepared' or row['request_sha256'] != sha(request) or row['admission_sha256'] != q['admission_sha256'] or row['models_fitted']:
        raise ValueError('prepared output identity differs')
    if len(row['files']) > 16000 or sum(p['bytes'] for p in row['files'].values()) > MAX_BYTES:
        raise ValueError('bounded prepared outputs required')
    study = Path(q['study'])
    roots = [study / n / ('outer-' + str(q['outer'])) for n in ('comparators', 'shapes', 'outer_inputs')]
    for name, pin in row['files'].items():
        p = Path(name)
        if not any(r in p.parents for r in roots) or p.is_symlink() or '..' in p.parts:
            raise ValueError('prepared output outside admitted partition')
        if p.exists() and sha(p) != pin['sha256']:
            raise ValueError('preserve conflicting prepared output')
    paths = request.with_name('OUTPUT_PATHS.txt')
    paths.write_bytes(b''.join(p.lstrip('/').encode() + b'\0' for p in row['files']))
    subprocess.run(['rsync', '-rlptR', '--omit-dir-times', '--no-implied-dirs', '--ignore-existing',
                    '--from0', '--files-from=' + str(paths), q['host'] + ':/', '/'], check=True)
    for name, pin in row['files'].items():
        if sha(name) != pin['sha256']:
            raise ValueError('prepared output transport differs')
    result.write_bytes(raw)
    from shaping_outer_execution import prepared
    prepared(study, q['outer'], study / 'selection' / ('outer-' + str(q['outer'])) / 'OUTER_ADMISSION.json')
    request.with_name('COLLECTION_VERIFIED.json').write_bytes(encoded(dict(request_sha256=sha(request),
        result_sha256=sha(result), all_transport_sha256_verified=True, models_fitted=0)))
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('register', 'verify', 'transfer', 'receive', 'prepare', 'collect'))
    for key in ('request', 'study', 'output'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--outer', type=int, choices=range(3))
    p.add_argument('--host')
    p.add_argument('--allow-missing', action='store_true')
    a, rest = p.parse_known_args()
    if a.mode == 'receive':
        receive(a.request, rest[1:] if rest[:1] == ['--'] else rest)
    else:
        if rest:
            p.error('unexpected arguments')
        if a.mode == 'register':
            row = register(a.study, a.outer, a.host, a.output)
        elif a.mode == 'verify':
            row = verify(a.request, a.allow_missing)
        else:
            row = globals()[a.mode](a.request)
        if row is not None:
            print(json.dumps({k: v for k, v in row.items() if k != 'files'}), flush=True)
