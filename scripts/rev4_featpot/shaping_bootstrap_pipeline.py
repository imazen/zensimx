"""Dispatch independent registered source-bootstrap analyses on approved workers."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from shaping_completion import REPO, encoded, sha
from shaping_outer_pipeline import HOST_SLOTS


def queue(study, registry):
    from shaping_bootstrap_plan import load_comparison
    from shaping_reduced_scope import coverage
    if coverage(study) is None:
        raise ValueError('owner-approved reduced uncertainty scope required')
    rows = json.loads((registry / 'MATCHED_REGISTRATION.json').read_text())['comparisons']
    if len(rows) != 24 or len({r['file'] for r in rows}) != 24:
        raise ValueError('all 24 reduced matched analyses required')
    result = [dict(name='primary', comparison=None, spec=None)]
    for row in rows:
        path = registry / row['file']
        spec, _ = load_comparison(study, path)
        if spec != row['spec']:
            raise ValueError('registered uncertainty comparison differs')
        result.append(dict(name=path.stem, comparison=str(path), spec=spec))
    counts = {host: 0 for host in HOST_SLOTS}
    for row in result:
        host = min(counts, key=lambda h: counts[h] / HOST_SLOTS[h])
        counts[host] += 1
        row['host'] = host
    return result


def ready(study, job):
    from shaping_execution import load_registration
    from shaping_outer_admission import resolve
    from shaping_outer_evidence import matched_cells, primary_cells
    from shaping_bootstrap import ready_cells
    if not all((study / 'selection' / f'outer-{o}' / 'OUTER_ADMISSION.json').exists() for o in range(3)):
        return False
    _, cells, _ = load_registration(study)
    admissions = {o: resolve(study, o)[0] for o in range(3)}
    if job['spec'] is None:
        wanted = primary_cells(cells, admissions)
    else:
        pairs, _ = matched_cells(cells, admissions, **job['spec'])
        wanted = list({c['id']: c for pair in pairs for c in (pair['candidate'], pair['control'])}.values())
    return ready_cells(study, wanted, admissions)


def submit(args, job):
    from shaping_bootstrap_worker import register
    from shaping_rescue_host_launch import launch
    wave = args.output / 'jobs' / job['name']
    request_dir = wave / 'request'
    result = args.output / 'results' / job['name']
    q = register(args.study, args.draws, args.binaries, result, job['host'], request_dir,
                 Path(job['comparison']) if job['comparison'] else None)
    request = request_dir / 'REQUEST.json'
    assignment = args.study / 'downstream_assignments' / ('bootstrap-' + q['cell']['id'] + '-' + job['host'] + '.json')
    with assignment.open('xb') as stream:
        stream.write((request_dir / 'ASSIGNMENT.json').read_bytes())
    item = dict(cell=q['cell'], host=job['host'], request=str(request), request_sha256=sha(request),
                assignment=str(assignment), assignment_sha256=sha(assignment))
    (wave / 'REGISTRATION.json').write_bytes(encoded(dict(stage='registered', source=str(REPO), jobs=[item],
        source_driver_sha256=sha(REPO / 'scripts/rev4_featpot/shaping_rescue_worker.py'), scientific_fit_count=0)))
    # The existing launcher journals submission before SSH and never retries.
    launch([wave], REPO, job['host'], wave / 'launch')
    return request


def collect(args, job, request):
    from shaping_bootstrap_worker import collect as owner
    from shaping_bootstrap import archive_output, complete_output
    q = json.loads(request.read_text())
    result = Path(q['result_output'])
    if complete_output(result):
        if not (result / 'WORKER_COLLECTION.json').exists():
            raise ValueError('completed bootstrap lacks worker collection evidence')
        archive_output(result, args.destination / job['name'], args.output / 'logs')
        return True
    status = request.with_name('STATUS.json')
    program = 'path=' + repr(str(status)) + '\n' + '''
from pathlib import Path
import json
p=Path(path)
print(p.read_text() if p.exists() else '{}')
'''
    probe = subprocess.run(['ssh', job['host'], 'python3 -B -'], input=program, text=True,
                           capture_output=True, check=True)
    row = json.loads(probe.stdout)
    if not row:
        return False
    if row.get('request_sha256') != sha(request):
        raise ValueError('bootstrap worker request differs')
    if row['stage'] == 'failed':
        raise RuntimeError('bootstrap worker failed; preserve original analysis')
    if row['stage'] != 'analysis_finished':
        return False
    transport = args.output / 'transport' / job['name']
    transport.mkdir(parents=True, exist_ok=False)
    bundle = args.study.parent / 'worker-bundles' / ('bootstrap-' + q['cell']['id'] + '.tar')
    argv = ['env', 'TMPDIR=/home/lilith/tmp', 'PYTHONPATH=' + str(args.study.parent / 'python-runtime'),
            '/home/lilith/work/claudehints/scripts/run-heavy', '--mem', '512M', '--jobs', '1', '--',
            'python3', '-B', str(REPO / 'scripts/rev4_featpot/shaping_bootstrap_worker.py'),
            'pack', '--request', str(request), '--bundle', str(bundle)]
    (transport / 'PACK_SUBMITTED.json').write_bytes(encoded(dict(command=argv, request_sha256=sha(request), host=job['host'])))
    with (transport / 'pack.log').open('x') as log:
        subprocess.run(['ssh', job['host'], shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(['rsync', '-a', '--partial', job['host'] + ':' + str(bundle),
                    job['host'] + ':' + str(bundle) + '.json', str(transport) + '/'], check=True)
    local = transport / bundle.name
    owner(request, local, json.loads(Path(str(local) + '.json').read_text())['bundle_sha256'])
    archive_output(result, args.destination / job['name'], args.output / 'logs')
    return True


def watch(args):
    jobs = queue(args.study, args.registry)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('results', 'logs'):
        (args.output / name).mkdir()
    with (args.output / 'OWNER.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / 'QUEUE.json').write_bytes(encoded(dict(jobs=jobs, scientific_fit_count=0,
            draws_sha256=sha(args.draws / 'REGISTRATION.json'), matched_registry_sha256=sha(args.registry / 'MATCHED_REGISTRATION.json'))))
        submitted = {}; completed = []
        while True:
            status = dict(pid=os.getpid(), required=len(jobs), completed=completed, submitted=list(submitted),
                          driver_sha256=sha(__file__), stage='waiting_for_complete_paired_outer_evidence', scientific_fit_count=0)
            try:
                for job in jobs:
                    name = job['name']
                    if name in completed:
                        continue
                    if name not in submitted:
                        if not ready(args.study, job):
                            continue
                        submitted[name] = submit(args, job)
                    if collect(args, job, submitted[name]):
                        completed.append(name)
                status.update(submitted=list(submitted), stage='complete' if len(completed) == len(jobs) else 'waiting_or_computing')
            except Exception as error:
                status.update(stage='failed_with_preserved_evidence', error=str(error))
                status['checked_utc'] = datetime.now(timezone.utc).isoformat()
                (args.output / 'STATUS.json').write_bytes(encoded(status))
                raise
            status['checked_utc'] = datetime.now(timezone.utc).isoformat()
            (args.output / 'STATUS.json').write_bytes(encoded(status))
            if status['stage'] == 'complete':
                return
            time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'draws', 'registry', 'binaries', 'output', 'destination'):
        p.add_argument('--' + key, type=Path, required=True)
    watch(p.parse_args())
