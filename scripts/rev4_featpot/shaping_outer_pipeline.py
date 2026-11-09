"""Wait for inner admission, prepare TRAIN inputs remotely, and spread outer fits."""
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

HOST_SLOTS = dict(r7900x=20, i270=20, i134=14, r3800x=14, r5600g=11, r3500=5)


def placements(cells, outer, loads=None):
    # Rotate ties by fold, then fill proportionally to approved CPU capacity.
    names = list(HOST_SLOTS)
    names = names[outer:] + names[:outer]
    counts = {host: 0 if loads is None else loads[host] for host in names}
    out = []
    for cell in sorted(cells, key=lambda c: (c['seed'] is None, c['seed'] or 0, c['family'], c['id'])):
        host = min(names, key=lambda h: counts[h] / HOST_SLOTS[h])
        counts[host] += 1
        out.append((cell, host))
    return out


def submit_preparation(request, q, output):
    from shaping_outer_prepare_worker import transfer
    transfer(request)
    script = REPO / 'scripts/rev4_featpot/shaping_outer_prepare_worker.py'
    unit = 'zensim-outer-inputs-' + str(q['outer']) + '-' + output.name
    argv = ['systemd-run', '--user', '--unit=' + unit, '--property=Restart=no',
            '--property=StandardOutput=append:' + str(request.with_name('service.log')),
            '--property=StandardError=append:' + str(request.with_name('service.log')),
            'env', 'TMPDIR=/home/lilith/tmp', 'PYTHONPATH=' + str(Path(q['study']).parent / 'python-runtime'),
            '/home/lilith/work/claudehints/scripts/run-heavy', '--mem', '2G', '--jobs', '1', '--',
            'python3', '-B', str(script), 'prepare', '--request', str(request)]
    path = output / 'PREPARATION_SUBMITTED.json'
    path.write_bytes(encoded(dict(host=q['host'], unit=unit, command=argv, request_sha256=sha(request))))
    subprocess.run(['ssh', q['host'], shlex.join(argv)], check=True)


def register_fits(args):
    from shaping_execution import load_registration
    from shaping_outer_admission import resolve
    from shaping_outer_dispatch import selected_cells
    from shaping_rescue_worker import register
    from shaping_control_worker import register as register_control
    _, cells, _ = load_registration(args.study)
    admission, _ = resolve(args.study, args.outer)
    selected = selected_cells(cells, admission)
    controls = [c for c in cells if c['stage'] == 'outer' and c['partition'] == f'outer-{args.outer}'
                and c['family'] in ('extra_trees', 'shaped_additive')]
    if (len(controls) != 4 or {(c['family'], c['seed']) for c in controls}
            != {('extra_trees', s) for s in range(3)} | {('shaped_additive', None)}):
        raise ValueError('complete registered outer controls required')
    selected += controls
    jobs = []
    from shaping_placement import loads
    live_loads, observation = loads(args.study, HOST_SLOTS)
    wave = args.output / 'neural-fits'
    if wave.exists():
        raise ValueError('preserve existing outer registration')
    wave.mkdir()
    (wave / 'PLACEMENT_OBSERVATION.json').write_bytes(encoded(observation))
    for cell, host in placements(selected, args.outer, live_loads):
        directory = wave / 'jobs' / cell['id']
        if cell['family'] in ('extra_trees', 'shaped_additive'):
            register_control(args.study, cell['id'], args.binaries, host, directory)
        else:
            register(args.study, cell['id'], host, args.binaries, args.qualification, directory, execution_kind='outer')
        request = directory / 'REQUEST.json'
        assignment = args.study / 'downstream_assignments' / ('outer-' + cell['id'] + '-' + host + '.json')
        if assignment.exists():
            raise ValueError('preserve existing outer assignment')
        assignment.write_bytes((directory / 'ASSIGNMENT.json').read_bytes())
        jobs.append(dict(cell=cell, host=host, request=str(request), request_sha256=sha(request), assignment=str(assignment), assignment_sha256=sha(assignment)))
    record = dict(stage='registered', source=str(REPO), jobs=jobs,
                  source_driver_sha256=sha(REPO / 'scripts/rev4_featpot/shaping_rescue_worker.py'),
                  admission_sha256=sha(args.study / 'selection' / f'outer-{args.outer}' / 'OUTER_ADMISSION.json'),
                  training_replayed=False)
    (wave / 'REGISTRATION.json').write_bytes(encoded(record))
    return wave, jobs


def collect_controls(args, jobs):
    from shaping_control_worker import collect
    required = [j for j in jobs if j['cell']['family'] in ('extra_trees', 'shaped_additive')]
    completed = []
    for job in required:
        request = Path(job['request'])
        q = json.loads(request.read_text())
        result = Path(q['result_output'])
        if (result / 'WORKER_COLLECTION.json').exists():
            receipt = json.loads((result / 'WORKER_COLLECTION.json').read_text())
            if receipt['request_sha256'] != sha(request):
                raise ValueError('outer control collection request changed')
            completed.append(job['cell']['id'])
            continue
        program = 'request=Path(' + repr(str(request)) + ')\nresult=Path(' + repr(str(result)) + ')\n' + '''
status=request.with_name('STATUS.json')
row=json.loads(status.read_text()) if status.exists() else {}
alive=False
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:argv=[v.decode() for v in p.read_bytes().split(b'\\0') if v]
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if 'work' in argv and str(request) in argv and any(Path(v).name=='shaping_rescue_worker.py' for v in argv):alive=True
print(json.dumps(dict(ready=(result/'WORKER_EXECUTION.json').exists(),stage=row.get('stage'),worker_alive=alive)))
'''
        probe = subprocess.run(['ssh', q['host'], 'python3 -B -'],
                               input='from pathlib import Path\nimport json\n' + program,
                               text=True, capture_output=True, check=True)
        observation = json.loads(probe.stdout)
        if not observation['ready']:
            if observation['stage'] == 'failed' or not observation['worker_alive']:
                raise RuntimeError('outer control failed; preserve original attempt')
            continue
        transport = args.output / 'control-transport' / job['cell']['id']
        transport.mkdir(parents=True, exist_ok=False)
        bundle = args.study.parent / 'worker-bundles' / ('outer-control-' + job['cell']['id'] + '.tar')
        argv = ['env', 'TMPDIR=/home/lilith/tmp', 'PYTHONPATH=' + str(args.study.parent / 'python-runtime'),
                '/home/lilith/work/claudehints/scripts/run-heavy', '--mem', '512M', '--jobs', '1', '--',
                'python3', '-B', str(REPO / 'scripts/rev4_featpot/shaping_control_worker.py'),
                'pack', '--request', str(request), '--bundle', str(bundle)]
        (transport / 'PACK_SUBMITTED.json').write_bytes(encoded(dict(command=argv, host=q['host'], request_sha256=sha(request))))
        with (transport / 'pack.log').open('x') as log:
            subprocess.run(['ssh', q['host'], shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['rsync', '-a', '--partial', q['host'] + ':' + str(bundle),
                        q['host'] + ':' + str(bundle) + '.json', str(transport) + '/'], check=True)
        local = transport / bundle.name
        pin = json.loads(Path(str(local) + '.json').read_text())['bundle_sha256']
        collect(request, local, pin)
        completed.append(job['cell']['id'])
    return completed, len(required)


def watch(args):
    from shaping_outer_prepare_worker import register, collect
    from shaping_rescue_host_launch import launch
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'OWNER.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), outer=args.outer, driver_sha256=sha(__file__), training_replayed=False)
        def save(stage, **values):
            status.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **values)
            (args.output / 'STATUS.json').write_bytes(encoded(status))
        admission = args.study / 'selection' / f'outer-{args.outer}' / 'OUTER_ADMISSION.json'
        try:
            while not admission.exists():
                save('waiting_for_complete_inner_admission')
                time.sleep(30)
            q = register(args.study, args.outer, args.prepare_host, args.output / 'preparation')
            request = args.output / 'preparation/REQUEST.json'
            save('transferring_admitted_preparation_inputs')
            submit_preparation(request, q, args.output)
            while True:
                result = request.with_name('PREPARED_RESULT.json')
                probe = ['ssh', q['host'], 'test -f ' + shlex.quote(str(result))]
                ready = subprocess.run(probe).returncode
                if ready == 0:
                    break
                if ready != 1:
                    raise RuntimeError('preparation host connection failed; preserve submission')
                # A failed service never becomes a new preparation attempt.
                unit = json.loads((args.output / 'PREPARATION_SUBMITTED.json').read_text())['unit']
                state = subprocess.check_output(['ssh', q['host'], 'systemctl --user show ' + shlex.quote(unit) + ' -p ActiveState --value'], text=True).strip()
                if state not in ('active', 'activating'):
                    raise RuntimeError('outer input preparation stopped before terminal receipt')
                save('waiting_for_bounded_worker_preparation')
                time.sleep(30)
            collect(request)
            save('registering_admitted_outer_neural_fits')
            wave, jobs = register_fits(args)
            for host in HOST_SLOTS:
                if any(j['host'] == host for j in jobs):
                    save('launching_admitted_outer_neural_fits', host=host)
                    launch([wave], REPO, host, args.output / ('launch-' + host))
            save('all_outer_fits_submitted', required_models=len(jobs), wave=str(wave))
            while True:
                completed, required = collect_controls(args, jobs)
                save('collecting_outer_controls', required_models=len(jobs), wave=str(wave),
                     collected_controls=completed, required_controls=required)
                if len(completed) == required:
                    save('all_outer_fits_submitted_and_controls_collected', required_models=len(jobs), wave=str(wave))
                    break
                time.sleep(30)
        except Exception as error:
            save('failed_with_preserved_evidence', error=str(error))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'output', 'binaries', 'qualification'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--outer', type=int, choices=range(3), required=True)
    p.add_argument('--prepare-host', choices=tuple(HOST_SLOTS), required=True)
    watch(p.parse_args())
