"""Spread admitted reduced variants, admitting human-only work before diagnostics."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from shaping_completion import REPO, encoded, sha
from shaping_outer_pipeline import HOST_SLOTS, placements, collect_controls

STAGES = ('human_only', 'source_curve', 'shuffle')
COUNTS = dict(human_only=16, source_curve=32, shuffle=16)


def selected(study, outer, stage):
    from shaping_execution import load_registration
    from shaping_outer_admission import resolve
    from shaping_outer_dispatch import variant_cells
    _, cells, _ = load_registration(study)
    admission, path = resolve(study, outer)
    if 'reduced_scope_sha256' not in admission:
        raise ValueError('owner-approved reduced variant scope required')
    chosen = variant_cells(cells, admission) + variant_cells(cells, admission, controls=True)
    chosen = [c for c in chosen if c['stage'] == stage]
    if stage not in STAGES or len(chosen) != COUNTS[stage] or len({c['id'] for c in chosen}) != len(chosen):
        raise ValueError('complete reduced paired variant stage required')
    return chosen, path


def register_fits(args, stage):
    from shaping_rescue_worker import register
    from shaping_control_worker import register as register_control
    cells, admission = selected(args.study, args.outer, stage)
    wave = args.output / stage
    wave.mkdir(exist_ok=False)
    jobs = []
    for cell, host in placements(cells, args.outer):
        directory = wave / 'jobs' / cell['id']
        if cell['family'] in ('extra_trees', 'shaped_additive'):
            register_control(args.study, cell['id'], args.binaries, host, directory)
        else:
            register(args.study, cell['id'], host, args.binaries, args.qualification, directory,
                     execution_kind='variant', human_qualification=args.human_qualification)
        request = directory / 'REQUEST.json'
        assignment = args.study / 'downstream_assignments' / ('variant-' + cell['id'] + '-' + host + '.json')
        with assignment.open('xb') as stream:
            stream.write((directory / 'ASSIGNMENT.json').read_bytes())
        jobs.append(dict(cell=cell, host=host, request=str(request), request_sha256=sha(request),
                         assignment=str(assignment), assignment_sha256=sha(assignment)))
    record = dict(stage='registered', source=str(REPO), jobs=jobs, training_replayed=False,
                  source_driver_sha256=sha(REPO / 'scripts/rev4_featpot/shaping_rescue_worker.py'),
                  admission_sha256=sha(admission), variant_stage=stage)
    (wave / 'REGISTRATION.json').write_bytes(encoded(record))
    return wave, jobs


def admission_started(jobs):
    """Do not enqueue diagnostics ahead of higher-priority, still-waiting fits."""
    waiting = []
    for host in HOST_SLOTS:
        required = [j for j in jobs if j['host'] == host]
        if not required:
            continue
        program = 'jobs=' + repr(required) + '\n' + '''
import json
from pathlib import Path
waiting=[]
for job in jobs:
 request=Path(job['request']);path=request.with_name('STATUS.json')
 if not path.exists():waiting.append(job['cell']['id']);continue
 row=json.loads(path.read_text())
 if row['request_sha256']!=job['request_sha256']:raise ValueError('variant worker request differs')
 if row['stage']=='failed':raise ValueError('variant failed; preserve original attempt')
 if row['stage'] not in ('running','trained'):waiting.append(job['cell']['id'])
print(json.dumps(waiting))
'''
        result = subprocess.run(['ssh', host, 'python3 -B -'], input=program, text=True,
                                capture_output=True, check=True)
        waiting.extend(json.loads(result.stdout))
    return waiting


def watch(args):
    from shaping_rescue_host_launch import launch
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'OWNER.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), outer=args.outer, training_replayed=False, driver_sha256=sha(__file__))
        def save(stage, **values):
            status.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **values)
            (args.output / 'STATUS.json').write_bytes(encoded(status))
        try:
            admission = args.study / 'selection' / f'outer-{args.outer}' / 'OUTER_ADMISSION.json'
            while not admission.exists():
                save('waiting_for_complete_inner_admission')
                time.sleep(30)
            # The outer owner submits its scientifically higher-priority fits first.
            while True:
                row = json.loads(args.outer_status.read_text()) if args.outer_status.exists() else {}
                if row.get('stage') == 'failed_with_preserved_evidence':
                    raise RuntimeError('outer dispatch failed; preserve its evidence')
                if row.get('stage') in ('all_outer_fits_submitted', 'collecting_outer_controls',
                                        'all_outer_fits_submitted_and_controls_collected'):
                    break
                save('waiting_for_outer_fit_submission')
                time.sleep(30)
            all_jobs = []
            for stage in STAGES:
                save('registering_admitted_variants', variant_stage=stage)
                wave, jobs = register_fits(args, stage)
                for host in HOST_SLOTS:
                    if any(j['host'] == host for j in jobs):
                        save('launching_admitted_variants', variant_stage=stage, host=host)
                        launch([wave], REPO, host, wave / ('launch-' + host))
                all_jobs.extend(jobs)
                while waiting := admission_started(jobs):
                    save('waiting_for_priority_admission', variant_stage=stage, waiting_cells=waiting)
                    time.sleep(30)
            while True:
                completed, required = collect_controls(args, all_jobs)
                save('collecting_variant_controls', required_models=len(all_jobs),
                     collected_controls=completed, required_controls=required)
                if len(completed) == required:
                    save('all_variants_submitted_and_controls_collected', required_models=len(all_jobs))
                    return
                time.sleep(30)
        except Exception as error:
            save('failed_with_preserved_evidence', error=str(error))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'output', 'binaries', 'qualification', 'human-qualification', 'outer-status'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--outer', type=int, choices=range(3), required=True)
    watch(p.parse_args())
