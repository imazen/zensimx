"""Spread every registered conditional exposure extension without changing it."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
from shaping_completion import REPO, encoded, sha


def watch(args):
    from shaping_extension import resolve
    from shaping_rescue_worker import register
    from shaping_outer_pipeline import placements, HOST_SLOTS
    from shaping_rescue_host_launch import launch
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'OWNER.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(pid=os.getpid(), outer=args.outer, driver_sha256=sha(__file__), training_replayed=False)
        def save(stage, **values):
            status.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **values)
            (args.output / 'STATUS.json').write_bytes(encoded(status))
        path = args.study / 'extensions' / f'outer-{args.outer}' / 'REGISTRATION.json'
        try:
            while not path.exists():
                save('waiting_for_complete_paired_rescue_selection')
                time.sleep(30)
            q = json.loads(path.read_text())
            cells = q['cells']
            for cell in cells:
                if resolve(args.study, cell['id'])['cell'] != cell:
                    raise ValueError('conditional extension registration differs')
            save('registering_existing_extension_conditions', registered_cells=len(cells), registration_sha256=sha(path))
            jobs = []
            for cell, host in placements(cells, args.outer):
                directory = args.output / 'jobs' / cell['id']
                register(args.study, cell['id'], host, args.binaries, args.qualification, directory, execution_kind='extension')
                request = directory / 'REQUEST.json'
                assignment = args.study / 'downstream_assignments' / ('extension-' + cell['id'] + '-' + host + '.json')
                if assignment.exists():
                    raise ValueError('preserve existing extension assignment')
                assignment.write_bytes((directory / 'ASSIGNMENT.json').read_bytes())
                jobs.append(dict(cell=cell, host=host, request=str(request), request_sha256=sha(request), assignment=str(assignment), assignment_sha256=sha(assignment)))
            record = dict(stage='registered', source=str(REPO), jobs=jobs, conditional_registration=str(path),
                          conditional_registration_sha256=sha(path), source_driver_sha256=sha(REPO / 'scripts/rev4_featpot/shaping_rescue_worker.py'),
                          training_replayed=False)
            (args.output / 'REGISTRATION.json').write_bytes(encoded(record))
            for host in HOST_SLOTS:
                if any(j['host'] == host for j in jobs):
                    save('launching_conditional_extensions', host=host, required_models=len(jobs))
                    launch([args.output], REPO, host, args.output / ('launch-' + host))
            save('all_registered_extensions_submitted', required_models=len(jobs))
        except Exception as error:
            save('failed_with_preserved_evidence', error=str(error))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'output', 'binaries', 'qualification'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--outer', type=int, choices=range(3), required=True)
    watch(p.parse_args())
