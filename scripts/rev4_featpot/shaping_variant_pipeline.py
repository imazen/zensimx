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
from shaping_outer_pipeline import HOST_SLOTS, placements, collect_controls, worker_slots

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
    wave.mkdir(exist_ok=getattr(args, "resume", False))
    saved = wave / "REGISTRATION.json"
    if saved.exists():
        record = json.loads(saved.read_text())
        if (record["admission_sha256"] != sha(admission)
                or {j["cell"]["id"] for j in record["jobs"]} != {c["id"] for c in cells}
                or any(sha(j["request"]) != j["request_sha256"] for j in record["jobs"])):
            raise ValueError("preserve changed variant registration")
        return wave, record["jobs"]
    from shaping_placement import loads
    live_loads, observation = loads(args.study, worker_slots(args))
    (wave / "PLACEMENT_OBSERVATION.json").write_bytes(encoded(observation))
    jobs = []
    remaining = []
    for cell in cells:
        candidates = list((args.study / "downstream_assignments").glob("variant-" + cell["id"] + "-*.json"))
        if not candidates:
            remaining.append(cell)
            continue
        if len(candidates) != 1:
            raise ValueError("preserve ambiguous existing variant assignments")
        assignment = candidates[0]
        a = json.loads(assignment.read_text())
        request = Path(a["worker_request_path"]) if "worker_request_path" in a else wave / "jobs" / cell["id"] / "REQUEST.json"
        q = json.loads(request.read_text())
        if (q["cell"] != cell or a["host"] != q["host"] or a["host"] not in worker_slots(args)
                or a["cells"] != [cell["id"]]
                or any(a[k] != q[k] for k in ("plan_sha256", "build_sha256"))
                or ("worker_request_sha256" in a and a["worker_request_sha256"] != sha(request))):
            raise ValueError("preserve conflicting existing variant request")
        jobs.append(dict(cell=cell, host=a["host"], request=str(request), request_sha256=sha(request),
                         assignment=str(assignment), assignment_sha256=sha(assignment), source=q["source_root"]))
        live_loads[a["host"]] += 1
    for cell, host in placements(remaining, args.outer, live_loads):
        directory = wave / "jobs" / cell["id"]
        if cell["family"] in ("extra_trees", "shaped_additive"):
            register_control(args.study, cell["id"], args.binaries, host, directory)
        else:
            register(args.study, cell["id"], host, args.binaries, args.qualification, directory,
                     execution_kind="variant", human_qualification=args.human_qualification)
        request = directory / "REQUEST.json"
        assignment = args.study / "downstream_assignments" / ("variant-" + cell["id"] + "-" + host + ".json")
        with assignment.open("xb") as stream:
            stream.write((directory / "ASSIGNMENT.json").read_bytes())
        jobs.append(dict(cell=cell, host=host, request=str(request), request_sha256=sha(request),
                         assignment=str(assignment), assignment_sha256=sha(assignment), source=str(REPO)))
    record = dict(stage="registered", source=str(REPO), jobs=jobs, training_replayed=False,
                  source_driver_sha256=sha(REPO / "scripts/rev4_featpot/shaping_rescue_worker.py"),
                  admission_sha256=sha(admission), variant_stage=stage)
    saved.write_bytes(encoded(record))
    return wave, jobs


def launch_registered(args, wave, jobs, save):
    from shaping_rescue_host_launch import launch
    for source in sorted({job.get("source", str(REPO)) for job in jobs}):
        subset = [j for j in jobs if j.get("source", str(REPO)) == source]
        part = wave / ("source-" + __import__("hashlib").sha256(source.encode()).hexdigest()[:12])
        part.mkdir(exist_ok=True)
        registration = dict(stage="registered", jobs=subset,
            source_driver_sha256=sha(Path(source) / "scripts/rev4_featpot/shaping_rescue_worker.py"))
        path = part / "REGISTRATION.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("preserve different source-specific variant wave")
        if not path.exists():
            path.write_bytes(encoded(registration))
        for host in worker_slots(args):
            required = [j for j in subset if j["host"] == host]
            if not required:
                continue
            output = part / ("launch-" + host)
            receipt = output / "LAUNCH.json"
            if receipt.exists():
                rows = json.loads(receipt.read_text())["rows"]
                if {r["cell"]["id"] for r in rows} != {j["cell"]["id"] for j in required}:
                    raise ValueError("preserve different completed host launch")
                continue
            if output.exists():
                raise ValueError("partial host submission needs process reconciliation; never replay")
            save("launching_admitted_variants", host=host)
            launch([part], Path(source), host, output)


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
    args.output.mkdir(parents=True, exist_ok=getattr(args, "resume", False))
    with (args.output / 'OWNER.lock').open('a' if getattr(args,'resume',False) else 'x') as lock:
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
            for stage in (getattr(args, 'stages', None) or STAGES):
                save('registering_admitted_variants', variant_stage=stage)
                wave, jobs = register_fits(args, stage)
                launch_registered(args, wave, jobs, save)
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker-host", action="append", dest="worker_hosts", choices=tuple(HOST_SLOTS))
    p.add_argument("--stage", action="append", dest="stages", choices=STAGES)
    watch(p.parse_args())
