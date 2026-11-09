"""Measure the frozen study raw read set independently of finalist scoring."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from shaping_completion import encoded, sha

CASES = tuple(f'{size}-{threads}' for size in (256, 1024, 4096) for threads in (1, 4))


def register(study, fixtures, binary, launcher, cpu_tool, output, cases=None):
    from tree_study import RAW_IDS
    cases=list(CASES) if cases is None else list(cases)
    if not cases or len(cases)!=len(set(cases)) or not set(cases)<=set(CASES):
        raise ValueError('distinct registered extraction cases required')
    if output.exists():
        raise ValueError('preserve existing extraction timing registration')
    fixture = json.loads((fixtures/'REGISTRATION.json').read_text())
    pins = {}
    for name in CASES:
        path = fixtures/(name+'.json')
        payload = json.loads(path.read_text())
        if (payload['schema'] != 'shaping-extract-cost-v1' or payload['raw_ids'] != RAW_IDS
                or payload['admission_sha256'] != sha(study/'PLAN.json')
                or str(payload['size'])+'-'+str(payload['threads']) != name
                or sha(path) != fixture['file_sha256'][path.name]):
            raise ValueError('frozen extraction workload differs')
        pins[name+'.json'] = sha(path)
    build = json.loads(binary.with_name('BUILD.json').read_text())
    if sha(binary) != build['binary_sha256']:
        raise ValueError('qualified extraction binary differs')
    output.mkdir(parents=True)
    inputs = output/'inputs'; inputs.mkdir()
    for path in (binary, launcher, cpu_tool):
        shutil.copy2(path, inputs/path.name)
        pins['inputs/'+path.name] = sha(path)
    for name in CASES:
        shutil.copy2(fixtures/(name+'.json'), output/(name+'.json'))
    (output/'BUILD.json').write_bytes(binary.with_name('BUILD.json').read_bytes())
    record = dict(schema='shaping-independent-extraction-cost-v1',
        plan_sha256=sha(study/'PLAN.json'), study=str(study), cases=cases, file_sha256=pins,
        source_fixture_registration_sha256=sha(fixtures/'REGISTRATION.json'),
        build_sha256=sha(output/'BUILD.json'), launcher_source=str(launcher),
        raw_ids=RAW_IDS, raw_ids_owner='tree_study.RAW_IDS',
        maximum_threads=4, native_memory_limit_bytes=4*1024**3,
        scope='Frozen raw read set used by the mandatory tree control and all registered derived neural inputs; extraction only. Finalist vector scoring remains separately gated.',
        scientific_selection=False, outer_results_read=False,
        created_utc=datetime.now(timezone.utc).isoformat(), driver_sha256=sha(__file__))
    (output/'REGISTRATION.json').write_bytes(encoded(record))
    return record


def verify(root, study):
    record = json.loads((root/'REGISTRATION.json').read_text())
    if (record['schema'] != 'shaping-independent-extraction-cost-v1'
            or record['plan_sha256'] != sha(study/'PLAN.json')
            or not record['cases'] or len(record['cases'])!=len(set(record['cases']))
            or not set(record['cases'])<=set(CASES) or record['scientific_selection']
            or record['outer_results_read'] or record['driver_sha256'] != sha(__file__)
            or record['build_sha256'] != sha(root/'BUILD.json')):
        raise ValueError('extraction cost registration differs')
    for name, pin in record['file_sha256'].items():
        if sha(root/name) != pin:
            raise ValueError('registered extraction input differs: '+name)
    return record


def run(root, study, cpus):
    from shaping_reduced_scope import admit_compute
    from shaping_rescue_dispatch import reserve_worker, bind_worker, release_worker
    from shaping_storage import reservation
    admit_compute(study)
    if platform.node().split('.')[0] in ('dev', 'r7900x'):
        raise ValueError('cost protocol requires another eligible benchmark host')
    record=verify(root, study)
    if (root/'STATUS.json').exists():
        raise ValueError('preserve previous timing owner; no implicit replay')
    keys = ['extraction-cost-'+sha(root/'REGISTRATION.json')[:16]+'-'+str(i) for i in range(4)]
    status = dict(pid=os.getpid(), host=platform.node(), stage='waiting_for_admission',
                  registration_sha256=sha(root/'REGISTRATION.json'), completed_cases=[], cpus=cpus)
    def save(stage):
        status.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat())
        (root/'STATUS.json').write_bytes(encoded(status))
    save('waiting_for_admission')
    held = []
    try:
        while True:
            held = []
            for key in keys:
                if reserve_worker(study, key)['can_start']:
                    held.append(key)
                else:
                    break
            if len(held) == len(keys):
                break
            for key in held:
                release_worker(study, key)
            held = []
            time.sleep(10)
        with reservation(study, root, 30*1024**3):
            env = os.environ.copy()
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'CARGO_BUILD_JOBS'):
                env.pop(key, None)
            env.update(TMPDIR=str(Path.home()/'tmp'), PATH=str(root/'inputs')+':'+env['PATH'])
            for name in record['cases']:
                verify(root, study)
                workload = root/(name+'.json'); payload = json.loads(workload.read_text())
                command = [str(Path.home()/'work/claudehints/scripts/run-heavy'), '--mem', '4G',
                    '--jobs', str(payload['threads']), '--', 'bash', str(root/'inputs/profile_test_cpuset.sh'),
                    cpus, 'env', 'ZENSIM_FORMULA_REV=7', str(root/'inputs/shaping_extract_cost'),
                    'bench', str(workload), str(root/('timing-'+name))]
                status.update(case=name, command=command)
                save('measuring')
                with (root/(name+'.log')).open('xb') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    for key in keys:
                        bind_worker(study, key, child.pid)
                    code = child.wait()
                if code:
                    raise RuntimeError('extraction benchmark stopped; preserve '+name+' evidence')
                from shaping_cost_report import report
                report(root/('timing-'+name), workload, root/(name+'-REPORT.json'))
                status['completed_cases'].append(name)
                save('measuring')
            verify(root, study)
            save('complete')
            return status
    except BaseException as error:
        status['error'] = str(error)
        save('failed_with_preserved_evidence')
        raise
    finally:
        for key in held:
            release_worker(study, key)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('register', 'run'))
    for key in ('study', 'root', 'fixtures', 'binary', 'launcher', 'cpu-tool'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--cpus')
    p.add_argument('--case',choices=CASES,action='append',help='register only still-unstarted cases; default all six')
    a = p.parse_args()
    result = register(a.study, a.fixtures, a.binary, a.launcher, a.cpu_tool, a.root,a.case) if a.mode == 'register' else run(a.root, a.study, a.cpus)
    print(json.dumps(dict(stage=result.get('stage', 'registered'), cases=result.get('cases', result.get('completed_cases')))), flush=True)
