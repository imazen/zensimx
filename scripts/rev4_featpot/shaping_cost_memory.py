"""Measure each native scoring model in its own fresh qualified process."""
from __future__ import annotations
import argparse, copy, json, os, platform, re, subprocess
from pathlib import Path
from shaping_completion import encoded, sha


def peak_rss(text):
    values = re.findall(r'^\s*Maximum resident set size \(kbytes\): (\d+)\s*$', text, re.M)
    exits = re.findall(r'^\s*Exit status: (\d+)\s*$', text, re.M)
    if len(values) != 1 or int(values[0]) <= 0 or exits != ['0']:
        raise ValueError('one successful GNU time maximum-RSS observation required')
    return int(values[0])


def measure(binary, workload, output):
    if output.exists():
        raise ValueError('preserve existing memory measurement')
    binary = binary.resolve(); workload = workload.resolve()
    source = json.loads(workload.read_text())
    if source['schema'] != 'shaping-cost-v1' or len(source['feature_bits']) != 40:
        raise ValueError('native forty-row workload required')
    ids = [a['id'] for a in source['arms']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('unique model identities required')
    pins = {str(binary): sha(binary), str(workload): sha(workload)}
    for arm in source['arms']:
        path = (workload.parent/arm['model']).resolve()
        if sha(path) != arm['sha256']:
            raise ValueError('memory workload model differs')
        arm['model'] = str(path); pins[str(path)] = arm['sha256']
    output.mkdir(parents=True)
    measurements = []
    for i, arm in enumerate(source['arms']):
        # Numeric directories avoid interpreting model labels as filesystem paths.
        root = output/f'model-{i:02}'; root.mkdir()
        payload = copy.deepcopy(source); payload['arms'] = [arm]
        path = root/'WORKLOAD.json'; path.write_bytes(encoded(payload))
        command = ['/usr/bin/time', '-v', '-o', str(root/'time.txt'),
                   str(binary), 'qualify', str(path), str(root/'native')]
        with (root/'process.log').open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, LC_ALL='C'))
        if result.returncode:
            raise ValueError('fresh model process failed; evidence retained at '+str(root))
        native = json.loads((root/'native/RESULT.json').read_text())
        identity = json.loads((root/'native/IDENTITY.json').read_text())
        if (any(native.get(k) is not True for k in ('complete', 'preflight_exact', 'postflight_exact'))
                or native.get('timing_measured') is not False or identity['mode'] != 'qualify'
                or identity['workload_sha256'] != sha(path) or identity['binary_sha256'] != pins[str(binary)]
                or identity['rows_per_call'] != 40 or identity['threads'] != 1):
            raise ValueError('fresh memory process qualification differs')
        measured = peak_rss((root/'time.txt').read_text())
        measurements.append(dict(id=arm['id'], model_sha256=arm['sha256'],
                                 native_model_bytes=Path(arm['model']).stat().st_size,
                                 process_peak_rss_kib=measured, command=command,
                                 dispatch_ceiling=identity['dispatch_ceiling'],
                                 evidence_sha256={str(p.relative_to(output)): sha(p) for p in root.rglob('*') if p.is_file()}))
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError('memory measurement input changed')
    receipt = dict(schema='shaping-cost-memory-v1', source_pins=pins,
                   host=platform.node(), architecture=platform.machine(), measurements=measurements,
                   scope='fresh one-model process maximum RSS including runtime, model loading, forty feature rows, exact preflight and postflight scoring; not isolated model heap or steady-state allocation',
                   latency_measured=False, scientific_finalist_admission_verified=False)
    (output/'MEMORY.json').write_bytes(encoded(receipt))
    return dict(output=str(output), models=len(measurements), receipt_sha256=sha(output/'MEMORY.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'workload', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(measure(a.binary, a.workload, a.output)))
