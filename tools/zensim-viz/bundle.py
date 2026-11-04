#!/usr/bin/env python3
"""Run a caller-owned export plan, one resource-guarded native job at a time.

Python orchestrates files/processes only. All pixels and statistics stay in Rust.
Completed jobs are reusable only with the identical binary and argument list.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import shutil


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    binary = str(Path(plan['binary']).resolve())
    identity = sha(binary)
    frozen = Path(plan['log_directory']) / 'native' / f'zensim-viz-export-{identity}'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        with open(binary, 'rb') as source, frozen.open('xb') as destination:
            shutil.copyfileobj(source, destination)
        frozen.chmod(0o755)
    if sha(frozen) != identity:
        raise RuntimeError('binary changed while freezing; preserve this artifact and retry with a frozen binary')
    scratch = Path.home() / 'tmp/zensim-viz'
    scratch.mkdir(parents=True, exist_ok=True)
    for job in plan['jobs']:
        output = Path(job['out'])
        command = [binary, *job['args'], '--out', str(output)]
        marker = output / 'complete.json'
        expected = {'binary_sha256': identity, 'command': command}
        if marker.exists():
            old = json.loads(marker.read_text())
            if any(old.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f'{output}: completed job identity changed; use a new output directory')
            print(f"verified completed job {job['id']}", flush=True)
            continue
        if output.exists():
            raise RuntimeError(f'{output}: preserve this incomplete run and choose a new output directory')
        log = Path(plan['log_directory']) / f"{job['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        progress = {'stage': 'queued', 'job': job['id'], 'out': str(output), 'log': str(log), 'binary_sha256': identity, 'command': command, 'started': now}
        record = Path(plan['log_directory']) / 'progress.json'
        record.write_text(json.dumps(progress, indent=2) + '\n')
        env = dict(__import__('os').environ, ZENSIM_FORMULA_REV='7', TMPDIR=str(scratch), VIZ_CERTIFIER=plan['certifier'])
        guarded = ['flock', str(Path.home() / 'tmp/heavy.lock'), str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '16G', '--jobs', '8', '--', str(frozen), *command[1:]]
        with log.open('x') as stream:
            child = subprocess.Popen(guarded, stdout=stream, stderr=subprocess.STDOUT, env=env)
            progress['pid'] = child.pid
            record.write_text(json.dumps(progress, indent=2) + '\n')
            print(f"{job['id']}: PID {child.pid}; {log}", flush=True)
            rc = child.wait()
        progress['returncode'] = rc
        progress['stage'] = 'complete' if rc == 0 else 'failed'
        progress['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record.write_text(json.dumps(progress, indent=2) + '\n')
        if rc:
            raise RuntimeError(f"{job['id']} failed ({rc}); inspect {log}")
        for required in job['required']:
            if not (output / required).is_file():
                raise RuntimeError(f'{output}: missing {required}')
        marker.write_text(json.dumps(dict(expected, executed_binary=str(frozen), started=now, finished=progress['finished'], log=str(log)), indent=2) + '\n')
        print(f"completed {job['id']}", flush=True)


if __name__ == '__main__':
    main()
