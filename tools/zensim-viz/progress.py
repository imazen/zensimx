#!/usr/bin/env python3
"""Observe durable artifacts and recorded live PIDs; never renew locks on a timer."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
scratch = Path.home() / 'tmp/zensim-viz'
output = Path(os.environ.get('VIZ_OUTPUT', '/mnt/v/output/zensim/viz/2026-11-04'))
locks = json.loads(subprocess.check_output(['lslocks', '--json', '--notruncate', '--output', 'PID,COMMAND,MODE,PATH']))['locks']
heavy = [v for v in locks if v.get('path') == str(Path.home() / 'tmp/heavy.lock')]
for lock in heavy:
    try:
        lock['args'] = Path(f"/proc/{lock['pid']}/cmdline").read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        lock['args'] = 'process exited during observation'
record = {'last_verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'heavy_lock_observation': heavy, 'runs': [], 'checks': {}}
for folder in ['export-logs', 'hdr-export-logs', 'curve-audit-logs', 'atlas-v2-logs', 'review1-logs', 'design2-logs', 'review2-logs', 'review2-final-logs', 'function-v1-logs', 'function-final-logs', 'function-affine-logs']:
    path = output / folder / 'progress.json'
    if not path.exists():
        continue
    run = json.loads(path.read_text())
    pid = run.get('pid')
    live = pid is not None and Path(f'/proc/{pid}').exists()
    run['verified_pid_alive'] = live
    if run['stage'] != 'complete':
        if live:
            held = any(v['pid'] == pid and v['mode'] == 'WRITE' for v in heavy)
            run['observed_stage'] = 'running' if held else 'queued'
            curve = Path(run['out']) / 'progress.json'
            if curve.exists():
                run['curve_progress'] = json.loads(curve.read_text())
        else:
            run['observed_stage'] = 'failed or interrupted; inspect retained log'
    else:
        run['observed_stage'] = 'complete (return code and required artifacts verified)'
    record['runs'].append(run)
for name in ['build', 'test-real', 'clippy', 'library-clippy', 'review2-browser', 'function-browser']:
    path = scratch / f'{name}.log'
    if path.exists():
        text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
        receipts = [line for line in text.splitlines() if line.startswith('run-heavy: done')]
        record['checks'][name] = {'log': str(path), 'receipt': receipts[-1] if receipts else None, 'test_results': [line for line in text.splitlines() if line.startswith('test result:')]}
record['completed_native_jobs'] = [str(p.parent.relative_to(output)) for p in sorted(output.glob('*/complete.json'))]
binary = root / 'target/release/zensim-viz-export'
if binary.exists():
    record['current_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
record['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / 'export').glob('*.rs'))}
(scratch / 'progress.json').write_text(json.dumps(record, indent=2) + '\n')
print('observed', record['last_verified_utc'], '; completed native jobs', len(record['completed_native_jobs']))
for run in record['runs']:
    print(run['job'], run['observed_stage'], 'PID', run.get('pid'), ';', run.get('curve_progress', {}).get('completed', ''), '/', run.get('curve_progress', {}).get('required', ''))
for name, check in record['checks'].items():
    print(name, check['receipt'] or 'no completion receipt yet')
