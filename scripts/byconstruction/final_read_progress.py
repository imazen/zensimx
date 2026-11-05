#!/usr/bin/env python3
"""Observe a final-read process and its logs without opening human targets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    path = args.root / 'PROGRESS.json'
    value = json.loads(path.read_text())
    observation = {'utc': datetime.now(timezone.utc).isoformat(), 'processes': {}}
    pids = {field: value.get(field) for field in ('controller_pid', 'child_pid')}
    controller = pids['controller_pid']
    if controller:
        try:
            children = (Path('/proc') / str(controller) / 'task' / str(controller) / 'children').read_text()
            for child in children.split():
                if int(child) not in pids.values():
                    pids[f'controller_child_{child}'] = int(child)
        except FileNotFoundError:
            pass
    for field, pid in pids.items():
        if pid:
            try:
                command = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
                observation['processes'][field] = dict(pid=pid, live=bool(command),
                    argv=[part.decode(errors='replace') for part in command.split(b'\0') if part])
            except FileNotFoundError:
                observation['processes'][field] = dict(pid=pid, live=False)
    population, action, _ = value['stage'].split(':')
    logs = [args.root / 'logs' / f'{population}-{action}.log']
    if population == 'clic22' and action == 'stage':
        logs.append(args.root / 'stage/clic-2022-perceptual/rsync.log')
    logs.extend((args.root / 'inputs' / population).glob('*.log'))
    observation['log_tails'] = {}
    for log in logs:
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 1000))
                observation['log_tails'][str(log.relative_to(args.root))] = stream.read().decode(errors='replace')
    # Keep observations separate: the stage owner may update PROGRESS concurrently.
    latest = json.loads(path.read_text())
    report = dict(stage=latest['stage'], completed=len(latest['completed_sets']),
        required=len(latest['required_sets']), failure=latest.get('failure'), **observation)
    (args.root / 'LIVE_OBSERVATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
