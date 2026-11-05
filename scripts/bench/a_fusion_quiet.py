#!/usr/bin/env python3
"""Admit one short host batch only after load and foreign-work checks pass."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket

from a_family_cost_admit import descendant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert socket.gethostname() in ('dev', 'i270')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    foreign = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(proc.name)
            name = (proc / 'comm').read_text().strip()
            executable = Path((proc / 'cmdline').read_bytes().split(b'\0')[0].decode()).name
            heavy = name in {'rustc', 'cargo', 'perf', 'zensim_mlp_trai'} or executable == 'zensim_mlp_train'
            if heavy and not descendant(pid, os.getpid()):
                foreign.append(dict(pid=pid, name=name, executable=executable))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    load = os.getloadavg()[0]
    row = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               host=socket.gethostname(), load1=load, foreign=foreign,
               ready=load < 2 and not foreign, command=command)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)
    if not row['ready']:
        raise SystemExit(75)
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
