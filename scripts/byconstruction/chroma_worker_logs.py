#!/usr/bin/env python3
"""Archive stopped workers for explicitly named runs, without capturing credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def archive(out, runs):
    assert not out.exists()
    wanted = {'jobs/'+run for run in runs}
    names = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}']).decode().splitlines()
    records = []
    for name in names:
        if not name.startswith('zen-score-chroma-'):
            continue
        info = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        env = dict(e.split('=', 1) for e in info['Config']['Env'] if '=' in e)
        if env.get('ZEN_RUN') not in wanted:
            continue
        assert not info['State']['Running'], 'stop only this run before archiving: '+name
        records.append((name, info, {k: env[k] for k in ['ZEN_RUN', 'ZEN_MAX_CHUNKS_PER_PASS',
            'ZEN_CHUNK_WALL_SEC', 'ZEN_CORE_OVERSUBSCRIBE'] if k in env}))
    assert records
    out.mkdir(parents=True)
    manifest = []
    for name, info, env in records:
        data = subprocess.check_output(['docker', 'logs', name], stderr=subprocess.STDOUT)
        (out/(name+'.log')).write_bytes(data)
        manifest.append(dict(name=name, run=env['ZEN_RUN'], image=info['Config']['Image'],
            image_id=info['Image'], started=info['State']['StartedAt'], finished=info['State']['FinishedAt'],
            exit_code=info['State']['ExitCode'], oom=info['State']['OOMKilled'], restarts=info['RestartCount'],
            cpus=info['HostConfig']['CpusetCpus'], cpu_shares=info['HostConfig']['CpuShares'],
            memory=info['HostConfig']['Memory'], memory_swap=info['HostConfig']['MemorySwap'],
            runtime_settings=env, log_sha256=hashlib.sha256(data).hexdigest(), log_bytes=len(data)))
    (out/'_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--run', action='append', required=True)
    a = p.parse_args()
    archive(a.out, a.run)
