"""Build a gradient microprobe under shared admission; never change a trainer."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from shaping_completion import encoded, sha
from shaping_rescue_dispatch import reserve_worker, release_worker, bind_worker
from shaping_reduced_scope import admit_compute


def run(args):
    admit_compute(args.study)
    if len(os.sched_getaffinity(0)) < 6:
        raise ValueError('two build jobs require at least six available CPUs')
    args.output.mkdir(parents=True, exist_ok=False)
    keys = ['gradient-probe-build-' + args.output.name + '-' + str(i) for i in range(2)]
    while True:
        acquired = []
        for key in keys:
            if reserve_worker(args.study, key)['can_start']:
                acquired.append(key)
            else:
                break
        if len(acquired) == 2:
            break
        for key in acquired:
            release_worker(args.study, key)
        time.sleep(10)
    env = os.environ.copy()
    env.update(TMPDIR=str(Path.home() / 'tmp'), CARGO_TARGET_DIR=str(args.target))
    heavy = Path.home() / 'work/claudehints/scripts/run-heavy'
    record = dict(pid=os.getpid(), stage='building', started_utc=datetime.now(timezone.utc).isoformat(),
                  training_changed=False, source=str(args.source), target=str(args.target), commands=[])
    identity = args.output / 'BUILD.json'
    try:
        for mode in ('test', 'clippy', 'build'):
            argv = [str(heavy), '--mem', '4G', '--jobs', '2', '--', 'cargo', mode, '--locked', '--release',
                    '-p', 'zensim-validate', '--example', 'shaping_gradient_cost']
            if mode == 'clippy':
                argv.extend(['--', '-D', 'warnings'])
            record['commands'].append(argv); identity.write_bytes(encoded(record))
            with (args.output / (mode + '.log')).open('x') as log:
                child = subprocess.Popen(argv, cwd=args.source, env=env, stdout=log, stderr=subprocess.STDOUT)
                for key in keys:
                    bind_worker(args.study, key, child.pid)
                code = child.wait()
            if code:
                raise RuntimeError(mode + ' failed; preserve the build log')
        source = args.target / 'release/examples/shaping_gradient_cost'
        binary = args.output / 'shaping-gradient-cost'
        shutil.copy2(source, binary)
        record.update(stage='built', binary_sha256=sha(binary), source_sha256=sha(args.source / 'zensim-validate/examples/shaping_gradient_cost.rs'))
    except Exception as error:
        record.update(stage='failed', error=str(error))
        raise
    finally:
        record['checked_utc'] = datetime.now(timezone.utc).isoformat(); identity.write_bytes(encoded(record))
        for key in acquired:
            release_worker(args.study, key)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('study', 'source', 'target', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())
