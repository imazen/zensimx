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
    # nproc honors OMP_NUM_THREADS. Let run-heavy resolve actual CPUs before
    # installing its explicit two-job limits, independent of imported ML code.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'CARGO_BUILD_JOBS'):
        env.pop(key, None)
    env.update(TMPDIR=str(Path.home() / 'tmp'), CARGO_TARGET_DIR=str(args.target))
    heavy = Path.home() / 'work/claudehints/scripts/run-heavy'
    record = dict(pid=os.getpid(), stage='building', started_utc=datetime.now(timezone.utc).isoformat(),
                  training_changed=False, source=str(args.source), target=str(args.target), commands=[])
    identity = args.output / 'BUILD.json'
    try:
        tasks = [('test', 'test', ['--example', 'shaping_gradient_cost']),
                 ('clippy', 'clippy', ['--example', 'shaping_gradient_cost', '--', '-D', 'warnings']),
                 ('build', 'build', ['--example', 'shaping_gradient_cost'])]
        if args.trainer:
            tasks = [('shape-tests', 'test', ['--lib', '--', 'mlp_train::input_shaping::tests']),
                     ('dot-tests', 'test', ['--example', 'shaping_gradient_cost']),
                     ('clippy', 'clippy', ['--bin', 'zensim_mlp_train', '--example', 'shaping_gradient_cost', '--', '-D', 'warnings']),
                     ('build', 'build', ['--bin', 'zensim_mlp_train'])]
            if args.shape_tests_only:
                tasks = tasks[:1]
        for label, mode, targets in tasks:
            argv = [str(heavy), '--mem', '4G', '--jobs', '2', '--', 'cargo', mode, '--locked', '--release',
                    '-p', 'zensim-validate', *targets]
            record['commands'].append(argv); identity.write_bytes(encoded(record))
            with (args.output / (label + '.log')).open('x') as log:
                child = subprocess.Popen(argv, cwd=args.source, env=env, stdout=log, stderr=subprocess.STDOUT)
                for key in keys:
                    bind_worker(args.study, key, child.pid)
                code = child.wait()
            if code:
                raise RuntimeError(mode + ' failed; preserve the build log')
        source = args.target / ('release/zensim_mlp_train' if args.trainer else 'release/examples/shaping_gradient_cost')
        binary = args.output / ('zensim_mlp_train' if args.trainer else 'shaping-gradient-cost')
        shutil.copy2(source, binary)
        source_names = ['zensim-validate/examples/shaping_gradient_cost.rs']
        if args.trainer:
            source_names += ['zensim-validate/src/mlp_train/' + name for name in ('mod.rs', 'input_shaping.rs', 'paired_gradient.rs')]
        record.update(stage='built', binary_sha256=sha(binary), trainer_candidate=args.trainer,
                      source_sha256=sha(args.source / source_names[0]),
                      source_files_sha256={name: sha(args.source / name) for name in source_names})
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
    p.add_argument('--trainer', action='store_true', help='build a candidate trainer for separate full-trajectory qualification')
    p.add_argument('--shape-tests-only', action='store_true', help='run the library shape tests against an already built candidate')
    a = p.parse_args()
    if a.shape_tests_only and not a.trainer:
        p.error('--shape-tests-only requires --trainer')
    run(a)
