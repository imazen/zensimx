"""Revised plan R2 (method review M3) commands: F3's base-fit step through the experiment copy of the scripts, with
  m3c, m3cn   --human-rows exp7/<arm>.json (zensimx scripts/byconstruction/e36_exp56/m3_rows.py), walk sampler
              (bin-walk2 trainer, build bdf7a835), as experiment 6's walk arm, which is the control.
The wrapper rewrites only the value after --pair-sampling."""
import json, os, sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'; OUT = H / 'exp7'
SRC = H / 'src-exp5/scripts'
step = json.load(open(Path.home() / 'work/zen/zensim-training/recipes/F3.json'))['steps'][0]
PY = str(Path.home() / '.cache/zensim-training-env/py3147-20261031/bin/python')


def bindir(trainer_dir, mode, tag):
    if mode == 'uniform' and trainer_dir == 'bin-fast':
        return H / 'bin-fast'
    d = OUT / f'bin-{tag}'
    d.mkdir(parents=True, exist_ok=True)
    for f in (H / 'bin-fast').iterdir():
        if f.name != 'zensim_mlp_train' and not (d / f.name).exists():
            os.symlink(f.resolve(), d / f.name)
    trainer = (H / trainer_dir / 'zensim_mlp_train').resolve()
    w = d / 'zensim_mlp_train'
    w.write_text(f'''#!/bin/sh
# E36 experiment 6 arm {tag}: {trainer} with --pair-sampling {mode}
n=$#; prev=
for a in "$@"; do
  if [ "$prev" = "--pair-sampling" ]; then a={mode}; fi
  set -- "$@" "$a"; prev=$a
done
shift $n
exec {trainer} "$@"
''')
    w.chmod(0o755)
    return d


def command(name, seed):
    tag = f'{name}-s{seed}'
    dest = OUT / 'runs' / tag
    argv = [PY if a == '{python}' else a for a in step['argv']]
    argv[2] = str(SRC / 'rev4_featpot/v2_confirm_fit.py')
    argv[argv.index('--seed-index') + 1] = str(seed)
    argv[argv.index('--dest') + 1] = str(dest)
    if name in ('m3c', 'm3cn'):
        argv += ['--human-rows', str(OUT / f'{name}.json')]
        bd = bindir('bin-walk2', 'walk', tag)
    else:
        raise ValueError(name)
    env = dict(step['env'], REV4_V2_BIN_DIR=str(bd), ZEN_PANEL_BIN=str(H / 'bin-fast/panel'),
               BYC_SCRATCH=str(OUT / 'scratch'), TMPDIR=str(OUT / f'scratch/{tag}'),
               PYTHONPATH=f"{SRC / 'rev4_featpot'}:{SRC}")
    env['ZENSIM_SAMPLE_DIGEST'] = '1'
    (OUT / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
    (OUT / 'cmds').mkdir(exist_ok=True)
    (OUT / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return tag


seed = int(sys.argv[1]); names = sys.argv[2].split(',')
print(' '.join(command(n, seed) for n in names))
