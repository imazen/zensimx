"""Experiment 6 commands: F3's base-fit step through the experiment copy of the scripts (as experiment 5), with
  prune-*     --human-rows exp6/<arm>.json (bin-fast trainer, build d2e89524)
  strat       bin-fast trainer, --pair-sampling stratified
  walk        bin-walk trainer (the walk build), --pair-sampling walk
  control-wb  bin-walk trainer, uniform: must reproduce the experiment 5 control's checkpoints
The pair-sampling arms run a wrapper that rewrites only the value after --pair-sampling."""
import json, os, sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'; OUT = H / 'exp6'
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
    if name.startswith('prune-'):
        argv += ['--human-rows', str(OUT / f'{name}.json')]
        bd = bindir('bin-fast', 'uniform', tag)
    elif name == 'strat':
        bd = bindir('bin-fast', 'stratified', tag)
    elif name == 'walk':
        bd = bindir('bin-walk2', 'walk', tag)  # bdf7a835: walk admitted on the subjective pair path
    elif name == 'control-wb':
        bd = bindir('bin-walk', 'uniform', tag)
    else:
        raise ValueError(name)
    env = dict(step['env'], REV4_V2_BIN_DIR=str(bd), ZEN_PANEL_BIN=str(H / 'bin-fast/panel'),
               BYC_SCRATCH=str(OUT / 'scratch'), TMPDIR=str(OUT / f'scratch/{tag}'),
               PYTHONPATH=f"{SRC / 'rev4_featpot'}:{SRC}")
    if name in ('walk', 'control-wb'):
        env['ZENSIM_SAMPLE_DIGEST'] = '1'
    (OUT / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
    (OUT / 'cmds').mkdir(exist_ok=True)
    (OUT / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return tag


seed = int(sys.argv[1]); names = sys.argv[2].split(',')
print(' '.join(command(n, seed) for n in names))
