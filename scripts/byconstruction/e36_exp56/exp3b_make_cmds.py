"""Experiment 3 amendment 2: F3's base fit (120 epochs, from scratch) with the map term alone (map-only pool,
every 50 primary steps, batch 8), on the map-only build (bin-map2). Control: experiment 5's control-s0 base fit."""
import json, os, sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'
OUT = H / 'exp3b'
SRC = H / 'src-exp5/scripts'
TRAINER = (H / 'bin-map2/zensim_mlp_train').resolve()
MAP = Path('/mnt/v/zensim/e36-features/2026-11-01/map-pool-16/manifest.json')
step = json.load(open(Path.home() / 'work/zen/zensim-training/recipes/F3.json'))['steps'][0]
PY = str(Path.home() / '.cache/zensim-training-env/py3147-20261031/bin/python')


def command(weight, seed):
    tag = f'base-map{weight:g}-s{seed}'
    bindir = OUT / f'bin-{tag}'
    bindir.mkdir(parents=True, exist_ok=True)
    for f in (H / 'bin-fast').iterdir():
        if f.name != 'zensim_mlp_train' and not (bindir / f.name).exists():
            os.symlink(f.resolve(), bindir / f.name)
    w = bindir / 'zensim_mlp_train'
    w.write_text(f'#!/bin/sh\n# E36 experiment 3 amendment 2 arm {tag}\nexec {TRAINER} "$@" '
                 f'--map-pool {MAP} --map-weight {weight!r} --map-batch 8 --map-sign 1 --tv-apply-every 50\n')
    w.chmod(0o755)
    argv = [PY if a == '{python}' else a for a in step['argv']]
    argv[2] = str(SRC / 'rev4_featpot/v2_confirm_fit.py')
    argv[argv.index('--seed-index') + 1] = str(seed)
    argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
    env = dict(step['env'], REV4_V2_BIN_DIR=str(bindir), ZEN_PANEL_BIN=str(H / 'bin-fast/panel'),
               BYC_SCRATCH=str(OUT / 'scratch'), TMPDIR=str(OUT / f'scratch/{tag}'),
               PYTHONPATH=f"{SRC / 'rev4_featpot'}:{SRC}")
    (OUT / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
    (OUT / 'cmds').mkdir(exist_ok=True)
    (OUT / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return tag


seed = int(sys.argv[1])
print(' '.join(command(float(w), seed) for w in sys.argv[2].split(',')))
