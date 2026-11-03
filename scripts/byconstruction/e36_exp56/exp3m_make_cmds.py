"""Experiment 3 step 3 commands (map sign +1: the first sign -1 batch measured 1-pearson 1.684): p30L's continuation (amendment 2, make_arms2.py) with the map term, on the map
build (bin-map, zensimx f60fa7e5). p30L-mapbuild has no map term and must reproduce amendment 2's p30L-s0."""
import json, os, sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'
O = Path('/mnt/v/zensim/e36-features/2026-11-01')
OUT = H / 'exp3m'
TRAINER = (H / 'bin-map/zensim_mlp_train').resolve()
LEVEL = O / 'level-chroma-32k/manifest.json'
MAP = O / 'map-pool-16/manifest.json'
REF = {0: H / 'control-s0/ckpt/ckpt_epoch059.bin', 1: H / 'arms/control-s1/ckpt/ckpt_epoch059.bin',
       2: H / 'arms/control-s2/ckpt/ckpt_epoch059.bin'}
ARMS = {'p30L-mapbuild': None, 'p30L-map1': 1.0, 'p30L-map10': 10.0, 'p30L-map30': 30.0,
        'p30L-map100': 100.0, 'p30L-map300': 300.0, 'p30L-map1000': 1000.0}
base = json.loads((H / 'control-s0.cmd.json').read_text())


def command(name, seed):
    tag = f'{name}-s{seed}'
    bindir = OUT / f'bin-{tag}'
    bindir.mkdir(parents=True, exist_ok=True)
    for f in (H / 'bin-fast').iterdir():
        if f.name != 'zensim_mlp_train' and not (bindir / f.name).exists():
            os.symlink(f.resolve(), bindir / f.name)
    extra = (f'--tv-pool {O / "pool-F3" / "manifest.json"} --tv-weight 30.0 --tv-apply-every 50 --tv-batch 32 '
             f'--tv-margin 0.05 --level-pool {LEVEL} --level-ref {REF[seed]} --level-weight 1.0 --level-batch 32')
    if ARMS[name] is not None:
        extra += f' --map-pool {MAP} --map-weight {ARMS[name]!r} --map-batch 8 --map-sign 1'
    w = bindir / 'zensim_mlp_train'
    w.write_text(f'#!/bin/sh\n# E36 experiment 3 arm {tag}\nexec {TRAINER} "$@" {extra}\n')
    w.chmod(0o755)
    argv = list(base['argv'])
    argv[argv.index('--seed-index') + 1] = str(seed)
    i = argv.index('--init-from')
    argv[i + 1] = argv[i + 1].replace('/full_s0/', f'/full_s{seed}/')
    argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
    env = dict(base['env'], REV4_V2_BIN_DIR=str(bindir), ZEN_PANEL_BIN=str(bindir / 'panel'),
               TMPDIR=str(OUT / f'scratch/{tag}'))
    (OUT / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
    (OUT / 'cmds').mkdir(exist_ok=True)
    (OUT / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return tag


seed = int(sys.argv[1])
print(' '.join(command(n, seed) for n in (sys.argv[2].split(',') if len(sys.argv) > 2 else ARMS)))
