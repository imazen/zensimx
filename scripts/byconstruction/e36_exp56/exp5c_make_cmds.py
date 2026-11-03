"""Experiment 5 stage 3: F3's continuation from the TID2013 + MDID base fits, with that MOS mix and p30L's terms
(pool-F3 lambda 30, level weight 1 against F3's control at epoch 59), on the map build without the map term."""
import json, os, sys
from pathlib import Path
H = Path.home() / 'tmp/e36test'
O = Path('/mnt/v/zensim/e36-features/2026-11-01')
OUT = H / 'exp5c'
SRC = H / 'src-exp5/scripts'
TRAINER = (H / 'bin-map/zensim_mlp_train').resolve()
LEVEL = O / 'level-chroma-32k/manifest.json'
REF = {0: H / 'control-s0/ckpt/ckpt_epoch059.bin', 1: H / 'arms/control-s1/ckpt/ckpt_epoch059.bin',
       2: H / 'arms/control-s2/ckpt/ckpt_epoch059.bin'}
DS = ['pipal', 'mdid', 'live-md', 'cid-iq', 'cid-oa', 'irccyn-ivc', 'colourlab-rating-variability-2022']
FLAGS = ['--human-sources', 'tid2013', '--exclude-datasets', ','.join(d for d in DS if d != 'mdid'),
         '--mos-share', '36.035']
base = json.loads((H / 'control-s0.cmd.json').read_text())


def command(seed):
    tag = f'tid2013_mdid-p30L-s{seed}'
    bindir = OUT / f'bin-{tag}'
    bindir.mkdir(parents=True, exist_ok=True)
    for f in (H / 'bin-fast').iterdir():
        if f.name != 'zensim_mlp_train' and not (bindir / f.name).exists():
            os.symlink(f.resolve(), bindir / f.name)
    extra = (f'--tv-pool {O / "pool-F3" / "manifest.json"} --tv-weight 30.0 --tv-apply-every 50 --tv-batch 32 '
             f'--tv-margin 0.05 --level-pool {LEVEL} --level-ref {REF[seed]} --level-weight 1.0 --level-batch 32')
    w = bindir / 'zensim_mlp_train'
    w.write_text(f'#!/bin/sh\n# E36 experiment 5 stage 3 arm {tag}\nexec {TRAINER} "$@" {extra}\n')
    w.chmod(0o755)
    argv = list(base['argv'])
    argv[2] = str(SRC / 'rev4_featpot/v2_confirm_fit.py')
    argv[argv.index('--seed-index') + 1] = str(seed)
    init = H / f'exp5/runs/tid2013_mdid-s{seed}/refit/last.bin'
    assert init.is_file(), init
    argv[argv.index('--init-from') + 1] = str(init)
    argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
    argv += FLAGS
    env = dict(base['env'], REV4_V2_BIN_DIR=str(bindir), ZEN_PANEL_BIN=str(bindir / 'panel'),
               TMPDIR=str(OUT / f'scratch/{tag}'), PYTHONPATH=f"{SRC / 'rev4_featpot'}:{SRC}")
    (OUT / f'scratch/{tag}').mkdir(parents=True, exist_ok=True)
    (OUT / 'cmds').mkdir(exist_ok=True)
    (OUT / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return tag


print(' '.join(command(s) for s in (0, 1, 2)))
