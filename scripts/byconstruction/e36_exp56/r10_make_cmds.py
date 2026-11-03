"""R10 (zensimx record, registered 2026-11-03 before any run): the TID2013 + MDID mix on F3's continuation without
p30L's terms. Every command is its twin's, experiment 5 stage 3's mix + p30L (exp5c tid2013_mdid-p30L-s{seed}, and the
AIC2026 fold twins expaic mix-k{k} at seed 0), with the trainer wrapper's hinge and level arguments removed and only
the destination and the scratch dir changed. Prints the continuation tags, then the fold tags, one line each."""
import json
import os
from pathlib import Path
H = Path.home() / 'tmp/e36test'
OUT = H / 'r10'
O = Path('/mnt/v/zensim/e36-features/2026-11-01')


def terms(seed):
    ref = {0: H / 'control-s0/ckpt/ckpt_epoch059.bin', 1: H / 'arms/control-s1/ckpt/ckpt_epoch059.bin',
           2: H / 'arms/control-s2/ckpt/ckpt_epoch059.bin'}[seed]
    return (f' --tv-pool {O / "pool-F3" / "manifest.json"} --tv-weight 30.0 --tv-apply-every 50 --tv-batch 32 '
            f'--tv-margin 0.05 --level-pool {O / "level-chroma-32k" / "manifest.json"} --level-ref {ref} '
            f'--level-weight 1.0 --level-batch 32')


def save(exp, tag, argv, env):
    (H / exp / 'cmds').mkdir(parents=True, exist_ok=True)
    (H / exp / 'scratch' / tag).mkdir(parents=True, exist_ok=True)
    env = dict(env, TMPDIR=str(H / exp / 'scratch' / tag))
    (H / exp / 'cmds' / f'{tag}.cmd.json').write_text(json.dumps(dict(argv=argv, env=env), indent=1) + '\n')
    return f'{exp}:{tag}'


cont, folds = [], []
for seed in (0, 1, 2):
    twin = json.loads((H / 'exp5c/cmds' / f'tid2013_mdid-p30L-s{seed}.cmd.json').read_text())
    tdir = Path(twin['env']['REV4_V2_BIN_DIR'])
    wrapper = (tdir / 'zensim_mlp_train').read_text()
    head = f'#!/bin/sh\n# E36 experiment 5 stage 3 arm tid2013_mdid-p30L-s{seed}\nexec '
    tail = f' "$@"{terms(seed)}\n'
    assert wrapper.startswith(head) and wrapper.endswith(tail), wrapper
    trainer = wrapper[len(head):-len(tail)]
    assert Path(trainer).is_file() and ' ' not in trainer, trainer
    tag = f'mixF3-s{seed}'
    bindir = OUT / f'bin-{tag}'
    bindir.mkdir(parents=True, exist_ok=True)
    for f in tdir.iterdir():
        if f.name != 'zensim_mlp_train' and not (bindir / f.name).exists():
            os.symlink(f.resolve(), bindir / f.name)
    w = bindir / 'zensim_mlp_train'
    w.write_text(f'#!/bin/sh\n# E36 R10 arm {tag}\nexec {trainer} "$@"\n')
    w.chmod(0o755)
    argv = list(twin['argv'])
    argv[argv.index('--dest') + 1] = str(OUT / 'runs' / tag)
    env = dict(twin['env'], REV4_V2_BIN_DIR=str(bindir), ZEN_PANEL_BIN=str(bindir / 'panel'))
    cont.append(save('r10', tag, argv, env))
    if seed == 0:
        for k in range(7):
            fold = json.loads((H / 'expaic/cmds' / f'mix-k{k}.cmd.json').read_text())
            assert fold['env']['REV4_V2_BIN_DIR'] == str(tdir), fold['env']['REV4_V2_BIN_DIR']
            a2 = list(fold['argv'])
            a2[a2.index('--dest') + 1] = str(H / 'expaic' / 'runs' / f'mixF3-k{k}')
            same = [x for i, x in enumerate(a2) if a2[i - 1] not in ('--dest', '--aic-public-train')]
            ref = [x for i, x in enumerate(twin['argv']) if twin['argv'][i - 1] not in ('--dest', '--aic-public-train')]
            assert same == ref, k
            folds.append(save('expaic', f'mixF3-k{k}', a2, dict(fold['env'], REV4_V2_BIN_DIR=str(bindir),
                                                               ZEN_PANEL_BIN=str(bindir / 'panel'))))
print(' '.join(cont))
print(' '.join(folds))
